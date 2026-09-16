import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

from engine.audio import (
    LOUDNESS_TARGET_I,
    LOUDNESS_UNDERSHOOT_TOLERANCE,
    AudioGenerator,
    measure_integrated_loudness,
    normalize_loudness,
    parse_directive,
)


def _make_tone_mp3(path, volume_db, duration=2.0):
    """A real MP3 sine tone at a chosen level, for loudness assertions."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:sample_rate=44100:duration={duration}",
            "-af", f"volume={volume_db}dB",
            "-c:a", "libmp3lame", str(path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return path


def _fake_raw_response(chunks, request_id="fake-request-id"):
    """A context-manager mock matching client.text_to_speech.with_raw_response
    .convert()'s shape: `with ... as response: response.headers, response.data`."""
    response = MagicMock()
    response.headers = {"request-id": request_id}
    response.data = chunks
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


def test_parse_directive_calm_curious_lowers_stability_variance():
    settings = parse_directive("Calm, curious, a little conspiratorial. Unhurried.")
    assert settings.stability == 0.70
    assert settings.style == 0.0


def test_parse_directive_default_when_no_keywords_match():
    settings = parse_directive("")
    assert settings.stability == 0.55
    assert settings.similarity_boost == 0.80


def test_audio_generator_requires_voice_id():
    with pytest.raises(ValueError, match="voice_id is required"):
        AudioGenerator(api_key="key", voice_id="")


def test_generate_segment_rejects_empty_text(tmp_path):
    with patch("engine.audio.ElevenLabs"):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        with pytest.raises(ValueError, match="empty text"):
            gen.generate_segment("   ", tmp_path / "out.mp3")


def test_generate_segment_writes_audio_bytes(tmp_path):
    fake_client = MagicMock()
    fake_client.text_to_speech.with_raw_response.convert.return_value = _fake_raw_response([b"x" * 200])

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out, directive="calm")

    assert result.path == out
    assert out.read_bytes() == b"x" * 200


def test_generate_segment_returns_request_id_from_response_header(tmp_path):
    fake_client = MagicMock()
    fake_client.text_to_speech.with_raw_response.convert.return_value = _fake_raw_response(
        [b"x" * 200], request_id="req-abc-123",
    )

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        result = gen.generate_segment("Hello there.", tmp_path / "audio_00.mp3")

    assert result.request_id == "req-abc-123"


def test_generate_segment_passes_previous_request_ids_to_api(tmp_path):
    """ElevenLabs 'request stitching': conditioning a segment's generation on
    the request IDs of the segments immediately before it measurably improves
    cross-segment prosody continuity for a script split into many calls."""
    fake_client = MagicMock()
    fake_client.text_to_speech.with_raw_response.convert.return_value = _fake_raw_response([b"x" * 200])

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        gen.generate_segment(
            "Hello there.", tmp_path / "audio_01.mp3", previous_request_ids=["req-000"],
        )

    _, call_kwargs = fake_client.text_to_speech.with_raw_response.convert.call_args
    assert call_kwargs["previous_request_ids"] == ["req-000"]


def test_generate_segment_omits_previous_request_ids_when_none_given(tmp_path):
    """The first segment in a script has no prior segment to stitch from --
    the API kwarg must be omitted entirely (not sent as an empty list or
    null), matching the SDK's own OMIT-by-default semantics."""
    fake_client = MagicMock()
    fake_client.text_to_speech.with_raw_response.convert.return_value = _fake_raw_response([b"x" * 200])

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        gen.generate_segment("Hello there.", tmp_path / "audio_00.mp3")

    _, call_kwargs = fake_client.text_to_speech.with_raw_response.convert.call_args
    assert "previous_request_ids" not in call_kwargs


def test_generate_segment_retries_on_network_error_then_succeeds(tmp_path):
    fake_client = MagicMock()
    fake_client.text_to_speech.with_raw_response.convert.side_effect = [
        ConnectionError("getaddrinfo failed"),
        _fake_raw_response([b"x" * 200]),
    ]

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p), \
         patch("engine.audio.time.sleep") as mock_sleep:
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out)

    assert result.path == out
    assert out.read_bytes() == b"x" * 200
    mock_sleep.assert_called_once()


def test_normalize_loudness_equalizes_segment_levels(tmp_path):
    """Regression test: every segment is a separate ElevenLabs generation and
    generations come back at wildly different levels (a real render measured
    adjacent segments ~15-20 LU apart), so the assembled video's voiceover
    jumps in volume. Each segment must be normalized to one integrated
    loudness target before assembly."""
    # ffmpeg's lavfi sine sits around -22 LUFS at 0dB, so these land near
    # -34 and -16 LUFS — well apart, and well above the silence floor.
    quiet = _make_tone_mp3(tmp_path / "quiet.mp3", volume_db=-12)
    loud = _make_tone_mp3(tmp_path / "loud.mp3", volume_db=6)

    assert measure_integrated_loudness(loud) - measure_integrated_loudness(quiet) > 15

    normalize_loudness(quiet)
    normalize_loudness(loud)

    assert measure_integrated_loudness(quiet) == pytest.approx(LOUDNESS_TARGET_I, abs=1.0)
    assert measure_integrated_loudness(loud) == pytest.approx(LOUDNESS_TARGET_I, abs=1.0)


def test_normalize_loudness_falls_back_to_limiter_when_linear_undershoots(tmp_path):
    """Regression test using a real ElevenLabs take checked in as a fixture
    (tests/fixtures/undershoot_segment.mp3): a real render measured this
    segment stuck at -21.76 LUFS (target -16.0) even after normalize_loudness
    ran -- traced to linear-mode loudnorm capping its gain at the true-peak
    ceiling on a short, transient-heavy segment (one hot consonant burst
    against an otherwise quiet body). Neither linear nor dynamic loudnorm
    mode can fix this (there is no exploitable dynamic range within the
    segment for either to differentially compress), so normalize_loudness
    must detect the undershoot and fall back to an explicit gain-up +
    true-peak limiter.

    A synthetic (sine-wave burst + tone tail) fixture was tried first, but
    EBU R128's relative gating algorithm doesn't behave representatively on a
    near-total-silence-plus-one-tone-blip shape (unlike real continuous
    speech) -- the checked-in real segment is what actually reproduces the bug.
    """
    audio_path = tmp_path / "undershoot_segment.mp3"
    shutil.copy2(FIXTURES_DIR / "undershoot_segment.mp3", audio_path)
    raw_i = measure_integrated_loudness(audio_path)
    assert raw_i < LOUDNESS_TARGET_I - LOUDNESS_UNDERSHOOT_TOLERANCE * 2  # sanity: fixture is genuinely far off

    normalize_loudness(audio_path)

    assert measure_integrated_loudness(audio_path) == pytest.approx(LOUDNESS_TARGET_I, abs=1.5)


def test_normalize_loudness_leaves_silence_untouched(tmp_path):
    """Near-silent audio has no meaningful integrated loudness — applying a
    huge make-up gain would just amplify the noise floor. Leave it as-is."""
    silent = tmp_path / "silent.mp3"
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
         "-t", "1", str(silent)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    original_bytes = silent.read_bytes()

    returned = normalize_loudness(silent)

    assert returned == silent
    assert silent.read_bytes() == original_bytes


def test_generate_segment_normalizes_generated_audio(tmp_path):
    """generate_segment must hand back loudness-normalized audio, so every
    downstream consumer (main video, Shorts) gets consistent levels."""
    tone_bytes = _make_tone_mp3(tmp_path / "api_response.mp3", volume_db=-12).read_bytes()
    fake_client = MagicMock()
    fake_client.text_to_speech.with_raw_response.convert.return_value = _fake_raw_response([tone_bytes])

    with patch("engine.audio.ElevenLabs", return_value=fake_client):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        result = gen.generate_segment("Hello there.", tmp_path / "audio_00.mp3")

    assert measure_integrated_loudness(result.path) == pytest.approx(LOUDNESS_TARGET_I, abs=1.0)


def test_probe_duration_parses_ffprobe_json(tmp_path):
    fake_result = MagicMock(returncode=0, stdout=json.dumps({"format": {"duration": "12.345"}}))
    with patch("engine.audio.ElevenLabs"), patch("engine.audio.subprocess.run", return_value=fake_result):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        duration = gen.probe_duration(tmp_path / "audio_00.mp3")
    assert duration == pytest.approx(12.345)
