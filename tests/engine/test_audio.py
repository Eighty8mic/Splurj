import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from engine.audio import (
    LOUDNESS_TARGET_I,
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
    fake_client.text_to_speech.convert.return_value = [b"x" * 200]

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out, directive="calm")

    assert result == out
    assert out.read_bytes() == b"x" * 200


def test_generate_segment_retries_on_network_error_then_succeeds(tmp_path):
    fake_client = MagicMock()
    fake_client.text_to_speech.convert.side_effect = [
        ConnectionError("getaddrinfo failed"),
        [b"x" * 200],
    ]

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.normalize_loudness", side_effect=lambda p, **kw: p), \
         patch("engine.audio.time.sleep") as mock_sleep:
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out)

    assert result == out
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
    fake_client.text_to_speech.convert.return_value = [tone_bytes]

    with patch("engine.audio.ElevenLabs", return_value=fake_client):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = gen.generate_segment("Hello there.", tmp_path / "audio_00.mp3")

    assert measure_integrated_loudness(out) == pytest.approx(LOUDNESS_TARGET_I, abs=1.0)


def test_probe_duration_parses_ffprobe_json(tmp_path):
    fake_result = MagicMock(returncode=0, stdout=json.dumps({"format": {"duration": "12.345"}}))
    with patch("engine.audio.ElevenLabs"), patch("engine.audio.subprocess.run", return_value=fake_result):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        duration = gen.probe_duration(tmp_path / "audio_00.mp3")
    assert duration == pytest.approx(12.345)
