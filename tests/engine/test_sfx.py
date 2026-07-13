import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from engine.audio import measure_integrated_loudness
from engine.sfx import SFX_LOUDNESS_TARGET_I, SoundEffectGenerator, overlay_cues


def _tone_mp3_bytes(volume_db, duration=1.0, frequency=440):
    """Real MP3 sine-tone bytes at a chosen level, standing in for an SFX API response."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:sample_rate=44100:duration={duration}",
            "-af", f"volume={volume_db}dB",
            "-c:a", "libmp3lame", "-f", "mp3", "pipe:1",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _probe_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def test_sound_effect_generator_requires_api_key():
    with pytest.raises(EnvironmentError, match="ELEVENLABS_API_KEY"):
        SoundEffectGenerator(api_key="")


def test_generate_writes_audio_and_normalizes_loudness(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.sfx._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.text_to_sound_effects.convert.return_value = iter([_tone_mp3_bytes(volume_db=-6)])

    with patch("engine.sfx.ElevenLabs", return_value=fake_client):
        gen = SoundEffectGenerator(api_key="key")
        out = gen.generate("a card-reader tap chime", tmp_path / "sfx_00.mp3")

    assert out.exists()
    assert measure_integrated_loudness(out) == pytest.approx(SFX_LOUDNESS_TARGET_I, abs=1.0)


def test_generate_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.sfx._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.text_to_sound_effects.convert.return_value = iter([_tone_mp3_bytes(volume_db=-6)])

    with patch("engine.sfx.ElevenLabs", return_value=fake_client):
        gen = SoundEffectGenerator(api_key="key")
        gen.generate("coin drop", tmp_path / "first.mp3")
        gen.generate("coin drop", tmp_path / "second.mp3")

    assert fake_client.text_to_sound_effects.convert.call_count == 1
    assert (tmp_path / "second.mp3").exists()


def test_generate_retries_on_network_error_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.sfx._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.text_to_sound_effects.convert.side_effect = [
        ConnectionError("getaddrinfo failed"),
        iter([_tone_mp3_bytes(volume_db=-6)]),
    ]

    with patch("engine.sfx.ElevenLabs", return_value=fake_client), \
         patch("engine.sfx.time.sleep") as mock_sleep:
        gen = SoundEffectGenerator(api_key="key")
        out = gen.generate("coin drop", tmp_path / "sfx.mp3")

    assert out.exists()
    mock_sleep.assert_called_once()


def test_overlay_cues_with_no_cues_copies_voice_audio_unchanged(tmp_path):
    voice = tmp_path / "voice.mp3"
    voice.write_bytes(_tone_mp3_bytes(volume_db=-6, duration=2.0))

    out = tmp_path / "mixed.mp3"
    result = overlay_cues(voice, [], out)

    assert result == out
    assert out.read_bytes() == voice.read_bytes()


def test_overlay_cues_mixes_sfx_and_preserves_voice_duration(tmp_path):
    voice = tmp_path / "voice.mp3"
    voice.write_bytes(_tone_mp3_bytes(volume_db=-6, duration=3.0, frequency=440))
    voice_duration = _probe_duration(voice)

    sfx = tmp_path / "sfx.mp3"
    sfx.write_bytes(_tone_mp3_bytes(volume_db=-6, duration=0.5, frequency=880))

    out = tmp_path / "mixed.mp3"
    # A cue starting near the end must not extend the segment past the voice length.
    result = overlay_cues(voice, [(sfx, 2.8)], out)

    assert result == out
    assert out.exists()
    assert _probe_duration(out) == pytest.approx(voice_duration, abs=0.2)
