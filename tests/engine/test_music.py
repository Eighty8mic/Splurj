import subprocess
from unittest.mock import MagicMock, patch

import pytest

from engine.audio import LOUDNESS_TARGET_I, measure_integrated_loudness
from engine.music import MusicGenerator


def _tone_mp3_bytes(volume_db, duration=2.0):
    """Real MP3 sine-tone bytes at a chosen level, standing in for a music API response."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"sine=frequency=220:sample_rate=44100:duration={duration}",
            "-af", f"volume={volume_db}dB",
            "-c:a", "libmp3lame", "-f", "mp3", "pipe:1",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_music_generator_requires_api_key():
    with pytest.raises(EnvironmentError, match="ELEVENLABS_API_KEY"):
        MusicGenerator(api_key="")


def test_compose_writes_audio_and_normalizes_loudness(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.music._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.music.compose.return_value = iter([_tone_mp3_bytes(volume_db=-6)])

    with patch("engine.music.ElevenLabs", return_value=fake_client):
        gen = MusicGenerator(api_key="key")
        out = gen.compose("calm lo-fi bed, no vocals", duration_ms=8000, output_path=tmp_path / "bed.mp3")

    assert out.exists()
    assert measure_integrated_loudness(out) == pytest.approx(LOUDNESS_TARGET_I, abs=1.0)


def test_compose_requests_instrumental_music(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.music._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.music.compose.return_value = iter([_tone_mp3_bytes(volume_db=-6)])

    with patch("engine.music.ElevenLabs", return_value=fake_client):
        gen = MusicGenerator(api_key="key")
        gen.compose("calm lo-fi bed", duration_ms=8000, output_path=tmp_path / "bed.mp3")

    _, kwargs = fake_client.music.compose.call_args
    assert kwargs["force_instrumental"] is True
    assert kwargs["music_length_ms"] == 8000


def test_compose_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.music._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.music.compose.return_value = iter([_tone_mp3_bytes(volume_db=-6)])

    with patch("engine.music.ElevenLabs", return_value=fake_client):
        gen = MusicGenerator(api_key="key")
        gen.compose("calm lo-fi bed", duration_ms=8000, output_path=tmp_path / "first.mp3")
        gen.compose("calm lo-fi bed", duration_ms=8000, output_path=tmp_path / "second.mp3")

    assert fake_client.music.compose.call_count == 1
    assert (tmp_path / "second.mp3").exists()


def test_compose_retries_on_network_error_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.music._CACHE_DIR", tmp_path / "cache")
    fake_client = MagicMock()
    fake_client.music.compose.side_effect = [
        ConnectionError("getaddrinfo failed"),
        iter([_tone_mp3_bytes(volume_db=-6)]),
    ]

    with patch("engine.music.ElevenLabs", return_value=fake_client), \
         patch("engine.music.time.sleep") as mock_sleep:
        gen = MusicGenerator(api_key="key")
        out = gen.compose("calm lo-fi bed", duration_ms=8000, output_path=tmp_path / "bed.mp3")

    assert out.exists()
    mock_sleep.assert_called_once()
