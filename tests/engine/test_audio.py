import json
from unittest.mock import MagicMock, patch

import pytest

from engine.audio import AudioGenerator, parse_directive


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

    with patch("engine.audio.ElevenLabs", return_value=fake_client):
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
         patch("engine.audio.time.sleep") as mock_sleep:
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out)

    assert result == out
    assert out.read_bytes() == b"x" * 200
    mock_sleep.assert_called_once()


def test_probe_duration_parses_ffprobe_json(tmp_path):
    fake_result = MagicMock(returncode=0, stdout=json.dumps({"format": {"duration": "12.345"}}))
    with patch("engine.audio.ElevenLabs"), patch("engine.audio.subprocess.run", return_value=fake_result):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        duration = gen.probe_duration(tmp_path / "audio_00.mp3")
    assert duration == pytest.approx(12.345)
