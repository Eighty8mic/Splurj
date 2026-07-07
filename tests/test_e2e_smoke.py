import shutil
import sys
from unittest.mock import MagicMock, patch

from engine.video import probe_video_resolution
from splurj_engine import load_blueprint, main


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr("splurj_engine.WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr("splurj_engine.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("splurj_engine.ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr("splurj_engine.CHANNEL_DATA", tmp_path / "channel_data")
    (tmp_path / "assets" / "ambient").mkdir(parents=True)
    (tmp_path / "channel_data").mkdir(parents=True)


def _patch_generators(fixture_image, fixture_audio):
    fake_audio_gen = MagicMock()
    fake_audio_gen.generate_segment.side_effect = (
        lambda text, output_path, directive="", max_retries=4: shutil.copy2(fixture_audio, output_path) or output_path
    )
    fake_audio_gen.probe_duration.side_effect = lambda audio_path: 1.0

    fake_image_gen = MagicMock()
    fake_image_gen.generate.side_effect = (
        lambda prompt, output_path, reference_image_path=None, max_retries=4: shutil.copy2(fixture_image, output_path) or output_path
    )
    return fake_audio_gen, fake_image_gen


def test_content_example_loads_and_validates():
    blueprints = load_blueprint("content_example.json")
    assert len(blueprints) == 1
    assert len(blueprints[0]["timeline"]) == 6


def test_cli_main_no_upload_renders_and_cleans_workspace(tmp_path, fixture_image, fixture_audio, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    fake_audio_gen, fake_image_gen = _patch_generators(fixture_image, fixture_audio)

    for key, val in {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }.items():
        monkeypatch.setenv(key, val)

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch.object(sys, "argv", ["splurj_engine.py", "--input", "content_example.json", "--no-upload"]):
        main()

    output_files = list((tmp_path / "output").glob("day_001_*.mp4"))
    assert len(output_files) == 1
    assert probe_video_resolution(output_files[0]) == (1920, 1080)
    assert not (tmp_path / "workspace" / "day_001").exists()  # cleaned up by default


def test_cli_main_keep_workspace_preserves_temp_files(tmp_path, fixture_image, fixture_audio, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    fake_audio_gen, fake_image_gen = _patch_generators(fixture_image, fixture_audio)

    for key, val in {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }.items():
        monkeypatch.setenv(key, val)

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch.object(sys, "argv", ["splurj_engine.py", "--input", "content_example.json", "--no-upload", "--keep-workspace"]):
        main()

    assert (tmp_path / "workspace" / "day_001").exists()
