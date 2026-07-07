import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from splurj_engine import (
    DISCLAIMER,
    build_description,
    build_shorts_title,
    build_thumbnail_prompts,
    load_blueprint,
    load_env,
    run_pipeline,
    slugify,
    _validate_blueprint,
)


def _valid_blueprint(day=1):
    return {
        "day": day,
        "format": "long",
        "metadata": {
            "title": "Why Your Brain Feels No Pain When You Tap a Card",
            "description": "A hook description.",
            "tags": ["money psychology", "splurj"],
        },
        "voiceover": {"directive": "calm, curious", "full_text": "Seg one. Seg two. Seg three. Seg four."},
        "timeline": [
            {"start": 0, "end": 15, "text": "Seg one.", "prompt": "doodle prompt one", "is_short_candidate": True},
            {"start": 15, "end": 30, "text": "Seg two.", "prompt": "doodle prompt two", "is_short_candidate": True},
            {"start": 30, "end": 45, "text": "Seg three.", "prompt": "doodle prompt three", "is_short_candidate": False},
            {"start": 45, "end": 60, "text": "Seg four.", "prompt": "doodle prompt four", "is_short_candidate": False},
        ],
    }


def test_validate_blueprint_accepts_well_formed_blueprint():
    _validate_blueprint(_valid_blueprint())  # must not raise


def test_validate_blueprint_rejects_missing_top_level_key():
    bp = _valid_blueprint()
    del bp["voiceover"]
    with pytest.raises(SystemExit, match="missing required keys"):
        _validate_blueprint(bp)


def test_validate_blueprint_rejects_missing_segment_key():
    bp = _valid_blueprint()
    del bp["timeline"][0]["is_short_candidate"]
    with pytest.raises(SystemExit, match="is_short_candidate"):
        _validate_blueprint(bp)


def test_validate_blueprint_rejects_full_text_mismatch():
    bp = _valid_blueprint()
    bp["voiceover"]["full_text"] = "This does not match the segments at all."
    with pytest.raises(SystemExit, match="full_text"):
        _validate_blueprint(bp)


def test_load_blueprint_returns_list_for_single_object(tmp_path):
    bp_path = tmp_path / "bp.json"
    bp_path.write_text(json.dumps(_valid_blueprint()))
    result = load_blueprint(str(bp_path))
    assert isinstance(result, list)
    assert result[0]["day"] == 1


def test_load_env_raises_when_keys_missing(monkeypatch):
    for key in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "GEMINI_API_KEY", "YOUTUBE_CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit, match="Missing environment variables"):
        load_env()


def test_load_env_returns_dict_when_keys_present(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "a")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "b")
    monkeypatch.setenv("GEMINI_API_KEY", "c")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "d")
    env = load_env()
    assert env["ELEVENLABS_API_KEY"] == "a"


def test_build_description_always_appends_disclaimer():
    result = build_description("A hook description with no disclaimer.", "Full script text.", day=1)
    assert DISCLAIMER in result
    assert "A hook description" in result


def test_build_description_keeps_disclaimer_intact_for_long_scripts():
    # Real Splurj scripts run 1,800-2,500 words (~10-15k chars). The uploader
    # slices description[:5000] before sending to the API, so the disclaimer
    # must survive that slice on every real-length video.
    long_script = "Seg. " * 2000  # ~10,000 characters
    result = build_description("A hook description.", long_script, day=1)

    assert len(result) <= 5000
    assert result.rstrip().endswith(DISCLAIMER)


def test_build_shorts_title_keeps_hashtag_intact_after_uploader_truncation():
    # engine/youtube.py's uploader applies title[:100] before sending to the API.
    # The Shorts title must still end with "#Shorts" after that slice, even for
    # an overly long blueprint title.
    long_title = "Why Your Brain Feels No Pain When You Tap a Card " * 3  # >150 chars
    short_title = build_shorts_title(long_title)

    assert (short_title[:100]).endswith("#Shorts")


def test_build_thumbnail_prompts_returns_two_style_locked_variants():
    prompts = build_thumbnail_prompts("Why Your Brain Feels No Pain")
    assert len(prompts) == 2
    for p in prompts:
        assert "WHY YOUR BRAIN FEELS NO PAIN" in p
        assert "16:9 aspect ratio" in p
    assert prompts[0] != prompts[1]


def test_slugify_produces_filesystem_safe_string():
    assert slugify("Why Your Brain Feels No Pain!") == "why_your_brain_feels_no_pain"


def test_run_pipeline_end_to_end_with_mocked_apis(tmp_path, fixture_image, fixture_audio, monkeypatch):
    monkeypatch.setattr("splurj_engine.WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr("splurj_engine.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("splurj_engine.ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr("splurj_engine.CHANNEL_DATA", tmp_path / "channel_data")
    (tmp_path / "assets" / "ambient").mkdir(parents=True)
    (tmp_path / "channel_data").mkdir(parents=True)

    fake_audio_gen = MagicMock()
    fake_audio_gen.generate_segment.side_effect = (
        lambda text, output_path, directive="", max_retries=4: shutil.copy2(fixture_audio, output_path) or output_path
    )
    fake_audio_gen.probe_duration.side_effect = lambda audio_path: 1.0

    fake_image_gen = MagicMock()
    fake_image_gen.generate.side_effect = (
        lambda prompt, output_path, reference_image_path=None, max_retries=4: shutil.copy2(fixture_image, output_path) or output_path
    )

    fake_uploader = MagicMock()
    fake_uploader.upload.return_value = {"id": "vid-fake"}

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch("splurj_engine.YouTubeUploader", return_value=fake_uploader):

        env = {
            "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
            "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
        }
        result = run_pipeline(_valid_blueprint(), env, skip_upload=False)

    from engine.video import probe_video_resolution

    assert result["long_form"].exists()
    assert probe_video_resolution(result["long_form"]) == (1920, 1080)
    assert len(result["shorts"]) == 1  # segments 0-1 form one contiguous run

    # Long-form upload + one Short upload = 2 calls; each description carries the disclaimer.
    assert fake_uploader.upload.call_count == 2
    for _, call_kwargs in fake_uploader.upload.call_args_list:
        assert DISCLAIMER in call_kwargs["description"]

    # One thumbnail generated per variant (2), first one set as the video's default thumbnail.
    fake_uploader.set_default_thumbnail.assert_called_once()


def test_run_pipeline_skip_upload_does_not_call_youtube(tmp_path, fixture_image, fixture_audio, monkeypatch):
    monkeypatch.setattr("splurj_engine.WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr("splurj_engine.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("splurj_engine.ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr("splurj_engine.CHANNEL_DATA", tmp_path / "channel_data")
    (tmp_path / "assets" / "ambient").mkdir(parents=True)
    (tmp_path / "channel_data").mkdir(parents=True)

    fake_audio_gen = MagicMock()
    fake_audio_gen.generate_segment.side_effect = (
        lambda text, output_path, directive="", max_retries=4: shutil.copy2(fixture_audio, output_path) or output_path
    )
    fake_audio_gen.probe_duration.side_effect = lambda audio_path: 1.0

    fake_image_gen = MagicMock()
    fake_image_gen.generate.side_effect = (
        lambda prompt, output_path, reference_image_path=None, max_retries=4: shutil.copy2(fixture_image, output_path) or output_path
    )

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch("splurj_engine.YouTubeUploader") as mock_uploader_cls:

        env = {
            "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
            "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
        }
        result = run_pipeline(_valid_blueprint(), env, skip_upload=True)

    assert result["long_form"].exists()
    mock_uploader_cls.assert_not_called()
