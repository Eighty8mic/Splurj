import json
from unittest.mock import MagicMock, patch

import pytest

from run_queued import run_queued


def _blueprint(day=9):
    return {
        "day": day,
        "format": "long",
        "metadata": {"title": "Test Video", "description": "A hook.", "tags": ["splurj"]},
        "voiceover": {"directive": "calm", "full_text": "Seg one."},
        "timeline": [
            {"start": 0, "end": 15, "text": "Seg one.", "prompt": "doodle prompt", "is_short_candidate": False},
        ],
    }


def test_run_queued_noops_when_nothing_queued(tmp_path):
    next_path = tmp_path / "next.json"  # does not exist
    processed_dir = tmp_path / "processed"

    with patch("splurj_engine.run_pipeline") as mock_run_pipeline:
        run_queued(next_path=next_path, processed_dir=processed_dir)

    mock_run_pipeline.assert_not_called()
    assert not processed_dir.exists()


def test_run_queued_renders_with_upload_enabled_and_archives_on_success(tmp_path):
    next_path = tmp_path / "next.json"
    next_path.write_text(json.dumps(_blueprint()), encoding="utf-8")
    processed_dir = tmp_path / "processed"

    fake_env = {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }

    with patch("splurj_engine.load_env", return_value=fake_env), \
         patch("splurj_engine.run_pipeline") as mock_run_pipeline:
        run_queued(next_path=next_path, processed_dir=processed_dir)

    mock_run_pipeline.assert_called_once()
    _, call_kwargs = mock_run_pipeline.call_args
    assert call_kwargs["skip_upload"] is False

    assert not next_path.exists()  # moved out, so a future run without a new file no-ops
    archived = list(processed_dir.glob("*_next.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text(encoding="utf-8"))["day"] == 9


def test_run_queued_auto_drafts_when_nothing_queued_then_renders(tmp_path):
    next_path = tmp_path / "next.json"  # does not exist -- triggers auto-draft
    processed_dir = tmp_path / "processed"
    drafted = _blueprint(day=3)

    fake_env = {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }

    with patch("splurj_engine.load_env", return_value=fake_env), \
         patch("splurj_draft._read_next_day", return_value=3), \
         patch("splurj_draft._write_next_day") as mock_write_next_day, \
         patch("splurj_draft.draft_blueprint", return_value=drafted) as mock_draft, \
         patch("splurj_engine.run_pipeline") as mock_run_pipeline:
        run_queued(next_path=next_path, processed_dir=processed_dir)

    mock_draft.assert_called_once_with(3, "c")
    mock_write_next_day.assert_called_once_with(4)
    mock_run_pipeline.assert_called_once()
    archived = list(processed_dir.glob("*_next.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text(encoding="utf-8"))["day"] == 3


def test_run_queued_noops_silently_when_auto_draft_fails(tmp_path):
    next_path = tmp_path / "next.json"
    processed_dir = tmp_path / "processed"

    fake_env = {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }

    with patch("splurj_engine.load_env", return_value=fake_env), \
         patch("splurj_draft._read_next_day", return_value=3), \
         patch("splurj_draft.draft_blueprint", side_effect=RuntimeError("citation QA exhausted")), \
         patch("splurj_engine.run_pipeline") as mock_run_pipeline:
        run_queued(next_path=next_path, processed_dir=processed_dir)  # must not raise

    mock_run_pipeline.assert_not_called()
    assert not next_path.exists()


def test_run_queued_leaves_file_in_place_when_render_fails(tmp_path):
    next_path = tmp_path / "next.json"
    next_path.write_text(json.dumps(_blueprint()), encoding="utf-8")
    processed_dir = tmp_path / "processed"

    fake_env = {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }

    with patch("splurj_engine.load_env", return_value=fake_env), \
         patch("splurj_engine.run_pipeline", side_effect=RuntimeError("API error")):
        with pytest.raises(RuntimeError, match="API error"):
            run_queued(next_path=next_path, processed_dir=processed_dir)

    assert next_path.exists()  # left in place so the next scheduled run retries it
    assert not processed_dir.exists() or not list(processed_dir.glob("*"))
