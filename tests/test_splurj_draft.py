import json
from unittest.mock import MagicMock, patch

import pytest

from splurj_draft import DraftError, draft_blueprint, _read_next_day, _write_next_day


def _citation(id=1, researchers="Richard Thaler", contested=False):
    return {
        "id": id, "researchers": researchers, "study_ref": "Mental Accounting Matters",
        "year": 1999, "venue": "JBDM", "summary": "summary text",
        "source_url": "https://example.com", "status": "approved", "contested": contested,
    }


def _script_payload(researchers_named=("Thaler",), n_sentences=6):
    script = " ".join(f"Sentence number {i} about money." for i in range(n_sentences))
    return {
        "title": "Why Your Brain Sorts Money",
        "description": "A hook about mental accounting.",
        "tags": ["splurj", "money psychology"],
        "directive": "Calm, curious.",
        "script": script,
        "researchers_named": list(researchers_named),
    }


def _setup_citations(tmp_path, monkeypatch, citations=None):
    citations_path = tmp_path / "citations.json"
    citations_path.write_text(json.dumps(citations or [_citation()]), encoding="utf-8")
    used_topics_path = tmp_path / "used_topics.json"
    monkeypatch.setattr("splurj_draft.CITATIONS_PATH", citations_path)
    monkeypatch.setattr("splurj_draft.USED_TOPICS_PATH", used_topics_path)
    return citations_path, used_topics_path


def test_draft_blueprint_returns_valid_blueprint_on_first_try(tmp_path, monkeypatch):
    _setup_citations(tmp_path, monkeypatch)

    fake_script_drafter = MagicMock()
    fake_script_drafter.draft.return_value = _script_payload()
    fake_image_drafter = MagicMock()
    fake_image_drafter.draft_scene_prompts.return_value = ["prompt one", "prompt two"]

    with patch("splurj_draft.GeminiScriptDrafter", return_value=fake_script_drafter), \
         patch("splurj_draft.GeminiImagePromptDrafter", return_value=fake_image_drafter):
        blueprint = draft_blueprint(day=7, gemini_api_key="key")

    assert blueprint["day"] == 7
    assert blueprint["metadata"]["title"] == "Why Your Brain Sorts Money"
    assert blueprint["voiceover"]["full_text"] == " ".join(s["text"] for s in blueprint["timeline"])
    fake_script_drafter.draft.assert_called_once()


def test_draft_blueprint_retries_on_citation_violation_then_succeeds(tmp_path, monkeypatch):
    _setup_citations(tmp_path, monkeypatch)

    fake_script_drafter = MagicMock()
    fake_script_drafter.draft.side_effect = [
        _script_payload(researchers_named=["Some Fabricated Person"]),
        _script_payload(researchers_named=["Thaler"]),
    ]
    fake_image_drafter = MagicMock()
    fake_image_drafter.draft_scene_prompts.return_value = ["prompt one", "prompt two"]

    with patch("splurj_draft.GeminiScriptDrafter", return_value=fake_script_drafter), \
         patch("splurj_draft.GeminiImagePromptDrafter", return_value=fake_image_drafter):
        blueprint = draft_blueprint(day=8, gemini_api_key="key")

    assert blueprint["day"] == 8
    assert fake_script_drafter.draft.call_count == 2


def test_draft_blueprint_retries_on_malformed_response_then_succeeds(tmp_path, monkeypatch):
    """A malformed/corrupted Gemini response (bad JSON, stray replacement
    character) must trigger a retry like a citation violation does, not
    crash draft_blueprint entirely."""
    _setup_citations(tmp_path, monkeypatch)

    fake_script_drafter = MagicMock()
    fake_script_drafter.draft.side_effect = [
        ValueError("Response field 'description' contains a U+FFFD replacement character"),
        _script_payload(),
    ]
    fake_image_drafter = MagicMock()
    fake_image_drafter.draft_scene_prompts.return_value = ["prompt one", "prompt two"]

    with patch("splurj_draft.GeminiScriptDrafter", return_value=fake_script_drafter), \
         patch("splurj_draft.GeminiImagePromptDrafter", return_value=fake_image_drafter):
        blueprint = draft_blueprint(day=11, gemini_api_key="key")

    assert blueprint["day"] == 11
    assert fake_script_drafter.draft.call_count == 2


def test_draft_blueprint_raises_after_max_attempts_all_violating(tmp_path, monkeypatch):
    _setup_citations(tmp_path, monkeypatch)

    fake_script_drafter = MagicMock()
    fake_script_drafter.draft.return_value = _script_payload(researchers_named=["Fabricated Person"])
    fake_image_drafter = MagicMock()

    with patch("splurj_draft.GeminiScriptDrafter", return_value=fake_script_drafter), \
         patch("splurj_draft.GeminiImagePromptDrafter", return_value=fake_image_drafter):
        with pytest.raises(DraftError, match="Citation QA failed"):
            draft_blueprint(day=9, gemini_api_key="key")

    fake_image_drafter.draft_scene_prompts.assert_not_called()  # never reached image drafting


def test_draft_blueprint_records_used_topic_on_success(tmp_path, monkeypatch):
    _, used_topics_path = _setup_citations(tmp_path, monkeypatch)

    fake_script_drafter = MagicMock()
    fake_script_drafter.draft.return_value = _script_payload()
    fake_image_drafter = MagicMock()
    fake_image_drafter.draft_scene_prompts.return_value = ["prompt one", "prompt two"]

    with patch("splurj_draft.GeminiScriptDrafter", return_value=fake_script_drafter), \
         patch("splurj_draft.GeminiImagePromptDrafter", return_value=fake_image_drafter):
        draft_blueprint(day=10, gemini_api_key="key")

    recorded = json.loads(used_topics_path.read_text(encoding="utf-8"))
    assert len(recorded) == 1
    assert recorded[0]["video_day"] == 10
    assert recorded[0]["citation_ids"] == [1]


def test_read_next_day_defaults_to_3_when_no_counter_file(tmp_path, monkeypatch):
    monkeypatch.setattr("splurj_draft.NEXT_DAY_PATH", tmp_path / "next_day.txt")
    assert _read_next_day() == 3


def test_write_then_read_next_day_round_trips(tmp_path, monkeypatch):
    counter_path = tmp_path / "next_day.txt"
    monkeypatch.setattr("splurj_draft.NEXT_DAY_PATH", counter_path)

    _write_next_day(15)

    assert _read_next_day() == 15
