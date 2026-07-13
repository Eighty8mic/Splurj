import json
from unittest.mock import MagicMock, patch

import pytest

from engine.drafting import (
    STYLE_ANCHOR,
    STYLE_LOCK,
    GeminiImagePromptDrafter,
    GeminiScriptDrafter,
    build_blueprint,
    check_citation_safety,
    group_into_scenes,
    split_into_segments,
)


def _fake_json_response(data) -> MagicMock:
    return MagicMock(text=json.dumps(data))


# ── split_into_segments ──────────────────────────────────────────────────

def test_split_into_segments_never_splits_a_sentence():
    script = "First sentence here. Second sentence here. Third sentence here."
    segments = split_into_segments(script)
    joined = " ".join(segments)
    assert joined == script


def test_split_into_segments_respects_word_budget():
    long_sentence_script = " ".join(f"Word{i} sentence number {i}." for i in range(30))
    segments = split_into_segments(long_sentence_script)
    for seg in segments:
        assert len(seg.split()) <= 50  # budget plus one sentence's worth of slack


# ── group_into_scenes ────────────────────────────────────────────────────

def test_group_into_scenes_groups_by_scene_size():
    segments = [f"seg{i}" for i in range(7)]
    groups = group_into_scenes(segments, scene_size=3)
    assert groups == [[0, 1, 2], [3, 4, 5], [6]]


# ── check_citation_safety ────────────────────────────────────────────────

def test_check_citation_safety_passes_when_names_match_citation():
    citations = [{"id": 1, "researchers": "Daniel Kahneman & Amos Tversky"}]
    violations = check_citation_safety(["Kahneman", "Tversky"], citations)
    assert violations == []


def test_check_citation_safety_flags_unlisted_researcher():
    citations = [{"id": 1, "researchers": "Daniel Kahneman & Amos Tversky"}]
    violations = check_citation_safety(["Kahneman", "Some Fabricated Researcher"], citations)
    assert violations == ["Some Fabricated Researcher"]


def test_check_citation_safety_is_case_insensitive():
    citations = [{"id": 1, "researchers": "Richard Thaler"}]
    violations = check_citation_safety(["THALER"], citations)
    assert violations == []


# ── build_blueprint ───────────────────────────────────────────────────────

def test_build_blueprint_full_text_matches_segment_concatenation():
    segments = ["First segment.", "Second segment.", "Third segment."]
    scene_groups = [[0, 1], [2]]
    scene_prompts = ["prompt A", "prompt B"]

    blueprint = build_blueprint(
        day=5, title="Title", description="Desc", tags=["splurj"], directive="calm",
        segments=segments, scene_groups=scene_groups, scene_prompts=scene_prompts,
    )

    assert blueprint["voiceover"]["full_text"] == " ".join(s["text"] for s in blueprint["timeline"])
    assert blueprint["voiceover"]["full_text"] == "First segment. Second segment. Third segment."


def test_build_blueprint_marks_first_and_last_scene_as_short_candidates():
    segments = ["A.", "B.", "C.", "D.", "E."]
    scene_groups = [[0, 1], [2], [3, 4]]
    scene_prompts = ["p1", "p2", "p3"]

    blueprint = build_blueprint(
        day=1, title="T", description="D", tags=["x"], directive="calm",
        segments=segments, scene_groups=scene_groups, scene_prompts=scene_prompts,
    )

    flags = [seg["is_short_candidate"] for seg in blueprint["timeline"]]
    assert flags == [True, True, False, True, True]


def test_build_blueprint_applies_scene_prompt_to_every_segment_in_the_scene():
    segments = ["A.", "B.", "C."]
    scene_groups = [[0, 1], [2]]
    scene_prompts = ["shared prompt", "solo prompt"]

    blueprint = build_blueprint(
        day=1, title="T", description="D", tags=["x"], directive="calm",
        segments=segments, scene_groups=scene_groups, scene_prompts=scene_prompts,
    )

    assert blueprint["timeline"][0]["prompt"] == "shared prompt"
    assert blueprint["timeline"][1]["prompt"] == "shared prompt"
    assert blueprint["timeline"][2]["prompt"] == "solo prompt"


def test_build_blueprint_truncates_oversized_description():
    blueprint = build_blueprint(
        day=1, title="T", description="x" * 2000, tags=["x"], directive="calm",
        segments=["A."], scene_groups=[[0]], scene_prompts=["p"],
    )
    assert len(blueprint["metadata"]["description"]) <= 900


# ── GeminiScriptDrafter ──────────────────────────────────────────────────

def test_script_drafter_parses_json_response():
    payload = {
        "title": "Why Your Brain X",
        "description": "A hook.",
        "tags": ["splurj", "money psychology"],
        "directive": "Calm, curious.",
        "script": "You tapped a card. Nothing happened.",
        "researchers_named": ["Thaler"],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_json_response(payload)
        drafter = GeminiScriptDrafter(api_key="key")
        result = drafter.draft(
            topic_hint="mental accounting",
            citation={"researchers": "Richard Thaler", "study_ref": "Mental Accounting Matters",
                      "year": 1999, "venue": "JBDM", "summary": "..."},
            angle_formula="The ___ Effect",
            day=3,
        )

    assert result == payload


def test_script_drafter_raises_on_replacement_character_in_output():
    """Regression test: a real draft run returned a description containing a
    U+FFFD replacement character (rare model-output glitch, isolated to one
    field). Silently shipping that to a real YouTube description looks
    broken, so treat it as a malformed draft and let the caller retry."""
    payload = {
        "title": "T", "description": "not a lack of willpower�your brain is",
        "tags": ["splurj"], "directive": "calm", "script": "Sentence one.",
        "researchers_named": [],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_json_response(payload)
        drafter = GeminiScriptDrafter(api_key="key")
        with pytest.raises(ValueError, match="replacement character"):
            drafter.draft(
                topic_hint="x", citation={"researchers": "x", "study_ref": "x", "year": 2000,
                                          "venue": "x", "summary": "x"},
                angle_formula="x", day=1,
            )


def test_script_drafter_raises_on_malformed_json():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = MagicMock(text="not json")
        drafter = GeminiScriptDrafter(api_key="key")
        with pytest.raises(ValueError):
            drafter.draft(
                topic_hint="x", citation={"researchers": "x", "study_ref": "x", "year": 2000,
                                          "venue": "x", "summary": "x"},
                angle_formula="x", day=1,
            )


# ── GeminiImagePromptDrafter ─────────────────────────────────────────────

def test_image_prompt_drafter_enforces_style_anchor_and_lock():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_json_response(
            {"scene_prompts": ["a stick figure holding a wallet"]}
        )
        drafter = GeminiImagePromptDrafter(api_key="key")
        result = drafter.draft_scene_prompts(["You tapped a card."])

    assert result[0].startswith(STYLE_ANCHOR)
    assert result[0].endswith("doodle style.")


def test_image_prompt_drafter_leaves_already_styled_prompts_unchanged():
    already_styled = STYLE_ANCHOR + "a stick figure" + STYLE_LOCK
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_json_response(
            {"scene_prompts": [already_styled]}
        )
        drafter = GeminiImagePromptDrafter(api_key="key")
        result = drafter.draft_scene_prompts(["text"])

    assert result[0] == already_styled
