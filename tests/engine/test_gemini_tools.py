import copy
from unittest.mock import MagicMock, patch

from engine.drafting import STYLE_ANCHOR, STYLE_LOCK
from engine.gemini_tools import GeminiPromptEnhancer, GeminiScriptPolisher


def _fake_text_response(text: str) -> MagicMock:
    return MagicMock(text=text)


def test_polish_segment_returns_polished_text_on_success():
    with patch("google.genai.Client") as mock_client_cls:
        original = "You know that feeling when you tap a card and nothing happens? Your brain doesn't register the payment. That's the dopamine gap—the disconnect between the action and the emotional impact."
        polished = "When you tap a card, does your brain register the action? Most people don't feel the payment anymore. That gap—between the tap and the emotion—that's where psychology and money collide."
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response(polished)
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_segment(original, directive="calm")

    assert result == polished


def test_polish_segment_falls_back_to_original_on_api_error():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("boom")
        polisher = GeminiScriptPolisher(api_key="key")
        original = "You tapped your card. You felt nothing."
        result = polisher.polish_segment(original, directive="calm")

    assert result == original


def test_polish_segment_falls_back_when_word_count_drifts_too_much():
    original = "You tapped your card. You felt nothing. " * 5  # ~35 words
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response("Short.")
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_segment(original, directive="calm")

    assert result == original


def test_polish_blueprint_rebuilds_full_text_from_polished_segments():
    blueprint = {
        "voiceover": {"directive": "calm", "full_text": "old text"},
        "timeline": [
            {"start": 0, "end": 15, "text": "First segment.", "poses": ["p1"], "is_short_candidate": False},
            {"start": 15, "end": 30, "text": "Second segment.", "poses": ["p2"], "is_short_candidate": False},
        ],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = [
            _fake_text_response("First segment polished."),
            _fake_text_response("Second segment polished."),
        ]
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_blueprint(blueprint)

    assert result["timeline"][0]["text"] == "First segment polished."
    assert result["timeline"][1]["text"] == "Second segment polished."
    assert result["voiceover"]["full_text"] == "First segment polished. Second segment polished."
    assert result["timeline"][0]["poses"] == ["p1"]  # untouched by the script polisher


def test_polish_blueprint_does_not_mutate_input():
    blueprint = {
        "voiceover": {"directive": "calm", "full_text": "old text"},
        "timeline": [
            {"start": 0, "end": 15, "text": "First segment.", "poses": ["p1"], "is_short_candidate": False},
        ],
    }
    blueprint_copy = copy.deepcopy(blueprint)

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response("First segment polished.")
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_blueprint(blueprint)

    assert blueprint == blueprint_copy
    assert result["timeline"][0] is not blueprint["timeline"][0]


def test_enhance_prompt_returns_enhanced_text_on_success():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response(
            STYLE_ANCHOR + "enhanced prompt" + STYLE_LOCK
        )
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "You feel nothing when you tap a card.")

    assert "enhanced prompt" in result


def test_enhance_prompt_reapplies_style_anchor_when_gemini_drops_it():
    """Regression test: enhance_prompt's success path used to return Gemini's
    raw text without re-running _enforce_style(), unlike draft_scene_poses --
    if Gemini's 'enhanced' text drops the required style anchor/lock, nothing
    re-injected it, silently producing an off-brand image prompt."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response(
            "a doodle wallet with no style markers at all"
        )
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "segment text")

    assert result.startswith(STYLE_ANCHOR)
    assert result.endswith("doodle style.")


def test_enhance_prompt_falls_back_to_original_on_api_error():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("boom")
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "segment text")

    assert result == "a doodle wallet"


def test_enhance_blueprint_rebuilds_timeline_poses():
    blueprint = {
        "timeline": [
            {"start": 0, "end": 15, "text": "First.", "poses": ["p1", "p2"], "is_short_candidate": False},
        ],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = [
            _fake_text_response("p1-enhanced"),
            _fake_text_response("p2-enhanced"),
        ]
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_blueprint(blueprint)

    # enhance_prompt re-applies _enforce_style, so the raw Gemini text is
    # wrapped in the anchor/lock strings -- assert the enhanced text survives
    # inside that wrapper rather than an exact match on the raw text.
    assert "p1-enhanced" in result["timeline"][0]["poses"][0]
    assert "p2-enhanced" in result["timeline"][0]["poses"][1]
    assert result["timeline"][0]["text"] == "First."  # untouched by the prompt enhancer
