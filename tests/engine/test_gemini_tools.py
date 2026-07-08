import copy
from unittest.mock import MagicMock, patch

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
            {"start": 0, "end": 15, "text": "First segment.", "prompt": "p1", "is_short_candidate": False},
            {"start": 15, "end": 30, "text": "Second segment.", "prompt": "p2", "is_short_candidate": False},
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
    assert result["timeline"][0]["prompt"] == "p1"  # untouched by the script polisher


def test_polish_blueprint_does_not_mutate_input():
    blueprint = {
        "voiceover": {"directive": "calm", "full_text": "old text"},
        "timeline": [
            {"start": 0, "end": 15, "text": "First segment.", "prompt": "p1", "is_short_candidate": False},
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
            "Hand-drawn 2D doodle... enhanced prompt ... 16:9 aspect ratio."
        )
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "You feel nothing when you tap a card.")

    assert "enhanced prompt" in result


def test_enhance_prompt_falls_back_to_original_on_api_error():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("boom")
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "segment text")

    assert result == "a doodle wallet"


def test_enhance_blueprint_rebuilds_timeline_prompts():
    blueprint = {
        "timeline": [
            {"start": 0, "end": 15, "text": "First.", "prompt": "p1", "is_short_candidate": False},
        ],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response("p1-enhanced")
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_blueprint(blueprint)

    assert result["timeline"][0]["prompt"] == "p1-enhanced"
    assert result["timeline"][0]["text"] == "First."  # untouched by the prompt enhancer
