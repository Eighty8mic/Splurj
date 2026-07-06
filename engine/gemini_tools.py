"""
Gemini-powered render-time pre-processing tools for the Splurj pipeline.

Two optional passes over an already-drafted blueprint, run before asset
generation:
  - GeminiScriptPolisher : refines narration text against Splurj voice rules
  - GeminiPromptEnhancer : verifies and strengthens doodle-style image prompts
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_MODEL = "gemini-3.5-flash"

_VOICE_RULES = (
    "Splurj narration voice rules:\n"
    "- Calm, curious, a little conspiratorial. 2nd person only ('you', 'your brain') — never 'we' or 'I'.\n"
    "- Rhythm: short sentence, short sentence, one longer sentence that builds depth, short sentence, "
    "a question every 4-6 sentences.\n"
    "- No jargon without an immediate plain-English decode.\n"
    "- Zero financial advice. No outcome promises. Explain behavior; never prescribe action.\n"
    "- Ends by reflecting the psychological truth onto a money decision the viewer will make this week.\n"
)

_VISUAL_RULES = (
    "Splurj visual style rules:\n"
    "- Hand-drawn 2D doodle cartoon animation. Flat colors, bold black outlines, sketchy marker lines.\n"
    "- Simple stick figures: large circular heads, dot eyes, thick expressive brow lines.\n"
    "- Background color signals tone: orange = urgency/sale, blue = science/experiment, tan = history, "
    "white/yellow = happy/discovery, green ground + blue sky = everyday outdoor life.\n"
    "- Required suffix: no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, "
    "educational YouTube explainer doodle style.\n"
)


class GeminiScriptPolisher:
    """Refines segment narration text against Splurj voice rules. Falls back
    to the original text on any failure or excessive word-count drift."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = _MODEL

    def polish_segment(self, text: str, directive: str) -> str:
        user_prompt = (
            f"{_VOICE_RULES}\n"
            f"Voice directive for this episode: {directive}\n\n"
            f"Polish the segment below. Keep approximately the same word count (~30-40 words).\n"
            f"Return ONLY the revised text — no quotes, no explanation.\n\n"
            f"SEGMENT:\n{text}"
        )

        try:
            response = self._client.models.generate_content(model=self._model, contents=user_prompt)
            result = response.text.strip()
        except Exception as exc:
            logger.warning("Script polish failed (segment) — using original: %s", exc)
            return text

        orig_wc = len(text.split())
        result_wc = len(result.split())
        if not (orig_wc * 0.5 <= result_wc <= orig_wc * 2.0):
            logger.warning("Polish word count drift (%d -> %d) — using original", orig_wc, result_wc)
            return text

        return result

    def polish_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        directive = blueprint["voiceover"].get("directive", "")
        new_timeline = []

        for i, seg in enumerate(blueprint["timeline"]):
            logger.info("  Polishing segment %d/%d…", i + 1, len(blueprint["timeline"]))
            polished = self.polish_segment(seg["text"], directive)
            new_timeline.append({**seg, "text": polished})

        full_text = " ".join(s["text"] for s in new_timeline)
        return {
            **blueprint,
            "timeline": new_timeline,
            "voiceover": {**blueprint["voiceover"], "full_text": full_text},
        }


class GeminiPromptEnhancer:
    """Verifies and strengthens doodle-style image prompts before image
    generation. Falls back to the original prompt on any failure."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = _MODEL

    def enhance_prompt(self, prompt: str, segment_text: str) -> str:
        user_prompt = (
            f"{_VISUAL_RULES}\n"
            f"The narration for this frame:\n{segment_text}\n\n"
            f"Enhance the image prompt below to:\n"
            f"  1. Better serve the narration.\n"
            f"  2. Strictly follow the Splurj visual style rules.\n"
            f"  3. Stay under 120 words.\n"
            f"Return ONLY the enhanced prompt — no explanation.\n\n"
            f"PROMPT:\n{prompt}"
        )

        try:
            response = self._client.models.generate_content(model=self._model, contents=user_prompt)
            return response.text.strip()
        except Exception as exc:
            logger.warning("Prompt enhancement failed — using original: %s", exc)
            return prompt

    def enhance_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        new_timeline = []
        for i, seg in enumerate(blueprint["timeline"]):
            logger.info("  Enhancing prompt %d/%d…", i + 1, len(blueprint["timeline"]))
            enhanced = self.enhance_prompt(seg["prompt"], seg["text"])
            new_timeline.append({**seg, "prompt": enhanced})

        return {**blueprint, "timeline": new_timeline}
