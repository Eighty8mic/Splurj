"""
Gemini-powered blueprint drafting for Splurj.

Two Gemini calls — a script draft, then a batch of scene image prompts —
mirror master_prompt_splurj.txt's own Stage 2 / Stage 3 split. Segmenting
the script and assembling the final blueprint dict is deterministic Python,
not left to the model, so the output always satisfies splurj_engine's
schema exactly.
"""
import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MODEL = "gemini-3.5-flash"

STYLE_ANCHOR = (
    "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, "
    "slightly imperfect sketchy marker lines, "
)
STYLE_LOCK = (
    ", no gradients, no shadows, no textures, no photorealism, no 3D, "
    "16:9 aspect ratio, educational YouTube explainer doodle style."
)

SEGMENT_WORD_BUDGET = 45
SCENE_SIZE = 3
WORDS_PER_SECOND = 2.3

_SCRIPT_RULES = (
    "You are the script writer for Splurj (@Splurj-it), a hand-drawn doodle YouTube channel "
    "about money psychology and behavioral finance. Tagline: 'Why you spend the way you do.'\n\n"
    "Voice: calm, curious, a little conspiratorial. Second person only ('you', 'your brain') — "
    "never 'we' or 'I'.\n"
    "Rhythm: short sentence, short sentence, one longer sentence that builds depth, short "
    "sentence, a question every 4-6 sentences.\n"
    "Length: 1,800-2,500 words of pure narration — no headers, no stage directions.\n"
    "Hook formula: open with a relatable money moment the viewer did this week, then reframe it.\n"
    "Close by echoing the opening line, reframed.\n\n"
    "SAFETY (YMYL — never violate):\n"
    "- Never give financial advice or recommend specific stocks, crypto, funds, banks, or apps.\n"
    "- Never promise outcomes. Explain behavior; never prescribe action.\n"
    "- You may name ONLY the researcher(s) given to you below. Do not invent, name, or attribute "
    "a claim to any other researcher, study, or statistic.\n"
)

_IMAGE_RULES = (
    "You are the storyboard artist for Splurj. Visual style: hand-drawn 2D doodle cartoon "
    "animation, flat colors, bold black outlines, sketchy marker lines. Simple stick figures "
    "with large circular heads, dot eyes, thick brow lines. Money motifs: dollar bills, piggy "
    "banks, coins, wallets, price tags, credit cards, brains with dollar-sign eyes.\n"
    "Background color signals tone: orange = urgency/sale/temptation, blue = science/experiment, "
    "tan = history/origins, white/yellow = happy/discovery, red accents = danger/debt/loss, "
    "green ground + blue sky = everyday outdoor life.\n"
    "Hold scenes: each prompt covers a short run of narration, so keep it general enough to "
    "represent that whole beat rather than one single instant.\n"
)


def _check_no_replacement_chars(data: Dict[str, Any]) -> None:
    """U+FFFD in a Gemini response is a rare model-output glitch (observed
    once, isolated to a single field, in an otherwise clean 2000-word draft)
    rather than a real character -- silently shipping it into a real YouTube
    description or narration looks broken, so treat it as a malformed draft."""
    for key, value in data.items():
        if isinstance(value, str) and "�" in value:
            raise ValueError(f"Response field '{key}' contains a U+FFFD replacement character")
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and "�" in item:
                    raise ValueError(f"Response field '{key}' contains a U+FFFD replacement character")


class GeminiScriptDrafter:
    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = _MODEL

    def draft(self, topic_hint: str, citation: Dict[str, Any], angle_formula: str, day: int) -> Dict[str, Any]:
        from google.genai import types

        citation_block = (
            f"Researcher(s): {citation['researchers']}\n"
            f"Study: \"{citation['study_ref']}\" ({citation['year']}, {citation['venue']})\n"
            f"Summary: {citation['summary']}"
        )
        prompt = (
            f"{_SCRIPT_RULES}\n"
            f"Topic angle to develop: {angle_formula}\n"
            f"Topic hint: {topic_hint}\n\n"
            f"The ONLY citation you may name in this script:\n{citation_block}\n\n"
            "Return JSON with exactly these keys: "
            '"title" (under 70 chars), "description" (2-3 sentence hook, under 400 chars), '
            '"tags" (8-15 lowercase strings, always include "splurj"), '
            '"directive" (one sentence describing the voice/tone for this episode), '
            '"script" (the full narration, plain text, 1800-2500 words), '
            '"researchers_named" (every researcher surname you actually named in the script).'
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        try:
            data = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Script draft did not return valid JSON: {exc}") from exc

        _check_no_replacement_chars(data)
        return data


class GeminiImagePromptDrafter:
    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = _MODEL

    def draft_scene_prompts(self, scene_texts: List[str]) -> List[str]:
        from google.genai import types

        numbered = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(scene_texts))
        prompt = (
            f"{_IMAGE_RULES}\n"
            f"Write one image prompt for each of the {len(scene_texts)} narration beats below. "
            "Each prompt must describe concrete visuals (characters, expressions, objects, "
            "background color, any on-screen ALL CAPS text) — translate abstract narration into "
            "a specific scene.\n\n"
            f"BEATS:\n{numbered}\n\n"
            'Return JSON: {"scene_prompts": [one string per beat, in order]}.'
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        try:
            data = json.loads(response.text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Image prompt draft did not return valid JSON: {exc}") from exc

        _check_no_replacement_chars(data)
        return [_enforce_style(p) for p in data["scene_prompts"]]


def _enforce_style(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt.startswith(STYLE_ANCHOR):
        prompt = STYLE_ANCHOR + prompt
    if not prompt.endswith("doodle style."):
        prompt = prompt.rstrip().rstrip(".") + STYLE_LOCK
    return prompt


def split_into_segments(script: str) -> List[str]:
    """Deterministic sentence-aware chunking — never splits a sentence, and
    groups consecutive sentences up to SEGMENT_WORD_BUDGET words per segment."""
    sentences = re.split(r"(?<=[.!?])\s+", script.strip())
    segments: List[str] = []
    current: List[str] = []
    current_words = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > SEGMENT_WORD_BUDGET:
            segments.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += words
    if current:
        segments.append(" ".join(current))
    return segments


def group_into_scenes(segments: List[str], scene_size: int = SCENE_SIZE) -> List[List[int]]:
    """Index-groups of consecutive segments that share one held image scene."""
    return [
        list(range(i, min(i + scene_size, len(segments))))
        for i in range(0, len(segments), scene_size)
    ]


def check_citation_safety(researchers_named: List[str], allowed_citations: List[Dict[str, Any]]) -> List[str]:
    """Names in researchers_named not covered by any allowed citation's
    researcher list. Empty = passes the citation QA gate."""
    allowed_text = " ".join(c["researchers"] for c in allowed_citations).lower()
    return [name for name in researchers_named if name.strip().lower() not in allowed_text]


def build_blueprint(
    day: int,
    title: str,
    description: str,
    tags: List[str],
    directive: str,
    segments: List[str],
    scene_groups: List[List[int]],
    scene_prompts: List[str],
) -> Dict[str, Any]:
    prompt_by_segment: Dict[int, str] = {}
    for group, prompt in zip(scene_groups, scene_prompts):
        for idx in group:
            prompt_by_segment[idx] = prompt

    short_candidate_indices = set(scene_groups[0]) | set(scene_groups[-1])

    timeline = []
    t = 0.0
    for i, text in enumerate(segments):
        dur = max(len(text.split()) / WORDS_PER_SECOND, 1.5)
        timeline.append({
            "start": round(t, 1),
            "end": round(t + dur, 1),
            "text": text,
            "prompt": prompt_by_segment[i],
            "is_short_candidate": i in short_candidate_indices,
        })
        t += dur

    return {
        "day": day,
        "format": "long",
        "metadata": {
            "title": title,
            "description": description[:900],
            "tags": tags,
        },
        "voiceover": {
            "directive": directive,
            "full_text": " ".join(seg["text"] for seg in timeline),
        },
        "timeline": timeline,
    }
