#!/usr/bin/env python3
"""
Splurj Draft — automatic topic + script writer.

Picks a topic from the approved citation bank, drafts a full script via
Gemini, generates per-scene image prompts, and runs the citation safety
gate (retrying on a violation, matching the design's one hard automated
safety control). The finished blueprint is written to queue_local/next.json
for the nightly render+upload task.

    python splurj_draft.py                # auto-incrementing day number
    python splurj_draft.py --day 12        # explicit day number
"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from engine.drafting import (
    GeminiImagePromptDrafter,
    GeminiScriptDrafter,
    build_blueprint,
    check_citation_safety,
    group_into_scenes,
    split_into_segments,
)
from engine.topic_picker import load_citations, load_used_topics, pick_topic, record_used_topic
from splurj_engine import _validate_blueprint

BASE_DIR = Path(__file__).parent
CITATIONS_PATH = BASE_DIR / "channel_data" / "citations.json"
USED_TOPICS_PATH = BASE_DIR / "channel_data" / "used_topics.json"
NEXT_DAY_PATH = BASE_DIR / "channel_data" / "next_day.txt"
QUEUE_NEXT_PATH = BASE_DIR / "queue_local" / "next.json"

MAX_DRAFT_ATTEMPTS = 3
# Days 1 and 2 were hand-authored before this drafter existed.
FIRST_AUTO_DAY = 3

logger = logging.getLogger("splurj_draft")


class DraftError(Exception):
    """Raised when a blueprint cannot be drafted safely after all retries."""


def draft_blueprint(day: int, gemini_api_key: str) -> Dict[str, Any]:
    citations = load_citations(CITATIONS_PATH)
    used_topics = load_used_topics(USED_TOPICS_PATH)
    pick = pick_topic(citations, used_topics)

    script_drafter = GeminiScriptDrafter(api_key=gemini_api_key)
    image_drafter = GeminiImagePromptDrafter(api_key=gemini_api_key)

    last_violations = []
    for attempt in range(1, MAX_DRAFT_ATTEMPTS + 1):
        logger.info(
            "Draft attempt %d/%d -- citation: %s, angle: %s",
            attempt, MAX_DRAFT_ATTEMPTS, pick["citation"]["researchers"], pick["angle_key"],
        )

        try:
            draft = script_drafter.draft(
                topic_hint=pick["citation"]["study_ref"],
                citation=pick["citation"],
                angle_formula=pick["angle_formula"],
                day=day,
            )
        except ValueError as exc:
            logger.warning("Script draft malformed (attempt %d): %s", attempt, exc)
            last_violations = [f"malformed response: {exc}"]
            continue

        violations = check_citation_safety(draft.get("researchers_named", []), [pick["citation"]])
        if violations:
            logger.warning("Citation QA failed (attempt %d): unlisted researcher(s) %s", attempt, violations)
            last_violations = violations
            continue

        segments = split_into_segments(draft["script"])
        logger.info("Script: %d words, %d segments", len(draft["script"].split()), len(segments))

        scene_groups = group_into_scenes(segments)
        scene_texts = [" ".join(segments[i] for i in group) for group in scene_groups]
        try:
            scene_prompts = image_drafter.draft_scene_prompts(scene_texts)
        except ValueError as exc:
            logger.warning("Image prompt draft malformed (attempt %d): %s", attempt, exc)
            last_violations = [f"malformed response: {exc}"]
            continue

        blueprint = build_blueprint(
            day=day, title=draft["title"], description=draft["description"], tags=draft["tags"],
            directive=draft["directive"], segments=segments, scene_groups=scene_groups,
            scene_prompts=scene_prompts,
        )
        _validate_blueprint(blueprint, source="drafted blueprint")

        record_used_topic(USED_TOPICS_PATH, {
            "citation_ids": pick["citation_ids"], "angle_key": pick["angle_key"], "video_day": day,
        })
        return blueprint

    raise DraftError(
        f"Citation QA failed after {MAX_DRAFT_ATTEMPTS} attempts -- last unlisted researcher(s): "
        f"{last_violations}. Draft abandoned; nothing was queued."
    )


def _read_next_day() -> int:
    if NEXT_DAY_PATH.exists():
        return int(NEXT_DAY_PATH.read_text(encoding="utf-8").strip())
    return FIRST_AUTO_DAY


def _write_next_day(day: int) -> None:
    NEXT_DAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    NEXT_DAY_PATH.write_text(str(day), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="splurj_draft", description="Draft the next Splurj blueprint from the approved citation bank."
    )
    parser.add_argument("--day", type=int, default=None, help="Override the day number (default: auto-incrementing)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-16s  %(message)s", datefmt="%H:%M:%S",
    )

    load_dotenv()
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise SystemExit("GEMINI_API_KEY is not set")

    if QUEUE_NEXT_PATH.exists():
        raise SystemExit(
            f"{QUEUE_NEXT_PATH} already has a queued blueprint -- render it first "
            "(or remove it) before drafting a new one."
        )

    day = args.day if args.day is not None else _read_next_day()
    blueprint = draft_blueprint(day, gemini_api_key)

    QUEUE_NEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_NEXT_PATH.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    logger.info("Drafted day %d '%s' -> %s", day, blueprint["metadata"]["title"], QUEUE_NEXT_PATH)

    if args.day is None:
        _write_next_day(day + 1)


if __name__ == "__main__":
    main()
