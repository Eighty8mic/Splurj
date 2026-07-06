#!/usr/bin/env python3
"""
One-time generator for channel_data/character_reference.png — the neutral-pose
sheet of Splurj's recurring stick-figure character and props, passed as a
reference image to every scene generation call for visual consistency.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from engine.images import ImageGenerator

STYLE_ANCHOR = (
    "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, "
    "slightly imperfect sketchy marker lines,"
)
STYLE_LOCK = (
    "no gradients, no shadows, no textures, no photorealism, no 3D, "
    "16:9 aspect ratio, educational YouTube explainer doodle style."
)


def build_prompt() -> str:
    return (
        f"{STYLE_ANCHOR} a character reference sheet on a plain white background: "
        "the main stick-figure character (simple circular head, dot eyes, thick "
        "expressive brow lines) standing in a neutral pose, front-facing, arms at "
        "sides, next to its recurring props laid out beside it — a dollar bill with "
        "a simple face, a piggy bank, a stack of coins, a wallet with a face, a "
        "price tag. No background scenery, no on-screen text. {STYLE_LOCK}"
    ).replace("{STYLE_LOCK}", STYLE_LOCK)


def main(argv: list[str] | None = None) -> Path:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate Splurj's character reference sheet.")
    parser.add_argument(
        "--output", "-o",
        default="channel_data/character_reference.png",
        help="Output path for the reference PNG",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Add it to your .env file.")

    gen = ImageGenerator(api_key=api_key, use_cache=False)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = gen.generate(build_prompt(), output_path)
    print(f"Character reference saved -> {result}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
