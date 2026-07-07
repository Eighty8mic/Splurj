#!/usr/bin/env python3
"""
Splurj Engine — blueprint-driven video render + upload orchestrator.

Usage:
    python splurj_engine.py --input content_example.json
    python splurj_engine.py --input content_example.json --no-upload
    python splurj_engine.py --input content_example.json --no-upload --keep-workspace --verbose
"""

import argparse
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from engine.audio import AudioGenerator
from engine.images import ImageGenerator
from engine.video import VideoAssembler
from engine.youtube import YouTubeUploader

BASE_DIR = Path(__file__).parent
WORKSPACE = BASE_DIR / "workspace"
OUTPUT_DIR = BASE_DIR / "output"
ASSETS_DIR = BASE_DIR / "assets"
CHANNEL_DATA = BASE_DIR / "channel_data"

DISCLAIMER = "This video is for education and entertainment only. It is not financial advice."

# Safety margin under YouTube's hard 5,000-char description limit (engine/youtube.py
# applies description[:5000] before sending to the API). Assembled descriptions are
# capped here so the DISCLAIMER — which must appear on every uploaded video — always
# survives that slice intact, even when the quoted script portion is very long.
MAX_DESCRIPTION_CHARS = 4_900

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-16s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("splurj")


# ── Blueprint I/O ────────────────────────────────────────────────────────────

def _validate_blueprint(data: Dict[str, Any], source: str = "") -> None:
    label = f" in {source}" if source else ""

    required_top = {"day", "format", "metadata", "voiceover", "timeline"}
    missing = required_top - set(data.keys())
    if missing:
        raise SystemExit(f"Blueprint{label} is missing required keys: {missing}")

    required_meta = {"title", "description", "tags"}
    missing_meta = required_meta - set(data["metadata"].keys())
    if missing_meta:
        raise SystemExit(f"metadata block{label} missing: {missing_meta}")

    if not isinstance(data["timeline"], list) or not data["timeline"]:
        raise SystemExit(f"timeline{label} must be a non-empty list of segment objects")

    for i, seg in enumerate(data["timeline"]):
        for key in ("start", "end", "text", "prompt", "is_short_candidate"):
            if key not in seg:
                raise SystemExit(f"timeline[{i}]{label} is missing '{key}'")

    expected_full_text = " ".join(seg["text"] for seg in data["timeline"])
    if data["voiceover"].get("full_text", "").strip() != expected_full_text.strip():
        raise SystemExit(
            f"voiceover.full_text{label} does not match the concatenation of timeline segment texts"
        )


def load_blueprint(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path) as fh:
            data = json.load(fh)
    except FileNotFoundError:
        raise SystemExit(f"Blueprint not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}")

    blueprints = data if isinstance(data, list) else [data]
    for bp in blueprints:
        _validate_blueprint(bp, source=path)
    return blueprints


def load_env() -> Dict[str, str]:
    load_dotenv()
    required = {
        "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY"),
        "ELEVENLABS_VOICE_ID": os.getenv("ELEVENLABS_VOICE_ID"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "YOUTUBE_CLIENT_SECRET": os.getenv("YOUTUBE_CLIENT_SECRET"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SystemExit(
            "Missing environment variables:\n  " + "\n  ".join(missing)
            + "\n\nCopy .env.example -> .env and fill in your keys."
        )
    return required


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:50].strip("_")


def build_description(meta_description: str, full_script: str, day: int) -> str:
    """Assemble the YouTube description.

    The DISCLAIMER must survive the uploader's description[:5000] slice on every
    upload, but real Splurj scripts can be 10k+ characters. So the whole assembled
    description is capped at MAX_DESCRIPTION_CHARS, and if it would exceed that cap,
    only the quoted script portion is truncated (with a trailing "…") — the
    disclaimer itself is never modified or truncated and always stays last.
    """
    separator = "-" * 40
    header = f"{meta_description}\n\n{separator}\nEpisode {day}\n\n"
    footer = f"\n\n{separator}\n{DISCLAIMER}"

    # Budget derived from the actual assembled fixed parts (header + quote marks +
    # separators + disclaimer), not a hardcoded offset.
    fixed_len = len(header) + len('""') + len(footer)
    available_for_script = max(MAX_DESCRIPTION_CHARS - fixed_len, 0)

    script = full_script
    if len(script) > available_for_script:
        ellipsis = "…"
        keep = max(available_for_script - len(ellipsis), 0)
        script = script[:keep] + ellipsis

    return f'{header}"{script}"{footer}'


def build_shorts_title(title: str) -> str:
    """Build the title used for Shorts uploads.

    The uploader slices title[:100] before sending to the API. Base is capped at
    92 chars so base + " #Shorts" (8 chars) is at most 100 — keeping the hashtag
    intact instead of getting chopped mid-word.
    """
    return title[:92] + " #Shorts"


def build_thumbnail_prompts(title: str) -> List[str]:
    hook_words = title.upper()
    props = ["a fat happy wallet with a big smiling face", "a brain with dollar-sign eyes"]
    return [
        (
            "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, "
            f"slightly imperfect sketchy marker lines, {prop} centered on a plain white "
            f'background, with bold ALL CAPS hand-lettered marker text at the top reading '
            f'"{hook_words}", no gradients, no shadows, no textures, no photorealism, no 3D, '
            "16:9 aspect ratio, educational YouTube explainer doodle style, YouTube thumbnail composition."
        )
        for prop in props
    ]


# ── Asset generation ─────────────────────────────────────────────────────────

def _generate_one_segment(idx, segment, directive, audio_gen, image_gen, workspace, reference_image_path):
    audio_path = workspace / f"audio_{idx:02d}.mp3"
    image_path = workspace / f"image_{idx:02d}.png"

    audio_gen.generate_segment(segment["text"], audio_path, directive=directive)
    duration = audio_gen.probe_duration(audio_path)
    image_gen.generate(segment["prompt"], image_path, reference_image_path=reference_image_path)

    return {
        "index": idx,
        "audio": audio_path,
        "image": image_path,
        "duration": duration,
        "text": segment["text"],
    }


def generate_all_assets(blueprint, audio_gen, image_gen, workspace, reference_image_path, max_workers=3):
    timeline = blueprint["timeline"]
    directive = blueprint["voiceover"].get("directive", "")
    results = {}

    logger.info("Generating assets for %d segments (max %d workers)…", len(timeline), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_generate_one_segment, i, seg, directive, audio_gen, image_gen, workspace, reference_image_path): i
            for i, seg in enumerate(timeline)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
                logger.info("  [%d/%d] Segment %d complete", len(results), len(timeline), idx + 1)
            except Exception as exc:
                logger.error("Segment %d failed: %s", idx, exc)
                raise

    return [results[i] for i in sorted(results.keys())]


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(
    blueprint: Dict[str, Any],
    env: Dict[str, str],
    skip_upload: bool = False,
    polish_script: bool = False,
    enhance_prompts: bool = False,
) -> Dict[str, Any]:
    day = blueprint["day"]
    meta = blueprint["metadata"]
    title = meta["title"]

    run_ws = WORKSPACE / f"day_{day:03d}"
    run_ws.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DAY %d  |  %s", day, title)
    logger.info("=" * 60)

    audio_gen = AudioGenerator(
        api_key=env["ELEVENLABS_API_KEY"],
        voice_id=env["ELEVENLABS_VOICE_ID"],
        model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2"),
    )
    image_gen = ImageGenerator(
        api_key=env["GEMINI_API_KEY"],
        model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"),
    )
    assembler = VideoAssembler(workspace=run_ws, assets_dir=ASSETS_DIR)

    reference_image_path = CHANNEL_DATA / "character_reference.png"
    if not reference_image_path.exists():
        logger.warning("No character_reference.png found in %s — generating without it.", CHANNEL_DATA)
        reference_image_path = None

    if polish_script or enhance_prompts:
        from engine.gemini_tools import GeminiPromptEnhancer, GeminiScriptPolisher
        if polish_script:
            logger.info("Gemini script polish…")
            blueprint = GeminiScriptPolisher(env["GEMINI_API_KEY"]).polish_blueprint(blueprint)
        if enhance_prompts:
            logger.info("Gemini prompt enhancement…")
            blueprint = GeminiPromptEnhancer(env["GEMINI_API_KEY"]).enhance_blueprint(blueprint)

    t0 = time.perf_counter()
    segments = generate_all_assets(blueprint, audio_gen, image_gen, run_ws, reference_image_path)
    logger.info("Assets ready in %.0fs", time.perf_counter() - t0)

    logger.info("Building segment clips…")
    seg_clips: List[Path] = []
    for seg in segments:
        clip_path = run_ws / f"clip_{seg['index']:02d}.mp4"
        assembler.create_segment_video(seg["image"], seg["audio"], clip_path, seg["duration"])
        seg_clips.append(clip_path)

    logger.info("Concatenating %d clips…", len(seg_clips))
    concat_path = run_ws / "raw_concat.mp4"
    assembler.concatenate_segments(seg_clips, concat_path)

    logger.info("Ambient audio mix…")
    ambient_db = float(os.getenv("AMBIENT_DB", "-15"))
    mixed_path = run_ws / "with_ambient.mp4"
    assembler.mix_ambient_audio(concat_path, mixed_path, ambient_db=ambient_db)

    logger.info("Final render…")
    output_filename = f"day_{day:03d}_{slugify(title)}.mp4"
    output_path = OUTPUT_DIR / output_filename
    assembler.finalize(mixed_path, output_path)
    logger.info("Render complete -> %s", output_path.name)

    logger.info("Extracting Shorts…")
    shorts_dir = OUTPUT_DIR / f"day_{day:03d}_shorts"
    shorts = assembler.extract_shorts(seg_clips, blueprint["timeline"], shorts_dir)
    logger.info("%d Short(s) extracted", len(shorts))

    logger.info("Generating thumbnail variants…")
    thumbnail_paths: List[Path] = []
    for i, prompt in enumerate(build_thumbnail_prompts(title)):
        thumb_path = run_ws / f"thumbnail_{i:02d}.png"
        image_gen.generate(prompt, thumb_path, reference_image_path=reference_image_path)
        thumbnail_paths.append(thumb_path)

    if not skip_upload:
        logger.info("Uploading to YouTube…")
        privacy = os.getenv("YOUTUBE_PRIVACY", "private")
        uploader = YouTubeUploader(client_secret_file=env["YOUTUBE_CLIENT_SECRET"], privacy_status=privacy)

        full_script = blueprint["voiceover"].get("full_text", "")
        description = build_description(meta["description"], full_script, day)
        response = uploader.upload(
            video_path=output_path, title=title, description=description, tags=meta.get("tags", []),
        )
        video_id = response.get("id", "???")
        uploader.set_default_thumbnail(video_id, thumbnail_paths[0])
        logger.info("Published -> https://youtube.com/watch?v=%s [%s]", video_id, privacy)

        for i, short_path in enumerate(shorts):
            short_description = build_description(f"{meta['description']} #Shorts", full_script, day)
            short_response = uploader.upload(
                video_path=short_path,
                title=build_shorts_title(title),
                description=short_description,
                tags=meta.get("tags", []),
            )
            short_id = short_response.get("id", "???")
            logger.info("Short %d published -> https://youtube.com/shorts/%s", i, short_id)
    else:
        logger.info("Upload skipped (--no-upload).")

    return {"long_form": output_path, "shorts": shorts}


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="splurj_engine", description="Splurj Engine — blueprint-driven video creator.")
    parser.add_argument("--input", "-i", required=True, metavar="FILE", help="Path to the content JSON blueprint")
    parser.add_argument("--no-upload", action="store_true", help="Skip the YouTube upload step (render only)")
    parser.add_argument("--keep-workspace", action="store_true", help="Preserve the temp workspace after the run")
    parser.add_argument("--polish-script", action="store_true", help="Run Gemini script polish pass before TTS")
    parser.add_argument("--enhance-prompts", action="store_true", help="Run Gemini prompt enhancement before image gen")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG-level logging")
    return parser


def main() -> None:
    args = build_cli().parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    blueprints = load_blueprint(args.input)
    env = load_env()

    for blueprint in blueprints:
        start = time.perf_counter()
        try:
            result = run_pipeline(
                blueprint, env,
                skip_upload=args.no_upload,
                polish_script=args.polish_script,
                enhance_prompts=args.enhance_prompts,
            )
            logger.info("Pipeline finished in %.0fs", time.perf_counter() - start)
            logger.info("Output: %s", result["long_form"].resolve())
        finally:
            if not args.keep_workspace:
                day = blueprint.get("day", 0)
                ws = WORKSPACE / f"day_{day:03d}"
                if ws.exists():
                    shutil.rmtree(ws)
                    logger.info("Workspace cleaned: %s", ws)


if __name__ == "__main__":
    main()
