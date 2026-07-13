#!/usr/bin/env python3
"""
Nightly entrypoint for the Splurj Windows Scheduled Task.

Renders + uploads whatever blueprint is sitting at queue_local/next.json.
If nothing is queued, tries to auto-draft one from the approved citation
bank (engine/topic_picker.py + splurj_draft.py) before giving up -- this is
what makes the nightly task fully autonomous rather than requiring a human
to drop in a blueprint every day. A failed draft (citation QA exhausted,
API error) is a silent no-op for the night, same as an empty queue, rather
than crashing the scheduled task. On a successful render the queued file is
archived to queue_local/processed/ so an unattended run never republishes
the same blueprint twice; on a failed render it is left in place so the
next night's run retries it.
"""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
NEXT_PATH = BASE_DIR / "queue_local" / "next.json"
PROCESSED_DIR = BASE_DIR / "queue_local" / "processed"

logger = logging.getLogger("run_queued")


def _auto_draft(next_path: Path) -> None:
    import splurj_draft
    from splurj_engine import load_env

    try:
        # load_env() raises SystemExit (not Exception) on missing keys -- an
        # unattended nightly run must skip silently on that too, not crash.
        env = load_env()
        day = splurj_draft._read_next_day()
        blueprint = splurj_draft.draft_blueprint(day, env["GEMINI_API_KEY"])
    except (Exception, SystemExit) as exc:
        logger.warning("Auto-draft failed -- skipping tonight's render: %s", exc)
        return

    next_path.parent.mkdir(parents=True, exist_ok=True)
    next_path.write_text(json.dumps(blueprint, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    splurj_draft._write_next_day(day + 1)
    logger.info("Auto-drafted day %d '%s' -> %s", day, blueprint["metadata"]["title"], next_path)


def run_queued(next_path: Path = NEXT_PATH, processed_dir: Path = PROCESSED_DIR) -> None:
    if not next_path.exists():
        _auto_draft(next_path)

    if not next_path.exists():
        logger.info("No queued blueprint at %s -- nothing to render tonight.", next_path)
        return

    from splurj_engine import load_blueprint, load_env, run_pipeline

    env = load_env()
    blueprints = load_blueprint(str(next_path))

    for blueprint in blueprints:
        run_pipeline(blueprint, env, skip_upload=False)

    processed_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = processed_dir / f"{stamp}_{next_path.name}"
    shutil.move(str(next_path), str(archived))
    logger.info("Archived processed blueprint -> %s", archived)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    run_queued()
