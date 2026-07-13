#!/usr/bin/env python3
"""
Nightly entrypoint for the Splurj Windows Scheduled Task.

Renders + uploads whatever blueprint is sitting at queue_local/next.json.
If nothing is queued, exits cleanly so the task's nightly run is a silent
no-op rather than re-publishing the last video or erroring. On success the
queued file is archived to queue_local/processed/ so an unattended run
never republishes the same blueprint twice; on failure it is left in place
so the next night's run retries it.
"""
import logging
import shutil
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
NEXT_PATH = BASE_DIR / "queue_local" / "next.json"
PROCESSED_DIR = BASE_DIR / "queue_local" / "processed"

logger = logging.getLogger("run_queued")


def run_queued(next_path: Path = NEXT_PATH, processed_dir: Path = PROCESSED_DIR) -> None:
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
