"""
ElevenLabs background music generation for Splurj.
Generates a short instrumental bed per video, prompt-hash cached and
loudness-normalized so it mixes predictably under narration.
"""

import hashlib
import logging
import shutil
import time
from pathlib import Path

from elevenlabs import ElevenLabs

from engine.audio import LOUDNESS_TARGET_I, normalize_loudness

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "music"


def _prompt_cache_path(prompt: str, model: str, duration_ms: int) -> Path:
    key = hashlib.md5(f"{model}:{duration_ms}:{prompt}".encode(), usedforsecurity=False).hexdigest()
    return _CACHE_DIR / f"{key}.mp3"


class MusicGenerator:
    def __init__(self, api_key: str, model: str = "music_v2", use_cache: bool = True):
        if not api_key:
            raise EnvironmentError("ELEVENLABS_API_KEY is not set")
        self.client = ElevenLabs(api_key=api_key)
        self.model = model
        self.use_cache = use_cache

    def compose(
        self, prompt: str, duration_ms: int, output_path: Path, max_retries: int = 4
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_cache:
            cached = _prompt_cache_path(prompt, self.model, duration_ms)
            if cached.exists():
                logger.info("Cache hit  -> %s  (skipping API call)", output_path.name)
                shutil.copy2(cached, output_path)
                return output_path

        logger.info("Composing music (%s, %dms) -> %s", self.model, duration_ms, output_path.name)

        for attempt in range(1, max_retries + 1):
            try:
                audio_iter = self.client.music.compose(
                    prompt=prompt,
                    music_length_ms=duration_ms,
                    model_id=self.model,
                    force_instrumental=True,
                    output_format="mp3_44100_128",
                )
                with open(output_path, "wb") as fh:
                    for chunk in audio_iter:
                        if chunk:
                            fh.write(chunk)
                break

            except Exception as exc:
                err_str = str(exc).lower()
                is_network = any(k in err_str for k in ("getaddrinfo", "connect", "timeout", "network"))
                if is_network and attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "ElevenLabs music network error (attempt %d/%d) -- retrying in %ds: %s",
                        attempt, max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"ElevenLabs music generation failed after {attempt} attempt(s): {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size < 100:
            raise RuntimeError(f"Music file too small or missing: {output_path}")

        normalize_loudness(output_path, target_i=LOUDNESS_TARGET_I)

        if self.use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, _prompt_cache_path(prompt, self.model, duration_ms))

        logger.info("Music saved: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
        return output_path
