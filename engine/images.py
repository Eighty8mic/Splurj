"""
Gemini image generation for Splurj.
Generates doodle-style scene images, optionally conditioned on a locked
character/prop reference image for visual consistency across a video.

Prompt-hash image cache: generated images are stored in cache/images/ and
reused on subsequent runs with the same prompt — eliminates redundant API
calls, and is what makes "hold this scene" segments in a blueprint cheap.
"""

import hashlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "images"


def _prompt_cache_path(prompt: str, model: str, ref_digest: str = "") -> Path:
    key = hashlib.md5(f"{model}:{ref_digest}:{prompt}".encode(), usedforsecurity=False).hexdigest()
    return _CACHE_DIR / f"{key}.png"


class ImageGenerator:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-image", use_cache: bool = True):
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is not set")
        self.api_key = api_key
        self.model = model
        self.use_cache = use_cache

    def generate(
        self,
        prompt: str,
        output_path: Path,
        reference_image_path: Optional[Path] = None,
        max_retries: int = 4,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ref_bytes: Optional[bytes] = None
        ref_digest = ""
        if reference_image_path is not None:
            ref_bytes = Path(reference_image_path).read_bytes()
            ref_digest = hashlib.md5(ref_bytes, usedforsecurity=False).hexdigest()

        if self.use_cache:
            cached = _prompt_cache_path(prompt, self.model, ref_digest)
            if cached.exists():
                logger.info("Cache hit  -> %s  (skipping API call)", output_path.name)
                shutil.copy2(cached, output_path)
                return output_path

        logger.info("Generating image (%s) -> %s", self.model, output_path.name)
        logger.debug("Prompt: %s", prompt[:120])

        self._generate_gemini(prompt, output_path, ref_bytes, max_retries)

        if self.use_cache:
            self._write_cache(prompt, output_path, ref_digest)

        return output_path

    def _write_cache(self, prompt: str, output_path: Path, ref_digest: str) -> None:
        """Copy the freshly generated image into the prompt cache.

        Uses a unique temp file + os.replace so that two threads generating the
        same (prompt, reference) concurrently — e.g. under splurj_engine's
        ThreadPoolExecutor(max_workers=3) — never observe a partially-written
        cache file. A cache-write failure must never fail image generation
        itself, since the caller already has a valid output_path.
        """
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path = _prompt_cache_path(prompt, self.model, ref_digest)
            tmp_path = cache_path.parent / f"{cache_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
            shutil.copy2(output_path, tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning("Failed to write prompt cache for %s: %s", output_path.name, exc)

    def _generate_gemini(
        self,
        prompt: str,
        output_path: Path,
        ref_bytes: Optional[bytes],
        max_retries: int,
    ) -> None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        contents = [prompt]
        if ref_bytes is not None:
            contents.append(types.Part.from_bytes(data=ref_bytes, mime_type="image/png"))

        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )

                image_data: Optional[bytes] = None
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        image_data = part.inline_data.data
                        break

                if not image_data:
                    raise RuntimeError("Gemini returned no image data in response")

                output_path.write_bytes(image_data)
                logger.info("Image saved: %s (%.1f KB)", output_path.name, len(image_data) / 1024)
                return

            except Exception as exc:
                err_str = str(exc).lower()
                is_rate = any(k in err_str for k in ("429", "quota", "rate"))
                if is_rate and attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini rate limit (attempt %d/%d) -- waiting %ds: %s",
                        attempt, max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Gemini image generation failed after {attempt} attempt(s): {exc}"
                    ) from exc
