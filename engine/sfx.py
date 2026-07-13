"""
ElevenLabs sound-effect generation and mixing for Splurj.
Short cue clips (a coin drop, a tap chime) are generated from a text
description, prompt-hash cached, and overlaid onto a segment's voice
track at a specific timestamp — punctuation, not a second narrator.
"""

import hashlib
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

from elevenlabs import ElevenLabs

from engine.audio import normalize_loudness

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "sfx"

# Sits below the -16 LUFS narration target so cues read as punctuation,
# not a second voice competing with the narrator.
SFX_LOUDNESS_TARGET_I = -20.0


def _prompt_cache_path(text: str, model: str, duration_seconds: Optional[float]) -> Path:
    key = hashlib.md5(f"{model}:{duration_seconds}:{text}".encode(), usedforsecurity=False).hexdigest()
    return _CACHE_DIR / f"{key}.mp3"


class SoundEffectGenerator:
    def __init__(self, api_key: str, model: str = "eleven_text_to_sound_v2", use_cache: bool = True):
        if not api_key:
            raise EnvironmentError("ELEVENLABS_API_KEY is not set")
        self.client = ElevenLabs(api_key=api_key)
        self.model = model
        self.use_cache = use_cache

    def generate(
        self,
        text: str,
        output_path: Path,
        duration_seconds: Optional[float] = None,
        max_retries: int = 4,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.use_cache:
            cached = _prompt_cache_path(text, self.model, duration_seconds)
            if cached.exists():
                logger.info("Cache hit  -> %s  (skipping API call)", output_path.name)
                shutil.copy2(cached, output_path)
                return output_path

        logger.info("Generating sound effect -> %s", output_path.name)
        logger.debug("Cue: %s", text[:120])

        for attempt in range(1, max_retries + 1):
            try:
                audio_iter = self.client.text_to_sound_effects.convert(
                    text=text,
                    duration_seconds=duration_seconds,
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
                        "ElevenLabs SFX network error (attempt %d/%d) -- retrying in %ds: %s",
                        attempt, max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"ElevenLabs SFX generation failed after {attempt} attempt(s): {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size < 100:
            raise RuntimeError(f"Sound effect file too small or missing: {output_path}")

        normalize_loudness(output_path, target_i=SFX_LOUDNESS_TARGET_I)

        if self.use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, _prompt_cache_path(text, self.model, duration_seconds))

        logger.info("Sound effect saved: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
        return output_path


def overlay_cues(voice_audio_path: Path, cues: List[Tuple[Path, float]], output_path: Path) -> Path:
    """Mix sound-effect cues onto a voice track at their given offsets.

    Output duration always matches the voice track (amix duration=first) —
    a cue starting near the segment's end is simply truncated rather than
    extending the clip and desyncing it from the video length set elsewhere.
    """
    if not cues:
        shutil.copy2(voice_audio_path, output_path)
        return output_path

    inputs = ["-i", str(voice_audio_path)]
    filter_parts = []
    mix_labels = ["0:a"]
    for i, (sfx_path, at_seconds) in enumerate(cues, start=1):
        inputs += ["-i", str(sfx_path)]
        delay_ms = max(int(at_seconds * 1000), 0)
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[s{i}]")
        mix_labels.append(f"s{i}")

    mix_inputs = "".join(f"[{label}]" for label in mix_labels)
    filter_parts.append(f"{mix_inputs}amix=inputs={len(mix_labels)}:duration=first:normalize=0[aout]")
    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SFX overlay failed for {voice_audio_path.name}: {result.stderr}")

    logger.debug("Overlaid %d SFX cue(s) onto %s", len(cues), voice_audio_path.name)
    return output_path
