"""
ElevenLabs TTS audio generation for Splurj.
Parses voiceover directives into voice settings and generates per-segment MP3s.
"""

import json
import logging
import subprocess
import time
from pathlib import Path

from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

logger = logging.getLogger(__name__)


def parse_directive(directive: str) -> VoiceSettings:
    """Map a natural-language voice directive to ElevenLabs voice settings."""
    d = directive.lower()

    stability = 0.55
    similarity_boost = 0.80
    style = 0.0
    use_speaker_boost = True

    if any(w in d for w in ("gritty", "rough", "gravel", "raw", "dark", "worn")):
        stability = 0.30
        style = 0.15

    if any(w in d for w in ("flat", "detached", "monotone", "deadpan", "cold")):
        stability = 0.75
        style = 0.0

    if any(w in d for w in ("deep", "bass", "low", "gravelly")):
        similarity_boost = 0.90

    if any(w in d for w in ("energetic", "urgent", "intense", "sharp")):
        style = 0.30
        stability = 0.35

    if any(w in d for w in ("calm", "steady", "measured", "slow", "curious")):
        stability = 0.70
        style = 0.0

    return VoiceSettings(
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
    )


class AudioGenerator:
    def __init__(self, api_key: str, voice_id: str, model: str = "eleven_turbo_v2"):
        if not voice_id:
            raise ValueError(
                "voice_id is required — set ELEVENLABS_VOICE_ID in .env. "
                "Browse voices at https://elevenlabs.io/voice-library and pick one "
                "matching Splurj's calm/curious/2nd-person tone."
            )
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.model = model

    def generate_segment(
        self, text: str, output_path: Path, directive: str = "", max_retries: int = 4
    ) -> Path:
        if not text.strip():
            raise ValueError("Cannot generate audio from empty text")

        voice_settings = parse_directive(directive)
        logger.info("Generating audio -- %d chars -> %s", len(text), output_path.name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, max_retries + 1):
            try:
                audio_iter = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=self.voice_id,
                    model_id=self.model,
                    voice_settings=voice_settings,
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
                        "ElevenLabs network error (attempt %d/%d) -- retrying in %ds: %s",
                        attempt, max_retries, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"ElevenLabs API error after {attempt} attempt(s): {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size < 100:
            raise RuntimeError(f"Audio file too small or missing: {output_path}")

        logger.info("Audio saved: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
        return output_path

    def probe_duration(self, audio_path: Path) -> float:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed on {audio_path.name}: {result.stderr}")
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
