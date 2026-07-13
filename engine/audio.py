"""
ElevenLabs TTS audio generation for Splurj.
Parses voiceover directives into voice settings and generates per-segment MP3s.
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path

from elevenlabs import ElevenLabs
from elevenlabs.types import VoiceSettings

logger = logging.getLogger(__name__)

# Every segment is a separate ElevenLabs generation and generations come back
# at wildly different levels, so each one is normalized to this EBU R128
# integrated-loudness target before assembly.
LOUDNESS_TARGET_I = -16.0
LOUDNESS_TARGET_TP = -1.5
LOUDNESS_TARGET_LRA = 11.0
# Below this integrated loudness there is no signal worth normalizing —
# make-up gain would only amplify the noise floor.
SILENCE_FLOOR_LUFS = -50.0


def _measure_loudnorm_stats(audio_path: Path) -> dict:
    """First loudnorm pass: measure the file's loudness stats (JSON on stderr)."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", str(audio_path),
        "-af",
        f"loudnorm=I={LOUDNESS_TARGET_I}:TP={LOUDNESS_TARGET_TP}:"
        f"LRA={LOUDNESS_TARGET_LRA}:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Loudness measurement failed on {audio_path.name}: {result.stderr}")
    # The JSON block is not the last thing on stderr — muxer stats follow it —
    # so take the last brace-delimited block anywhere in the output.
    matches = re.findall(r"\{[^{}]*\}", result.stderr)
    if not matches:
        raise RuntimeError(f"No loudnorm stats in ffmpeg output for {audio_path.name}")
    return json.loads(matches[-1])


def measure_integrated_loudness(audio_path: Path) -> float:
    """Integrated loudness (LUFS) of an audio file, per EBU R128."""
    return float(_measure_loudnorm_stats(audio_path)["input_i"])


def normalize_loudness(audio_path: Path, target_i: float = LOUDNESS_TARGET_I) -> Path:
    """Normalize a file in place to the target integrated loudness.

    Two-pass linear loudnorm: pure make-up gain, so the delivery dynamics
    within a segment are preserved while segments land on one level.
    Near-silent audio is returned untouched.
    """
    stats = _measure_loudnorm_stats(audio_path)
    input_i = float(stats["input_i"])
    if input_i < SILENCE_FLOOR_LUFS:
        logger.warning(
            "Skipping loudness normalization on near-silent audio: %s (%.1f LUFS)",
            audio_path.name, input_i,
        )
        return audio_path

    normalized = audio_path.with_suffix(".norm.mp3")
    filter_arg = (
        f"loudnorm=I={target_i}:TP={LOUDNESS_TARGET_TP}:LRA={LOUDNESS_TARGET_LRA}:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats", "-i", str(audio_path),
        "-af", filter_arg,
        # loudnorm resamples to 192 kHz internally; restore the source rate.
        "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "192k",
        str(normalized),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Loudness normalization failed on {audio_path.name}: {result.stderr}")
    os.replace(normalized, audio_path)
    logger.debug("Normalized %s: %.1f -> %.1f LUFS", audio_path.name, input_i, target_i)
    return audio_path


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

        normalize_loudness(output_path)
        logger.info("Audio saved: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
        return output_path

    def probe_duration(self, audio_path: Path) -> float:
        cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(audio_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed on {audio_path.name}: {result.stderr}")
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
