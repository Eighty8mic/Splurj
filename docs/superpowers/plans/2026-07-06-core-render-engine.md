# Splurj Core Render Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Splurj render engine — a local, blueprint-driven pipeline that turns a JSON blueprint into a rendered, uploaded 16:9 long-form video plus 2-3 auto-cut Shorts, ported from Grafyte's proven architecture (`C:\Users\School\Desktop\Grafyte`).

**Architecture:** Five independent engine modules (audio, images, video, youtube, gemini_tools) called by one orchestrator (`splurj_engine.py`) that reads a blueprint JSON, fans out audio+image generation per segment, assembles clips with FFmpeg, and uploads to YouTube. No Supabase/dashboard/citation-bank dependency in this plan — those are separate plans layered on top later.

**Tech Stack:** Python 3.11+, ElevenLabs SDK (TTS), google-genai SDK (Gemini image gen), FFmpeg (subprocess), google-api-python-client (YouTube Data API v3), pytest (testing).

## Global Constraints

- Output video canvas is 1920×1080 (16:9) — Splurj's format, not Grafyte's 9:16.
- Image generation model: `gemini-3.1-flash-image` (not Grafyte's pro-tier default) — cost-driven choice, confirmed sufficient for flat 2D doodle style.
- Reference-image conditioning uses `contents=[prompt_text, types.Part.from_bytes(data=ref_bytes, mime_type="image/png")]` passed to `client.models.generate_content` — confirmed against the real python-genai SDK source and the v1beta REST discovery spec (`models.generateContent` is current; no separate "interactions" resource exists in the discovery spec despite some doc pages referencing one).
- YouTube upload category is `"27"` (Education), not Grafyte's `"24"` (Entertainment).
- Every video description must hard-append `"This video is for education and entertainment only. It is not financial advice."` — never sourced from generated text, so it can't be dropped.
- No external network calls in unit tests — mock `ElevenLabs`, `google.genai.Client`, and `googleapiclient.discovery.build` at their import boundary. FFmpeg itself is a real local system dependency (already installed) and IS invoked for real in tests that touch `video.py`.
- Modules under `engine/` are tested from `tests/engine/`; root-level modules (`splurj_engine.py`, `generate_character_reference.py`) are tested from `tests/` directly. `tests/__init__.py` and `tests/engine/__init__.py` (Task 1) make both `engine.*` and root-level modules importable under pytest's default import mode without any sys.path hacks — don't delete them.
- Every task ends with a real `pytest` run showing PASS before its commit step.

---

### Task 1: Project scaffold + `engine/audio.py`

**Files:**
- Create: `engine/__init__.py`
- Create: `engine/audio.py`
- Create: `tests/__init__.py`, `tests/engine/__init__.py`
- Create: `tests/engine/test_audio.py`
- Create: `requirements.txt`
- Create: `.env.example`
- Create directories: `queue_local/`, `output/`, `workspace/`, `assets/ambient/`, `cache/images/`, `channel_data/` (each with a `.gitkeep`)

**Interfaces:**
- Produces: `engine.audio.parse_directive(directive: str) -> VoiceSettings`, `engine.audio.AudioGenerator(api_key: str, voice_id: str, model: str = "eleven_turbo_v2")` with methods `.generate_segment(text: str, output_path: Path, directive: str = "", max_retries: int = 4) -> Path` and `.probe_duration(audio_path: Path) -> float`.

- [ ] **Step 1: Create the scaffold**

```bash
mkdir -p engine tests/engine queue_local output workspace assets/ambient cache/images channel_data
touch queue_local/.gitkeep output/.gitkeep workspace/.gitkeep assets/ambient/.gitkeep cache/images/.gitkeep channel_data/.gitkeep
touch engine/__init__.py tests/__init__.py tests/engine/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
python-dotenv>=1.0.0
requests>=2.31.0
elevenlabs>=1.3.0
google-genai>=1.0.0
google-api-python-client>=2.120.0
google-auth-httplib2>=0.2.0
google-auth-oauthlib>=1.2.0
pytest>=8.0.0
```

- [ ] **Step 3: Write `.env.example`**

```env
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=
ELEVENLABS_MODEL=eleven_turbo_v2

GEMINI_API_KEY=
GEMINI_IMAGE_MODEL=gemini-3.1-flash-image

YOUTUBE_CLIENT_SECRET=./client_secret.json
YOUTUBE_PRIVACY=private
YOUTUBE_CATEGORY_ID=27

AMBIENT_DB=-15
SHORT_MIN_SEGMENTS=3
SHORT_MAX_SEGMENTS=4
```

- [ ] **Step 4: Write the failing tests** — `tests/engine/test_audio.py`

```python
import json
from unittest.mock import MagicMock, patch

import pytest

from engine.audio import AudioGenerator, parse_directive


def test_parse_directive_calm_curious_lowers_stability_variance():
    settings = parse_directive("Calm, curious, a little conspiratorial. Unhurried.")
    assert settings.stability == 0.70
    assert settings.style == 0.0


def test_parse_directive_default_when_no_keywords_match():
    settings = parse_directive("")
    assert settings.stability == 0.55
    assert settings.similarity_boost == 0.80


def test_audio_generator_requires_voice_id():
    with pytest.raises(ValueError, match="voice_id is required"):
        AudioGenerator(api_key="key", voice_id="")


def test_generate_segment_rejects_empty_text(tmp_path):
    with patch("engine.audio.ElevenLabs"):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        with pytest.raises(ValueError, match="empty text"):
            gen.generate_segment("   ", tmp_path / "out.mp3")


def test_generate_segment_writes_audio_bytes(tmp_path):
    fake_client = MagicMock()
    fake_client.text_to_speech.convert.return_value = [b"x" * 200]

    with patch("engine.audio.ElevenLabs", return_value=fake_client):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out, directive="calm")

    assert result == out
    assert out.read_bytes() == b"x" * 200


def test_generate_segment_retries_on_network_error_then_succeeds(tmp_path):
    fake_client = MagicMock()
    fake_client.text_to_speech.convert.side_effect = [
        ConnectionError("getaddrinfo failed"),
        [b"x" * 200],
    ]

    with patch("engine.audio.ElevenLabs", return_value=fake_client), \
         patch("engine.audio.time.sleep") as mock_sleep:
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        out = tmp_path / "audio_00.mp3"
        result = gen.generate_segment("Hello there.", out)

    assert result == out
    assert out.read_bytes() == b"x" * 200
    mock_sleep.assert_called_once()


def test_probe_duration_parses_ffprobe_json(tmp_path):
    fake_result = MagicMock(returncode=0, stdout=json.dumps({"format": {"duration": "12.345"}}))
    with patch("engine.audio.ElevenLabs"), patch("engine.audio.subprocess.run", return_value=fake_result):
        gen = AudioGenerator(api_key="key", voice_id="abc123")
        duration = gen.probe_duration(tmp_path / "audio_00.mp3")
    assert duration == pytest.approx(12.345)
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `pytest tests/engine/test_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.audio'`

- [ ] **Step 6: Write `engine/audio.py`**

```python
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
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/engine/test_audio.py -v`
Expected: 7 passed

- [ ] **Step 8: Commit**

```bash
git add engine/__init__.py engine/audio.py tests/__init__.py tests/engine/__init__.py tests/engine/test_audio.py requirements.txt .env.example queue_local output workspace assets channel_data cache
git commit -m "feat: scaffold Splurj engine project + port ElevenLabs TTS module"
```

---

### Task 2: `engine/images.py` (Gemini image gen + character reference conditioning)

**Files:**
- Create: `engine/images.py`
- Create: `tests/engine/test_images.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `engine.images.ImageGenerator(api_key: str, model: str = "gemini-3.1-flash-image", use_cache: bool = True)` with `.generate(prompt: str, output_path: Path, reference_image_path: Optional[Path] = None, max_retries: int = 4) -> Path`. Also `engine.images._prompt_cache_path(prompt: str, model: str) -> Path` and module-level `engine.images._CACHE_DIR` (a `Path`, monkeypatchable by tests).

- [ ] **Step 1: Write the failing tests** — `tests/engine/test_images.py`

```python
from unittest.mock import MagicMock, patch

import pytest

from engine.images import ImageGenerator, _prompt_cache_path


def test_requires_api_key():
    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        ImageGenerator(api_key="")


def test_prompt_cache_path_is_stable_for_same_prompt_and_model():
    p1 = _prompt_cache_path("a doodle cat", "gemini-3.1-flash-image")
    p2 = _prompt_cache_path("a doodle cat", "gemini-3.1-flash-image")
    p3 = _prompt_cache_path("a doodle dog", "gemini-3.1-flash-image")
    assert p1 == p2
    assert p1 != p3


def _fake_image_response(data: bytes) -> MagicMock:
    fake_part = MagicMock(inline_data=MagicMock(data=data))
    return MagicMock(candidates=[MagicMock(content=MagicMock(parts=[fake_part]))])


def test_generate_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key")

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = (
            _fake_image_response(b"\x89PNG-fake-bytes-padding-000000")
        )
        out1 = tmp_path / "image_00.png"
        gen.generate("a doodle cat, flat colors", out1)
        out2 = tmp_path / "image_01.png"
        gen.generate("a doodle cat, flat colors", out2)  # identical prompt -> cache hit

    assert out1.read_bytes() == out2.read_bytes()
    assert mock_client_cls.return_value.models.generate_content.call_count == 1


def test_generate_passes_reference_image_alongside_prompt_text(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key", use_cache=False)

    ref_path = tmp_path / "character_reference.png"
    ref_path.write_bytes(b"\x89PNG-reference-bytes-padding-00")

    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = (
            _fake_image_response(b"\x89PNG-fake-scene-bytes-padding-0")
        )
        gen.generate(
            "a doodle cat holding a wallet",
            tmp_path / "image_00.png",
            reference_image_path=ref_path,
        )

    call_kwargs = mock_client_cls.return_value.models.generate_content.call_args.kwargs
    contents = call_kwargs["contents"]
    assert contents[0] == "a doodle cat holding a wallet"
    assert len(contents) == 2  # prompt text + one reference-image Part


def test_generate_retries_on_rate_limit_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key", use_cache=False)

    with patch("google.genai.Client") as mock_client_cls, \
         patch("engine.images.time.sleep") as mock_sleep:
        mock_client_cls.return_value.models.generate_content.side_effect = [
            RuntimeError("429 rate limit"),
            _fake_image_response(b"\x89PNG-fake-bytes-padding-000000"),
        ]
        gen.generate("a doodle cat", tmp_path / "image_00.png")

    mock_sleep.assert_called_once()


def test_generate_raises_after_exhausting_retries_on_persistent_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("engine.images._CACHE_DIR", tmp_path / "cache")
    gen = ImageGenerator(api_key="key", use_cache=False)

    with patch("google.genai.Client") as mock_client_cls, \
         patch("engine.images.time.sleep"):
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("429 rate limit")
        with pytest.raises(RuntimeError, match="failed after"):
            gen.generate("a doodle cat", tmp_path / "image_00.png", max_retries=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_images.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.images'`

- [ ] **Step 3: Write `engine/images.py`**

```python
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
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent / "cache" / "images"


def _prompt_cache_path(prompt: str, model: str) -> Path:
    key = hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()
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

        if self.use_cache:
            cached = _prompt_cache_path(prompt, self.model)
            if cached.exists():
                logger.info("Cache hit  -> %s  (skipping API call)", output_path.name)
                shutil.copy2(cached, output_path)
                return output_path

        logger.info("Generating image (%s) -> %s", self.model, output_path.name)
        logger.debug("Prompt: %s", prompt[:120])

        self._generate_gemini(prompt, output_path, reference_image_path, max_retries)

        if self.use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, _prompt_cache_path(prompt, self.model))

        return output_path

    def _generate_gemini(
        self,
        prompt: str,
        output_path: Path,
        reference_image_path: Optional[Path],
        max_retries: int,
    ) -> None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        contents = [prompt]
        if reference_image_path is not None:
            ref_bytes = Path(reference_image_path).read_bytes()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_images.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add engine/images.py tests/engine/test_images.py
git commit -m "feat: add Gemini image generation with reference-image conditioning and prompt cache"
```

---

### Task 3: `generate_character_reference.py` (one-time character sheet CLI)

**Files:**
- Create: `generate_character_reference.py`
- Create: `tests/test_generate_character_reference.py`

**Interfaces:**
- Consumes: `engine.images.ImageGenerator` (Task 2).
- Produces: `generate_character_reference.build_prompt() -> str`, `generate_character_reference.main(argv: list[str] | None = None) -> Path` — writes to `channel_data/character_reference.png` by default.

- [ ] **Step 1: Write the failing tests** — `tests/test_generate_character_reference.py`

```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from generate_character_reference import build_prompt, main


def test_build_prompt_includes_style_anchor_and_neutral_pose():
    prompt = build_prompt()
    assert "Hand-drawn 2D doodle cartoon animation" in prompt
    assert "neutral pose" in prompt.lower()
    assert "white background" in prompt.lower()


def test_main_writes_to_channel_data_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "key")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "channel_data").mkdir()

    with patch("generate_character_reference.ImageGenerator") as mock_gen_cls:
        mock_gen_cls.return_value.generate.side_effect = (
            lambda prompt, output_path, **kw: output_path.write_bytes(b"\x89PNG-ref") or output_path
        )
        result = main([])

    assert result == tmp_path / "channel_data" / "character_reference.png"
    assert result.read_bytes() == b"\x89PNG-ref"


def test_main_raises_without_gemini_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "channel_data").mkdir()

    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        main([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generate_character_reference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_character_reference'`

- [ ] **Step 3: Write `generate_character_reference.py`**

```python
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
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = gen.generate(build_prompt(), output_path)
    print(f"Character reference saved -> {result}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generate_character_reference.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add generate_character_reference.py tests/test_generate_character_reference.py
git commit -m "feat: add one-time character reference sheet generator"
```

---

### Task 4: `engine/video.py` — 16:9 assembly (segment clip, concat, ambient mix, finalize)

FFmpeg is a real required local dependency (not mocked) — these are integration tests using tiny real fixtures generated by ffmpeg's `lavfi` test sources, so no image/audio library dependency is needed.

**Files:**
- Create: `engine/video.py`
- Create: `tests/engine/test_video.py`
- Create: `tests/conftest.py` (project-level, so `fixture_image`/`fixture_audio` are reusable by Task 8's orchestrator test too)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `engine.video.probe_video_resolution(path: Path) -> tuple[int, int]`, `engine.video.VideoAssembler(workspace: Path, assets_dir: Path)` with `.create_segment_video(image_path, audio_path, output_path, duration) -> Path`, `.concatenate_segments(segment_paths: list[Path], output_path: Path) -> Path`, `.get_ambient_track() -> Optional[Path]`, `.mix_ambient_audio(video_path, output_path, ambient_db=-15.0) -> Path`, `.finalize(input_path, output_path) -> Path`.

- [ ] **Step 1: Write shared fixtures** — `tests/conftest.py`

```python
import subprocess
from pathlib import Path

import pytest


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def fixture_image(tmp_path) -> Path:
    """A tiny real 64x64 blue PNG, generated by ffmpeg's lavfi test source."""
    path = tmp_path / "fixture_image.png"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64", "-frames:v", "1", str(path)])
    return path


@pytest.fixture
def fixture_audio(tmp_path) -> Path:
    """A tiny real 1-second silent MP3, generated by ffmpeg's lavfi test source."""
    path = tmp_path / "fixture_audio.mp3"
    _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "1", str(path)])
    return path
```

- [ ] **Step 2: Write the failing tests** — `tests/engine/test_video.py`

```python
from pathlib import Path

import pytest

from engine.video import VideoAssembler, probe_video_resolution


def test_create_segment_video_outputs_16x9_canvas(tmp_path, fixture_image, fixture_audio):
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")
    out = tmp_path / "clip_00.mp4"

    result = assembler.create_segment_video(fixture_image, fixture_audio, out, duration=1.0)

    assert result == out
    assert out.exists()
    assert probe_video_resolution(out) == (1920, 1080)


def test_concatenate_segments_combines_clips(tmp_path, fixture_image, fixture_audio):
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")
    clip_a = assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "a.mp4", 1.0)
    clip_b = assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "b.mp4", 1.0)

    out = tmp_path / "concat.mp4"
    result = assembler.concatenate_segments([clip_a, clip_b], out)

    assert result == out
    assert out.exists()


def test_get_ambient_track_returns_none_when_empty(tmp_path):
    assets_dir = tmp_path / "assets"
    (assets_dir / "ambient").mkdir(parents=True)
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=assets_dir)

    assert assembler.get_ambient_track() is None


def test_mix_ambient_audio_falls_back_to_copy_when_no_track(tmp_path, fixture_image, fixture_audio):
    assets_dir = tmp_path / "assets"
    (assets_dir / "ambient").mkdir(parents=True)
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=assets_dir)

    clip = assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "clip.mp4", 1.0)
    out = tmp_path / "mixed.mp4"
    result = assembler.mix_ambient_audio(clip, out)

    assert result == out
    assert out.read_bytes() == clip.read_bytes()


def test_mix_ambient_audio_mixes_when_track_present(tmp_path, fixture_image, fixture_audio):
    assets_dir = tmp_path / "assets"
    ambient_dir = assets_dir / "ambient"
    ambient_dir.mkdir(parents=True)
    # Reuse the silent fixture as a stand-in ambient track — real audio content
    # doesn't matter for this test, only that the mix path runs and produces output.
    import shutil
    shutil.copy2(fixture_audio, ambient_dir / "drone.mp3")

    assembler = VideoAssembler(workspace=tmp_path, assets_dir=assets_dir)
    clip = assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "clip.mp4", 1.0)
    out = tmp_path / "mixed.mp4"
    result = assembler.mix_ambient_audio(clip, out, ambient_db=-15.0)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_finalize_produces_exact_16x9_canvas(tmp_path, fixture_image, fixture_audio):
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")
    clip = assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "clip.mp4", 1.0)

    out = tmp_path / "final.mp4"
    result = assembler.finalize(clip, out)

    assert result == out
    assert probe_video_resolution(out) == (1920, 1080)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/engine/test_video.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.video'`

- [ ] **Step 4: Write `engine/video.py`**

```python
"""
FFmpeg video assembly module for Splurj — 16:9 long-form output.

Pipeline stages:
  1. create_segment_video  — still image + audio -> MP4 clip (Ken Burns zoom)
  2. concatenate_segments   — clips -> single timeline
  3. mix_ambient_audio      — overlay looping drone at -15 dB (optional)
  4. finalize               — quality encode, pad to 1920x1080, faststart
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _run(cmd: List[str], label: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        snippet = result.stderr[-3000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"[{label}] failed (exit {result.returncode}):\n{snippet}")


def probe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def probe_video_resolution(path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {result.stderr}")
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


class VideoAssembler:
    def __init__(self, workspace: Path, assets_dir: Path):
        self.workspace = workspace
        self.assets_dir = assets_dir
        self.ambient_dir = assets_dir / "ambient"

    def create_segment_video(
        self, image_path: Path, audio_path: Path, output_path: Path, duration: float
    ) -> Path:
        frames = max(1, int(duration * 30))
        zoom_vf = (
            f"scale=1920:1080:force_original_aspect_ratio=increase,"
            f"crop=1920:1080,"
            f"zoompan=z='min(zoom+0.0001,1.04)':d={frames}"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-framerate", "30",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-vf", zoom_vf,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            "-movflags", "+faststart",
            str(output_path),
        ]
        _run(cmd, f"segment:{output_path.name}")
        logger.info("Segment done: %s", output_path.name)
        return output_path

    def concatenate_segments(self, segment_paths: List[Path], output_path: Path) -> Path:
        concat_file = self.workspace / "concat_list.txt"
        with open(concat_file, "w") as fh:
            for seg in segment_paths:
                fh.write(f"file '{seg.resolve()}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)]
        _run(cmd, "concat")
        logger.info("Concatenated %d segments -> %s", len(segment_paths), output_path.name)
        return output_path

    def get_ambient_track(self) -> Optional[Path]:
        if not self.ambient_dir.exists():
            return None
        exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
        tracks = sorted(f for f in self.ambient_dir.iterdir() if f.suffix.lower() in exts)
        return tracks[0] if tracks else None

    def mix_ambient_audio(self, video_path: Path, output_path: Path, ambient_db: float = -15.0) -> Path:
        ambient_track = self.get_ambient_track()

        if ambient_track is None:
            logger.warning("No ambient tracks in %s — skipping ambient mix.", self.ambient_dir)
            shutil.copy2(video_path, output_path)
            return output_path

        logger.info("Mixing ambient '%s' at %.0f dB", ambient_track.name, ambient_db)
        filter_graph = (
            f"[1:a]volume={ambient_db}dB[amb];"
            f"[0:a][amb]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-stream_loop", "-1",
            "-i", str(ambient_track),
            "-filter_complex", filter_graph,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]
        _run(cmd, "ambient_mix")
        logger.info("Ambient mix saved: %s", output_path.name)
        return output_path

    def finalize(self, input_path: Path, output_path: Path) -> Path:
        scale_pad = (
            "scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", scale_pad,
            "-c:v", "libx264",
            "-profile:v", "high",
            "-level:v", "4.0",
            "-crf", "18",
            "-preset", "slow",
            "-r", "30",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            "-pix_fmt", "yuv420p",
            str(output_path),
        ]
        _run(cmd, "finalize")
        size_mb = output_path.stat().st_size / 1_000_000
        logger.info("Final render: %s (%.1f MB)", output_path.name, size_mb)
        return output_path
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/engine/test_video.py -v`
Expected: 6 passed (may take ~10-20s — real ffmpeg encodes are happening)

- [ ] **Step 6: Commit**

```bash
git add engine/video.py tests/engine/test_video.py tests/conftest.py
git commit -m "feat: add 16:9 FFmpeg video assembly (segment clip, concat, ambient mix, finalize)"
```

---

### Task 5: `engine/video.py` — Shorts auto-cut

Adds Shorts extraction to the same `VideoAssembler` class: groups contiguous `is_short_candidate` segments into standalone 1080×1920 vertical clips with a burned-in ALL-CAPS caption. This machine is Windows (per environment), so `drawtext` uses an explicit Windows font file path rather than relying on fontconfig, which isn't reliably present in ffmpeg "essentials" builds.

**Files:**
- Modify: `engine/video.py` (append `extract_shorts` + `_find_candidate_runs` to `VideoAssembler`)
- Modify: `tests/engine/test_video.py` (append tests)

**Interfaces:**
- Consumes: `VideoAssembler` from Task 4 (same class, same file).
- Produces: `VideoAssembler.extract_shorts(segment_clips: List[Path], segments: List[dict], output_dir: Path, font_path: str = "C:/Windows/Fonts/arialbd.ttf") -> List[Path]`, `VideoAssembler._find_candidate_runs(segments: List[dict]) -> List[Tuple[int, int]]` (static method).

- [ ] **Step 1: Write the failing tests** — append to `tests/engine/test_video.py`

```python
from engine.video import VideoAssembler


def test_find_candidate_runs_groups_contiguous_true_segments():
    segments = [
        {"is_short_candidate": True},
        {"is_short_candidate": True},
        {"is_short_candidate": False},
        {"is_short_candidate": False},
        {"is_short_candidate": True},
        {"is_short_candidate": True},
        {"is_short_candidate": True},
    ]
    runs = VideoAssembler._find_candidate_runs(segments)
    assert runs == [(0, 1), (4, 6)]


def test_find_candidate_runs_returns_empty_when_none_marked():
    segments = [{"is_short_candidate": False}, {"is_short_candidate": False}]
    assert VideoAssembler._find_candidate_runs(segments) == []


def test_extract_shorts_produces_vertical_clips_with_captions(tmp_path, fixture_image, fixture_audio):
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")

    segments = [
        {"text": "You tapped a card and felt nothing.", "is_short_candidate": True},
        {"text": "That silence was the whole point.", "is_short_candidate": True},
        {"text": "Here is the science behind it.", "is_short_candidate": False},
        {"text": "The twist nobody expects.", "is_short_candidate": True},
        {"text": "It changes how you'll spend this week.", "is_short_candidate": True},
    ]
    clips = [
        assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / f"clip_{i:02d}.mp4", 1.0)
        for i in range(len(segments))
    ]

    output_dir = tmp_path / "shorts"
    shorts = assembler.extract_shorts(clips, segments, output_dir)

    assert len(shorts) == 2
    for short_path in shorts:
        assert short_path.exists()
        assert probe_video_resolution(short_path) == (1080, 1920)
```

Add `probe_video_resolution` to the existing `from engine.video import ...` line at the top of the test file if not already imported (it already is, from Task 4).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_video.py -v -k "candidate_runs or extract_shorts"`
Expected: FAIL with `AttributeError: type object 'VideoAssembler' has no attribute '_find_candidate_runs'`

- [ ] **Step 3: Append to `engine/video.py`** — add `Tuple` is already imported; add these two methods inside the `VideoAssembler` class, after `finalize`:

```python
    @staticmethod
    def _find_candidate_runs(segments: List[dict]) -> List[Tuple[int, int]]:
        """Return (start_idx, end_idx) inclusive ranges of contiguous is_short_candidate segments."""
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        for i, seg in enumerate(segments):
            if seg.get("is_short_candidate"):
                if start is None:
                    start = i
            elif start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(segments) - 1))
        return runs

    def extract_shorts(
        self,
        segment_clips: List[Path],
        segments: List[dict],
        output_dir: Path,
        font_path: str = "C:/Windows/Fonts/arialbd.ttf",
    ) -> List[Path]:
        """
        Group contiguous is_short_candidate segments into standalone 1080x1920
        Shorts, with a burned-in ALL-CAPS caption from each run's first segment.
        No new TTS/image generation — these are cut from already-rendered clips.
        """
        runs = self._find_candidate_runs(segments)
        output_dir.mkdir(parents=True, exist_ok=True)
        shorts: List[Path] = []

        for i, (start, end) in enumerate(runs):
            run_clips = segment_clips[start:end + 1]
            raw_concat = self.workspace / f"short_{i:02d}_raw.mp4"
            self.concatenate_segments(run_clips, raw_concat)

            caption = (
                segments[start]["text"].upper().replace("\\", "").replace("'", r"\'").replace(":", r"\:")
            )
            vf = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,"
                f"drawtext=fontfile='{font_path}':text='{caption}':fontcolor=white:"
                "fontsize=64:borderw=4:bordercolor=black:x=(w-text_w)/2:y=120:"
                "line_spacing=10:box=0"
            )
            out = output_dir / f"short_{i:02d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(raw_concat),
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                "-c:a", "aac", "-b:a", "192k",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(out),
            ]
            _run(cmd, f"short:{out.name}")
            shorts.append(out)

        return shorts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_video.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add engine/video.py tests/engine/test_video.py
git commit -m "feat: add Shorts auto-cut (contiguous-run grouping + vertical crop + burned-in caption)"
```

---

### Task 6: `engine/youtube.py` (upload, category 27, multi-thumbnail support)

**Files:**
- Create: `engine/youtube.py`
- Create: `tests/engine/test_youtube.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `engine.youtube.DEFAULT_CATEGORY` (`"27"`), `engine.youtube.YouTubeUploader(client_secret_file: str, privacy_status: str = "private")` with `.upload(video_path, title, description, tags, category_id=DEFAULT_CATEGORY) -> dict`, `.set_default_thumbnail(video_id: str, thumbnail_path: Path) -> None`, `.update_privacy(video_id: str, privacy_status: str) -> None`. Description/disclaimer text and `/shorts` vs `/watch` URL framing are the caller's responsibility (Task 8), not this module's.

- [ ] **Step 1: Write the failing tests** — `tests/engine/test_youtube.py`

```python
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from engine.youtube import DEFAULT_CATEGORY, YouTubeUploader


def _make_uploader(tmp_path):
    with patch("engine.youtube.YouTubeUploader._authenticate", return_value=MagicMock()):
        return YouTubeUploader(client_secret_file=str(tmp_path / "secret.json"), privacy_status="private")


def test_default_category_is_education():
    assert DEFAULT_CATEGORY == "27"


def test_upload_builds_body_with_default_category_and_privacy(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_request = MagicMock()
    fake_request.next_chunk.return_value = (None, {"id": "vid123"})
    uploader.youtube.videos.return_value.insert.return_value = fake_request

    video_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake-mp4-bytes")

    response = uploader.upload(
        video_path=video_path,
        title="Why Your Brain Feels No Pain When You Tap a Card",
        description="A description with the disclaimer already appended.",
        tags=["money psychology", "splurj"],
    )

    assert response["id"] == "vid123"
    _, call_kwargs = uploader.youtube.videos.return_value.insert.call_args
    assert call_kwargs["body"]["snippet"]["categoryId"] == "27"
    assert call_kwargs["body"]["status"]["privacyStatus"] == "private"


def test_upload_retries_on_5xx_then_succeeds(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_request = MagicMock()
    fake_request.next_chunk.side_effect = [
        HttpError(resp=MagicMock(status=500), content=b"server error"),
        (None, {"id": "vid456"}),
    ]
    uploader.youtube.videos.return_value.insert.return_value = fake_request

    video_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"fake-mp4-bytes")

    with patch("engine.youtube.time.sleep"):
        response = uploader.upload(
            video_path=video_path, title="t", description="d", tags=[],
        )

    assert response["id"] == "vid456"


def test_set_default_thumbnail_calls_thumbnails_set(tmp_path):
    uploader = _make_uploader(tmp_path)
    fake_set = MagicMock()
    uploader.youtube.thumbnails.return_value.set.return_value = fake_set

    thumb_path = tmp_path / "thumb_00.png"
    thumb_path.write_bytes(b"fake-png-bytes")

    uploader.set_default_thumbnail("vid123", thumb_path)

    _, call_kwargs = uploader.youtube.thumbnails.return_value.set.call_args
    assert call_kwargs["videoId"] == "vid123"
    fake_set.execute.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_youtube.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.youtube'`

- [ ] **Step 3: Write `engine/youtube.py`**

```python
"""
YouTube Data API v3 upload module for Splurj.

Auth flow: OAuth2 via a client_secret JSON file from Google Cloud Console.
Token is cached at ~/.splurj/yt_token.pickle so the browser prompt only
fires once (or when the refresh token expires).

This module is format-agnostic: whether a video is the long-form upload or
an auto-cut Short, and what URL framing/hashtags to log, is the caller's
decision (see splurj_engine.py) — not this module's.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube"]
TOKEN_CACHE = Path.home() / ".splurj" / "yt_token.pickle"
DEFAULT_CATEGORY = "27"  # Education


class YouTubeUploader:
    def __init__(self, client_secret_file: str, privacy_status: str = "private"):
        self.client_secret_file = client_secret_file
        self.privacy_status = privacy_status
        self.youtube = self._authenticate()

    def _authenticate(self):
        creds: Optional[Credentials] = None
        TOKEN_CACHE.parent.mkdir(parents=True, exist_ok=True)

        if TOKEN_CACHE.exists():
            with open(TOKEN_CACHE, "rb") as fh:
                creds = pickle.load(fh)

        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                logger.info("YouTube token refreshed.")
            except Exception as exc:
                logger.warning("Token refresh failed (%s); re-authenticating.", exc)
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(self.client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)
            logger.info("YouTube OAuth complete.")

        with open(TOKEN_CACHE, "wb") as fh:
            pickle.dump(creds, fh)

        return build("youtube", "v3", credentials=creds)

    def upload(
        self,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = DEFAULT_CATEGORY,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags[:500],
                "categoryId": category_id,
                "defaultLanguage": "en",
            },
            "status": {
                "privacyStatus": self.privacy_status,
                "selfDeclaredMadeForKids": False,
                "madeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024,
        )

        logger.info("Starting upload: '%s' [%s]", title, self.privacy_status)
        request = self.youtube.videos().insert(
            part=",".join(body.keys()), body=body, media_body=media,
        )

        response = self._resumable_upload(request)
        logger.info("Upload complete -> video id %s", response.get("id", "unknown"))
        return response

    def set_default_thumbnail(self, video_id: str, thumbnail_path: Path) -> None:
        media = MediaFileUpload(str(thumbnail_path), mimetype="image/png")
        self.youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        logger.info("Thumbnail set for video %s -> %s", video_id, thumbnail_path.name)

    def update_privacy(self, video_id: str, privacy_status: str) -> None:
        self.youtube.videos().update(
            part="status",
            body={"id": video_id, "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False}},
        ).execute()
        logger.info("Video %s -> %s", video_id, privacy_status)

    def _resumable_upload(self, request, max_retries: int = 6) -> Dict[str, Any]:
        response = None
        retry = 0

        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    logger.info("Upload progress: %d%%", int(status.progress() * 100))
            except HttpError as exc:
                if exc.resp.status in {500, 502, 503, 504} and retry < max_retries:
                    retry += 1
                    wait = 2 ** retry
                    logger.warning("HTTP %s — retry %d/%d in %ds", exc.resp.status, retry, max_retries, wait)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Upload failed after {retry} retries: {exc}") from exc
            except Exception as exc:
                if retry < max_retries:
                    retry += 1
                    wait = 2 ** retry
                    logger.warning("Upload error — retry %d/%d: %s", retry, max_retries, exc)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Upload aborted: {exc}") from exc

        return response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_youtube.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add engine/youtube.py tests/engine/test_youtube.py
git commit -m "feat: add YouTube upload module (category 27, multi-thumbnail support)"
```

---

### Task 7: `engine/gemini_tools.py` (script polish + prompt enhancement, Splurj rules)

Optional render-time passes over an already-loaded blueprint (invoked via `splurj_engine.py --polish-script`/`--enhance-prompts` in Task 8) — not the citation-gated topic-to-blueprint drafting step, which is a separate later plan.

**Files:**
- Create: `engine/gemini_tools.py`
- Create: `tests/engine/test_gemini_tools.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `engine.gemini_tools.GeminiScriptPolisher(api_key: str)` with `.polish_segment(text: str, directive: str) -> str` and `.polish_blueprint(blueprint: dict) -> dict`; `engine.gemini_tools.GeminiPromptEnhancer(api_key: str)` with `.enhance_prompt(prompt: str, segment_text: str) -> str` and `.enhance_blueprint(blueprint: dict) -> dict`.

- [ ] **Step 1: Write the failing tests** — `tests/engine/test_gemini_tools.py`

```python
from unittest.mock import MagicMock, patch

from engine.gemini_tools import GeminiPromptEnhancer, GeminiScriptPolisher


def _fake_text_response(text: str) -> MagicMock:
    return MagicMock(text=text)


def test_polish_segment_returns_polished_text_on_success():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response(
            "You tapped your card this morning. You felt nothing at all."
        )
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_segment("You tapped your card. You felt nothing.", directive="calm")

    assert result == "You tapped your card this morning. You felt nothing at all."


def test_polish_segment_falls_back_to_original_on_api_error():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("boom")
        polisher = GeminiScriptPolisher(api_key="key")
        original = "You tapped your card. You felt nothing."
        result = polisher.polish_segment(original, directive="calm")

    assert result == original


def test_polish_segment_falls_back_when_word_count_drifts_too_much():
    original = "You tapped your card. You felt nothing. " * 5  # ~20 words
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response("Short.")
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_segment(original, directive="calm")

    assert result == original


def test_polish_blueprint_rebuilds_full_text_from_polished_segments():
    blueprint = {
        "voiceover": {"directive": "calm", "full_text": "old text"},
        "timeline": [
            {"start": 0, "end": 15, "text": "First segment.", "prompt": "p1", "is_short_candidate": False},
            {"start": 15, "end": 30, "text": "Second segment.", "prompt": "p2", "is_short_candidate": False},
        ],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = [
            _fake_text_response("First segment polished."),
            _fake_text_response("Second segment polished."),
        ]
        polisher = GeminiScriptPolisher(api_key="key")
        result = polisher.polish_blueprint(blueprint)

    assert result["timeline"][0]["text"] == "First segment polished."
    assert result["timeline"][1]["text"] == "Second segment polished."
    assert result["voiceover"]["full_text"] == "First segment polished. Second segment polished."
    assert result["timeline"][0]["prompt"] == "p1"  # untouched by the script polisher


def test_enhance_prompt_returns_enhanced_text_on_success():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response(
            "Hand-drawn 2D doodle... enhanced prompt ... 16:9 aspect ratio."
        )
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "You feel nothing when you tap a card.")

    assert "enhanced prompt" in result


def test_enhance_prompt_falls_back_to_original_on_api_error():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.side_effect = RuntimeError("boom")
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_prompt("a doodle wallet", "segment text")

    assert result == "a doodle wallet"


def test_enhance_blueprint_rebuilds_timeline_prompts():
    blueprint = {
        "timeline": [
            {"start": 0, "end": 15, "text": "First.", "prompt": "p1", "is_short_candidate": False},
        ],
    }
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value.models.generate_content.return_value = _fake_text_response("p1-enhanced")
        enhancer = GeminiPromptEnhancer(api_key="key")
        result = enhancer.enhance_blueprint(blueprint)

    assert result["timeline"][0]["prompt"] == "p1-enhanced"
    assert result["timeline"][0]["text"] == "First."  # untouched by the prompt enhancer
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/engine/test_gemini_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'engine.gemini_tools'`

- [ ] **Step 3: Write `engine/gemini_tools.py`**

```python
"""
Gemini-powered render-time pre-processing tools for the Splurj pipeline.

Two optional passes over an already-drafted blueprint, run before asset
generation:
  - GeminiScriptPolisher : refines narration text against Splurj voice rules
  - GeminiPromptEnhancer : verifies and strengthens doodle-style image prompts
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_MODEL = "gemini-3.5-flash"

_VOICE_RULES = (
    "Splurj narration voice rules:\n"
    "- Calm, curious, a little conspiratorial. 2nd person only ('you', 'your brain') — never 'we' or 'I'.\n"
    "- Rhythm: short sentence, short sentence, one longer sentence that builds depth, short sentence, "
    "a question every 4-6 sentences.\n"
    "- No jargon without an immediate plain-English decode.\n"
    "- Zero financial advice. No outcome promises. Explain behavior; never prescribe action.\n"
    "- Ends by reflecting the psychological truth onto a money decision the viewer will make this week.\n"
)

_VISUAL_RULES = (
    "Splurj visual style rules:\n"
    "- Hand-drawn 2D doodle cartoon animation. Flat colors, bold black outlines, sketchy marker lines.\n"
    "- Simple stick figures: large circular heads, dot eyes, thick expressive brow lines.\n"
    "- Background color signals tone: orange = urgency/sale, blue = science/experiment, tan = history, "
    "white/yellow = happy/discovery, green ground + blue sky = everyday outdoor life.\n"
    "- Required suffix: no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, "
    "educational YouTube explainer doodle style.\n"
)


class GeminiScriptPolisher:
    """Refines segment narration text against Splurj voice rules. Falls back
    to the original text on any failure or excessive word-count drift."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = _MODEL

    def polish_segment(self, text: str, directive: str) -> str:
        user_prompt = (
            f"{_VOICE_RULES}\n"
            f"Voice directive for this episode: {directive}\n\n"
            f"Polish the segment below. Keep approximately the same word count (~30-40 words).\n"
            f"Return ONLY the revised text — no quotes, no explanation.\n\n"
            f"SEGMENT:\n{text}"
        )

        try:
            response = self._client.models.generate_content(model=self._model, contents=user_prompt)
            result = response.text.strip()
        except Exception as exc:
            logger.warning("Script polish failed (segment) — using original: %s", exc)
            return text

        orig_wc = len(text.split())
        result_wc = len(result.split())
        if not (orig_wc * 0.5 <= result_wc <= orig_wc * 1.5):
            logger.warning("Polish word count drift (%d -> %d) — using original", orig_wc, result_wc)
            return text

        return result

    def polish_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        directive = blueprint["voiceover"].get("directive", "")
        new_timeline = []

        for i, seg in enumerate(blueprint["timeline"]):
            logger.info("  Polishing segment %d/%d…", i + 1, len(blueprint["timeline"]))
            polished = self.polish_segment(seg["text"], directive)
            new_timeline.append({**seg, "text": polished})

        full_text = " ".join(s["text"] for s in new_timeline)
        return {
            **blueprint,
            "timeline": new_timeline,
            "voiceover": {**blueprint["voiceover"], "full_text": full_text},
        }


class GeminiPromptEnhancer:
    """Verifies and strengthens doodle-style image prompts before image
    generation. Falls back to the original prompt on any failure."""

    def __init__(self, api_key: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = _MODEL

    def enhance_prompt(self, prompt: str, segment_text: str) -> str:
        user_prompt = (
            f"{_VISUAL_RULES}\n"
            f"The narration for this frame:\n{segment_text}\n\n"
            f"Enhance the image prompt below to:\n"
            f"  1. Better serve the narration.\n"
            f"  2. Strictly follow the Splurj visual style rules.\n"
            f"  3. Stay under 120 words.\n"
            f"Return ONLY the enhanced prompt — no explanation.\n\n"
            f"PROMPT:\n{prompt}"
        )

        try:
            response = self._client.models.generate_content(model=self._model, contents=user_prompt)
            return response.text.strip()
        except Exception as exc:
            logger.warning("Prompt enhancement failed — using original: %s", exc)
            return prompt

    def enhance_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        new_timeline = []
        for i, seg in enumerate(blueprint["timeline"]):
            logger.info("  Enhancing prompt %d/%d…", i + 1, len(blueprint["timeline"]))
            enhanced = self.enhance_prompt(seg["prompt"], seg["text"])
            new_timeline.append({**seg, "prompt": enhanced})

        return {**blueprint, "timeline": new_timeline}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/engine/test_gemini_tools.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add engine/gemini_tools.py tests/engine/test_gemini_tools.py
git commit -m "feat: add Gemini script-polish and prompt-enhancement passes with Splurj voice/visual rules"
```

---

### Task 8: `splurj_engine.py` — main orchestrator + CLI

Wires Tasks 1-7 into one pipeline: load blueprint → generate audio+images per segment (parallel, conditioned on the character reference if present) → per-segment clips → concat → ambient mix → finalize (16:9) → extract Shorts → generate thumbnail variants → upload long-form + Shorts to YouTube, with the disclaimer hardcoded into every description.

**Files:**
- Create: `splurj_engine.py`
- Create: `tests/test_splurj_engine.py`

**Interfaces:**
- Consumes: `AudioGenerator` (Task 1), `ImageGenerator` (Task 2), `VideoAssembler` (Tasks 4-5), `YouTubeUploader` (Task 6), `GeminiScriptPolisher`/`GeminiPromptEnhancer` (Task 7).
- Produces: `splurj_engine.load_blueprint(path: str) -> List[dict]`, `splurj_engine.load_env() -> Dict[str, str]`, `splurj_engine.build_description(meta_description: str, full_script: str, day: int) -> str`, `splurj_engine.build_thumbnail_prompts(title: str) -> List[str]`, `splurj_engine.slugify(text: str) -> str`, `splurj_engine.run_pipeline(blueprint: dict, env: dict, skip_upload: bool = False, polish_script: bool = False, enhance_prompts: bool = False) -> Dict[str, Any]` returning `{"long_form": Path, "shorts": List[Path]}`.

- [ ] **Step 1: Write the failing tests** — `tests/test_splurj_engine.py`

```python
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from splurj_engine import (
    DISCLAIMER,
    build_description,
    build_thumbnail_prompts,
    load_blueprint,
    load_env,
    run_pipeline,
    slugify,
    _validate_blueprint,
)


def _valid_blueprint(day=1):
    return {
        "day": day,
        "format": "long",
        "metadata": {
            "title": "Why Your Brain Feels No Pain When You Tap a Card",
            "description": "A hook description.",
            "tags": ["money psychology", "splurj"],
        },
        "voiceover": {"directive": "calm, curious", "full_text": "Seg one. Seg two. Seg three. Seg four."},
        "timeline": [
            {"start": 0, "end": 15, "text": "Seg one.", "prompt": "doodle prompt one", "is_short_candidate": True},
            {"start": 15, "end": 30, "text": "Seg two.", "prompt": "doodle prompt two", "is_short_candidate": True},
            {"start": 30, "end": 45, "text": "Seg three.", "prompt": "doodle prompt three", "is_short_candidate": False},
            {"start": 45, "end": 60, "text": "Seg four.", "prompt": "doodle prompt four", "is_short_candidate": False},
        ],
    }


def test_validate_blueprint_accepts_well_formed_blueprint():
    _validate_blueprint(_valid_blueprint())  # must not raise


def test_validate_blueprint_rejects_missing_top_level_key():
    bp = _valid_blueprint()
    del bp["voiceover"]
    with pytest.raises(SystemExit, match="missing required keys"):
        _validate_blueprint(bp)


def test_validate_blueprint_rejects_missing_segment_key():
    bp = _valid_blueprint()
    del bp["timeline"][0]["is_short_candidate"]
    with pytest.raises(SystemExit, match="is_short_candidate"):
        _validate_blueprint(bp)


def test_validate_blueprint_rejects_full_text_mismatch():
    bp = _valid_blueprint()
    bp["voiceover"]["full_text"] = "This does not match the segments at all."
    with pytest.raises(SystemExit, match="full_text"):
        _validate_blueprint(bp)


def test_load_blueprint_returns_list_for_single_object(tmp_path):
    bp_path = tmp_path / "bp.json"
    bp_path.write_text(json.dumps(_valid_blueprint()))
    result = load_blueprint(str(bp_path))
    assert isinstance(result, list)
    assert result[0]["day"] == 1


def test_load_env_raises_when_keys_missing(monkeypatch):
    for key in ("ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID", "GEMINI_API_KEY", "YOUTUBE_CLIENT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(SystemExit, match="Missing environment variables"):
        load_env()


def test_load_env_returns_dict_when_keys_present(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "a")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "b")
    monkeypatch.setenv("GEMINI_API_KEY", "c")
    monkeypatch.setenv("YOUTUBE_CLIENT_SECRET", "d")
    env = load_env()
    assert env["ELEVENLABS_API_KEY"] == "a"


def test_build_description_always_appends_disclaimer():
    result = build_description("A hook description with no disclaimer.", "Full script text.", day=1)
    assert DISCLAIMER in result
    assert "A hook description" in result


def test_build_thumbnail_prompts_returns_two_style_locked_variants():
    prompts = build_thumbnail_prompts("Why Your Brain Feels No Pain")
    assert len(prompts) == 2
    for p in prompts:
        assert "WHY YOUR BRAIN FEELS NO PAIN" in p
        assert "16:9 aspect ratio" in p
    assert prompts[0] != prompts[1]


def test_slugify_produces_filesystem_safe_string():
    assert slugify("Why Your Brain Feels No Pain!") == "why_your_brain_feels_no_pain"


def test_run_pipeline_end_to_end_with_mocked_apis(tmp_path, fixture_image, fixture_audio, monkeypatch):
    monkeypatch.setattr("splurj_engine.WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr("splurj_engine.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("splurj_engine.ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr("splurj_engine.CHANNEL_DATA", tmp_path / "channel_data")
    (tmp_path / "assets" / "ambient").mkdir(parents=True)
    (tmp_path / "channel_data").mkdir(parents=True)

    fake_audio_gen = MagicMock()
    fake_audio_gen.generate_segment.side_effect = (
        lambda text, output_path, directive="", max_retries=4: shutil.copy2(fixture_audio, output_path) or output_path
    )
    fake_audio_gen.probe_duration.side_effect = lambda audio_path: 1.0

    fake_image_gen = MagicMock()
    fake_image_gen.generate.side_effect = (
        lambda prompt, output_path, reference_image_path=None, max_retries=4: shutil.copy2(fixture_image, output_path) or output_path
    )

    fake_uploader = MagicMock()
    fake_uploader.upload.return_value = {"id": "vid-fake"}

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch("splurj_engine.YouTubeUploader", return_value=fake_uploader):

        env = {
            "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
            "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
        }
        result = run_pipeline(_valid_blueprint(), env, skip_upload=False)

    from engine.video import probe_video_resolution

    assert result["long_form"].exists()
    assert probe_video_resolution(result["long_form"]) == (1920, 1080)
    assert len(result["shorts"]) == 1  # segments 0-1 form one contiguous run

    # Long-form upload + one Short upload = 2 calls; each description carries the disclaimer.
    assert fake_uploader.upload.call_count == 2
    for _, call_kwargs in fake_uploader.upload.call_args_list:
        assert DISCLAIMER in call_kwargs["description"]

    # One thumbnail generated per variant (2), first one set as the video's default thumbnail.
    fake_uploader.set_default_thumbnail.assert_called_once()


def test_run_pipeline_skip_upload_does_not_call_youtube(tmp_path, fixture_image, fixture_audio, monkeypatch):
    monkeypatch.setattr("splurj_engine.WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr("splurj_engine.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("splurj_engine.ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr("splurj_engine.CHANNEL_DATA", tmp_path / "channel_data")
    (tmp_path / "assets" / "ambient").mkdir(parents=True)
    (tmp_path / "channel_data").mkdir(parents=True)

    fake_audio_gen = MagicMock()
    fake_audio_gen.generate_segment.side_effect = (
        lambda text, output_path, directive="", max_retries=4: shutil.copy2(fixture_audio, output_path) or output_path
    )
    fake_audio_gen.probe_duration.side_effect = lambda audio_path: 1.0

    fake_image_gen = MagicMock()
    fake_image_gen.generate.side_effect = (
        lambda prompt, output_path, reference_image_path=None, max_retries=4: shutil.copy2(fixture_image, output_path) or output_path
    )

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch("splurj_engine.YouTubeUploader") as mock_uploader_cls:

        env = {
            "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
            "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
        }
        result = run_pipeline(_valid_blueprint(), env, skip_upload=True)

    assert result["long_form"].exists()
    mock_uploader_cls.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_splurj_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'splurj_engine'`

- [ ] **Step 3: Write `splurj_engine.py`**

```python
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
import sys
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
    return (
        f"{meta_description}\n\n"
        f"{'-' * 40}\n"
        f"Episode {day}\n\n"
        f'"{full_script}"\n\n'
        f"{'-' * 40}\n"
        f"{DISCLAIMER}"
    )


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
                title=title[:95] + " #Shorts",
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_splurj_engine.py -v`
Expected: 12 passed (the two `run_pipeline` tests are slower — real ffmpeg encodes for 4 segments + concat + ambient + finalize + shorts)

- [ ] **Step 5: Commit**

```bash
git add splurj_engine.py tests/test_splurj_engine.py
git commit -m "feat: add splurj_engine orchestrator wiring audio/image/video/youtube into one pipeline"
```

---

### Task 9: End-to-end smoke test — `content_example.json` + CLI entrypoint

A checked-in sample blueprint (mirroring Grafyte's `content_example.json`) for manual first-run testing, plus a test proving the real CLI entrypoint (`main()`, not just `run_pipeline()` called directly) works: loads a real file from disk, respects `--keep-workspace`, and cleans up the workspace by default. Deliberately small (6 segments, ~90s) for fast iteration — the full 45-56 segment / 10-14 min blueprints are produced by `splurj_draft.py` in a later plan, not authored by hand.

**Files:**
- Create: `content_example.json`
- Create: `tests/test_e2e_smoke.py`

**Interfaces:**
- Consumes: `splurj_engine.main`, `splurj_engine.load_blueprint`, `splurj_engine.run_pipeline` (Task 8).

- [ ] **Step 1: Write `content_example.json`**

```json
{
  "day": 1,
  "format": "long",
  "metadata": {
    "title": "Why Your Brain Feels No Pain When You Tap a Card",
    "description": "You tap a card every day without a second thought. Here's the real reason that silence at checkout isn't an accident.",
    "tags": ["money psychology", "behavioral finance", "splurj", "pain of paying"]
  },
  "voiceover": {
    "directive": "Calm, curious, a little conspiratorial. Unhurried.",
    "full_text": "This morning, you tapped a small piece of plastic against a machine. Money left your account. You felt nothing at all. Economists call that missing flicker the pain of paying. Your brain evolved to feel a small sting every time you spend. A tap removes that sting on purpose, not by accident. That silence at checkout is proof the system worked exactly as intended."
  },
  "timeline": [
    {
      "start": 0, "end": 15,
      "text": "This morning, you tapped a small piece of plastic against a machine.",
      "prompt": "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, slightly imperfect sketchy marker lines, a stick figure with a big circular head and dot eyes tapping a credit card against a card reader on a plain white counter, neutral expression, background is flat white, no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, educational YouTube explainer doodle style.",
      "is_short_candidate": true
    },
    {
      "start": 15, "end": 30,
      "text": "Money left your account. You felt nothing at all.",
      "prompt": "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, slightly imperfect sketchy marker lines, the same stick figure standing calmly with a thought bubble showing a small dollar bill with a face flying away, flat white background, no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, educational YouTube explainer doodle style.",
      "is_short_candidate": true
    },
    {
      "start": 30, "end": 45,
      "text": "Economists call that missing flicker the pain of paying.",
      "prompt": "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, slightly imperfect sketchy marker lines, a large brain with dollar-sign eyes centered on a solid blue background with bold ALL CAPS yellow text reading PAIN OF PAYING above it, no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, educational YouTube explainer doodle style.",
      "is_short_candidate": false
    },
    {
      "start": 45, "end": 60,
      "text": "Your brain evolved to feel a small sting every time you spend.",
      "prompt": "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, slightly imperfect sketchy marker lines, a caveman-styled simple stick figure handing over coins with a wincing expression, tan background suggesting history, no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, educational YouTube explainer doodle style.",
      "is_short_candidate": false
    },
    {
      "start": 60, "end": 75,
      "text": "A tap removes that sting on purpose, not by accident.",
      "prompt": "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, slightly imperfect sketchy marker lines, a credit card drawn with a sly grinning face next to a card reader, red ALL CAPS text reading BY DESIGN above it, flat orange background, no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, educational YouTube explainer doodle style.",
      "is_short_candidate": true
    },
    {
      "start": 75, "end": 90,
      "text": "That silence at checkout is proof the system worked exactly as intended.",
      "prompt": "Hand-drawn 2D doodle cartoon animation, flat colors, bold black outlines, slightly imperfect sketchy marker lines, the same original stick figure walking away calmly from the counter, flat white background, no gradients, no shadows, no textures, no photorealism, no 3D, 16:9 aspect ratio, educational YouTube explainer doodle style.",
      "is_short_candidate": true
    }
  ]
}
```

- [ ] **Step 2: Write the failing test** — `tests/test_e2e_smoke.py`

```python
import shutil
import sys
from unittest.mock import MagicMock, patch

from engine.video import probe_video_resolution
from splurj_engine import load_blueprint, main


def _patch_dirs(monkeypatch, tmp_path):
    monkeypatch.setattr("splurj_engine.WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr("splurj_engine.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("splurj_engine.ASSETS_DIR", tmp_path / "assets")
    monkeypatch.setattr("splurj_engine.CHANNEL_DATA", tmp_path / "channel_data")
    (tmp_path / "assets" / "ambient").mkdir(parents=True)
    (tmp_path / "channel_data").mkdir(parents=True)


def _patch_generators(fixture_image, fixture_audio):
    fake_audio_gen = MagicMock()
    fake_audio_gen.generate_segment.side_effect = (
        lambda text, output_path, directive="", max_retries=4: shutil.copy2(fixture_audio, output_path) or output_path
    )
    fake_audio_gen.probe_duration.side_effect = lambda audio_path: 1.0

    fake_image_gen = MagicMock()
    fake_image_gen.generate.side_effect = (
        lambda prompt, output_path, reference_image_path=None, max_retries=4: shutil.copy2(fixture_image, output_path) or output_path
    )
    return fake_audio_gen, fake_image_gen


def test_content_example_loads_and_validates():
    blueprints = load_blueprint("content_example.json")
    assert len(blueprints) == 1
    assert len(blueprints[0]["timeline"]) == 6


def test_cli_main_no_upload_renders_and_cleans_workspace(tmp_path, fixture_image, fixture_audio, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    fake_audio_gen, fake_image_gen = _patch_generators(fixture_image, fixture_audio)

    for key, val in {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }.items():
        monkeypatch.setenv(key, val)

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch.object(sys, "argv", ["splurj_engine.py", "--input", "content_example.json", "--no-upload"]):
        main()

    output_files = list((tmp_path / "output").glob("day_001_*.mp4"))
    assert len(output_files) == 1
    assert probe_video_resolution(output_files[0]) == (1920, 1080)
    assert not (tmp_path / "workspace" / "day_001").exists()  # cleaned up by default


def test_cli_main_keep_workspace_preserves_temp_files(tmp_path, fixture_image, fixture_audio, monkeypatch):
    _patch_dirs(monkeypatch, tmp_path)
    fake_audio_gen, fake_image_gen = _patch_generators(fixture_image, fixture_audio)

    for key, val in {
        "ELEVENLABS_API_KEY": "a", "ELEVENLABS_VOICE_ID": "b",
        "GEMINI_API_KEY": "c", "YOUTUBE_CLIENT_SECRET": "d",
    }.items():
        monkeypatch.setenv(key, val)

    with patch("splurj_engine.AudioGenerator", return_value=fake_audio_gen), \
         patch("splurj_engine.ImageGenerator", return_value=fake_image_gen), \
         patch.object(sys, "argv", ["splurj_engine.py", "--input", "content_example.json", "--no-upload", "--keep-workspace"]):
        main()

    assert (tmp_path / "workspace" / "day_001").exists()
```

- [ ] **Step 3: Run the test file before writing the fixture, to confirm a genuine RED state**

Temporarily rename `content_example.json` out of the way (or run this on a clean checkout before Step 1) and run:

Run: `pytest tests/test_e2e_smoke.py -v`
Expected: FAIL — `test_content_example_loads_and_validates` fails with `SystemExit: Blueprint not found: content_example.json`

- [ ] **Step 4: Run tests to verify they pass**

With `content_example.json` in place (Step 1), run: `pytest tests/test_e2e_smoke.py -v`
Expected: 3 passed (~15-25s — six real ffmpeg segment encodes + concat + ambient + finalize + 3 shorts, twice). No new production code was needed for this task — Task 8's `splurj_engine.py` already implements everything this test exercises; this task adds the real-file/real-CLI-entrypoint coverage and a reusable example fixture.

- [ ] **Step 5: Run the full test suite once, to confirm nothing earlier broke**

Run: `pytest -v`
Expected: all tests across every task pass together

- [ ] **Step 6: Commit**

```bash
git add content_example.json tests/test_e2e_smoke.py
git commit -m "test: add end-to-end smoke test with a checked-in sample blueprint and real CLI entrypoint coverage"
```
