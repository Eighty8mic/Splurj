# Splurj

**Splurj** (`@Splurj-it`) is a fully autonomous YouTube pipeline that writes, narrates, illustrates, animates, and publishes short explainer videos about money psychology and behavioral finance — one new video a night, unattended.

Every video traces its claims to a real published study (Kahneman, Thaler, Ariely, and others) from an approved citation bank, narrated over hand-drawn "doodle" stick-figure animation, with auto-cut YouTube Shorts pulled from the same render.

## How a video gets made

```
citation bank (channel_data/citations.json)
        │
        ▼
  topic + script draft  ──  Gemini writes ~2000 words of narration,
  (splurj_draft.py)         picks an unused (citation, angle) pair,
        │                   drafts per-scene pose prompts
        ▼
  queue_local/next.json  (the "blueprint" — see below)
        │
        ▼
  render + upload  ──  ElevenLabs TTS (chained for prosody continuity)
  (splurj_engine.py)    + Gemini images (3 pose variants/scene, cross-
        │                faded into motion) + ffmpeg assembly + Shorts
        ▼
  YouTube (private, long-form + 2 Shorts) + local output/*.mp4
```

`run_queued.py` is the nightly entrypoint: if a blueprint is already queued it renders and uploads it; if the queue is empty it drafts a fresh one from the citation bank first. Either way, the queued file is archived to `queue_local/processed/` on success so nothing republishes twice.

## Quick start

**Prerequisites**
- Python 3.11+ (this project develops against 3.14; use the `py` launcher on Windows if plain `python` isn't on PATH)
- [ffmpeg](https://ffmpeg.org/) and `ffprobe` on PATH (8.x confirmed working)
- API keys: [ElevenLabs](https://elevenlabs.io/) (TTS + music), [Gemini](https://ai.google.dev/) (script, images), and a Google Cloud OAuth client with the YouTube Data API v3 enabled

**Setup**
```bash
pip install -r requirements.txt
cp .env.example .env        # fill in your keys, see Configuration below
```

Generate the character reference sheet once — every scene image is conditioned on it for a consistent-looking recurring character:
```bash
py generate_character_reference.py
```

**Render one video manually**
```bash
py splurj_engine.py --input content_example.json --no-upload --keep-workspace
```
Drop `--no-upload` once you're ready to publish for real (uploads land as **private** by default — see `YOUTUBE_PRIVACY`). The first real upload opens a browser for one-time Google OAuth consent; after that the token is cached at `~/.splurj/yt_token.pickle`.

**Draft + queue tonight's video from the citation bank**
```bash
py splurj_draft.py              # auto-incrementing day number
py splurj_draft.py --day 12     # explicit day number
py run_queued.py                # render + upload whatever's queued (auto-drafts first if empty)
```

## Configuration

All variables live in `.env` (see `.env.example` for the template).

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ELEVENLABS_API_KEY` | yes | — | TTS + music generation |
| `ELEVENLABS_VOICE_ID` | yes | — | Voice to narrate with — browse [elevenlabs.io/voice-library](https://elevenlabs.io/voice-library) |
| `ELEVENLABS_MODEL` | no | `eleven_turbo_v2` | TTS model |
| `GEMINI_API_KEY` | yes | — | Script drafting, image generation |
| `GEMINI_IMAGE_MODEL` | no | `gemini-3.1-flash-image` | Image generation model |
| `YOUTUBE_CLIENT_SECRET` | yes | — | Path to your Google Cloud OAuth client JSON |
| `YOUTUBE_PRIVACY` | no | `private` | Upload visibility: `private`, `unlisted`, or `public` |
| `YOUTUBE_CATEGORY_ID` | no | `27` (Education) | YouTube category |
| `AMBIENT_DB` | no | `-15` | Background music level relative to voice, in dB. Quiet by design — raise it (e.g. `-9`) if the music is too subtle to notice |
| `SHORT_MIN_SEGMENTS` / `SHORT_MAX_SEGMENTS` | no | `3` / `4` | Segment-count bounds for auto-cut Shorts |
| `NOTIFY_SMTP_HOST` / `_PORT` / `_USER` / `_PASSWORD` / `NOTIFY_EMAIL_TO` | no | — | Optional email alert when the nightly run finds a dead YouTube token. A desktop toast fires either way, no config needed for that |

## Nightly automation

The intended steady state is a Windows Scheduled Task running `run_queued.py` once a day:

```powershell
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
  -Argument '/c "C:\path\to\python.exe" "C:\path\to\Splurj\run_queued.py" >> "C:\path\to\Splurj\queue_local\nightly.log" 2>&1' `
  -WorkingDirectory "C:\path\to\Splurj"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -TaskName "SplurjNightlyRun" -Action $action -Trigger $trigger
```

Use the **real** `python.exe` path (`py -c "import sys; print(sys.executable)"`), not the Windows Store `py.exe` alias — the alias does not reliably launch under Task Scheduler's non-interactive context.

**A known gotcha:** while your Google Cloud OAuth consent screen is in "Testing" publishing status, refresh tokens expire after ~7 days — expect to re-run the interactive login roughly weekly. `run_queued.py` checks token validity before doing any work each night; if it's dead, it fires a desktop notification (and email, if configured) and skips that night's render entirely rather than hanging on a browser prompt nobody's there to click. Moving the consent screen to Production removes this, but requires a Google security review for the YouTube scope — for a single-operator pipeline like this, re-authenticating weekly is usually the more practical trade.

## The blueprint format

Every render is driven by a JSON "blueprint" (see `content_example.json` for a full worked example):

```jsonc
{
  "day": 1,
  "format": "long",
  "metadata": { "title": "...", "description": "...", "tags": ["..."] },
  "voiceover": { "directive": "Calm, curious...", "full_text": "..." },
  "timeline": [
    {
      "start": 0, "end": 15,
      "text": "One segment of narration, ~45 words or fewer.",
      "poses": ["pose 1 image prompt", "pose 2 image prompt", "pose 3 image prompt"],
      "is_short_candidate": true,
      "sfx": [{ "cue": "a soft chime", "at": 0.4 }]   // optional
    }
  ]
}
```

- `timeline` segments are grouped into "scenes" (default 3 segments/scene); every segment in a scene shares the *same* `poses` list, so the image cache collapses repeats into one API call per unique scene rather than per segment.
- Each scene's 3 pose variants (same character/setting, varied gesture/expression) are cross-faded into a short animation loop at render time, instead of a single static image — see `engine/video.py`'s `build_pose_sequence`.
- `sfx` cues are optional per-segment sound effects, generated and mixed in at the given offset.

## Testing

```bash
py -m pytest tests/ -q
```
The suite includes real-ffmpeg and real-audio-processing tests (not everything is mocked), so expect it to take a minute or two rather than a few seconds. It intentionally never hits real ElevenLabs/Gemini/YouTube APIs — those are mocked throughout.

## Project layout

```
splurj_engine.py       Render + upload orchestrator (CLI entrypoint)
splurj_draft.py         Auto-draft a blueprint from the citation bank (CLI entrypoint)
run_queued.py            Nightly entrypoint: draft-if-empty, render, upload, archive
generate_character_reference.py   One-time reference-sheet generator

engine/
  drafting.py            Gemini script + scene-pose drafting, blueprint assembly
  gemini_tools.py         Optional pre-render script/prompt polish passes
  audio.py                 ElevenLabs TTS, loudness normalization, request stitching
  images.py                 Gemini image generation + prompt-hash caching
  music.py                   ElevenLabs background music generation
  sfx.py                       Sound-effect generation + audio overlay
  video.py                       ffmpeg assembly: pose crossfades, concat, mix, Shorts
  youtube.py                       OAuth + upload
  topic_picker.py                    Citation/angle selection for auto-drafting
  notify.py                           Desktop + email alerts on nightly failure

channel_data/           Citation bank + runtime state (character reference, day counter)
queue_local/              Queued blueprint + processed/ archive
content_example.json        Hand-authored reference blueprint (also an e2e test fixture)
master_prompt_splurj.txt      Standalone manual-workflow prompt for co-writing a video
                                 by hand in a chat UI — not used by the automated pipeline
docs/superpowers/               Design spec + implementation plan for the render engine
```

## Content safety

Every script is grounded in one citation from `channel_data/citations.json` and is restricted to naming *only* the researcher(s) attached to that citation — `engine/drafting.py`'s `check_citation_safety` rejects (and retries) any draft that names an unlisted researcher. The channel never gives financial advice or recommends specific products; every video's description ends with a fixed disclaimer.

Approving new citations is currently manual: edit `channel_data/citations.json` and set `"status": "approved"`.
