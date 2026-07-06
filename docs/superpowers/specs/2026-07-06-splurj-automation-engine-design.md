# Splurj Automation Engine — Design Spec

Status: Approved by user, pending implementation plan
Date: 2026-07-06

## 1. Overview

A fully autonomous daily content pipeline for Splurj (@Splurj-it), ported from the proven Grafyte engine architecture (`C:\Users\School\Desktop\Grafyte`) and adapted to Splurj's format, brand rules, and YMYL (financial-content) safety requirements. The pipeline reads a JSON "blueprint" describing one video, generates all assets (narration audio, doodle-style images), assembles a 16:9 video with FFmpeg, and uploads a long-form video plus 2-3 auto-cut Shorts to YouTube — with zero manual steps after the queue is filled.

This is a **port and adapt**, not a from-scratch build: Grafyte's `engine/audio.py`, `engine/youtube.py`, and orchestration patterns carry over with minimal changes. The genuinely new work is in content generation safety (citation bank + QA gate), image consistency (character reference), the 16:9/long-duration video path, and the Shorts auto-cut.

## 2. Decisions already made (from brainstorming)

- **Length**: keep Splurj's existing 10-14 minute long-form format. Segments scale from Grafyte's 13×15s to ~45-56×15s per video, using the "hold scenes across consecutive timestamps" rule already in Splurj's master prompt so most segments reuse a cached image rather than generating a new one every 15s.
- **Autonomy**: fully autonomous, no human checkpoint between topic and publish. Because Splurj's brand rules require every claim to trace to a real, verifiable researcher/study, this is offset by an automated citation bank + QA gate (Section 6) instead of a human review step.
- **Scope**: long-form daily + auto-cut Shorts pulled from the same generation (not a separate Shorts pipeline). Shorts add no marginal API cost — they're re-encoded crops of already-generated segments.
- **Images**: Gemini image generation with a saved character/prop reference image attached to every scene prompt, for visual consistency. Model: `gemini-3.1-flash-image` (not the pro-tier model Grafyte defaults to) — Splurj's flat 2D doodle style doesn't need photoreal fidelity, and flash-image is roughly half the per-image cost.
- **Estimated cost**: ~$2.00-2.10/video (~$0.73 TTS, ~$1.27 images incl. thumbnail, ~$0.08 text-gen), ~$80-90/month at daily cadence. See Section 12.

## 3. Directory layout

```
Splurj/
├── engine/
│   ├── __init__.py
│   ├── audio.py               # ElevenLabs TTS — ported from Grafyte near-unchanged
│   ├── images.py               # Gemini image gen — extended for reference-image input
│   ├── video.py                 # FFmpeg assembly — 16:9, + Shorts auto-cut
│   ├── youtube.py               # YouTube Data API v3 upload — ported, category/disclaimer changes
│   └── gemini_tools.py         # Script polish + prompt enhancement + citation QA gate
├── queue/                       # Blueprint JSON files awaiting render
│   └── done/                    # Completed blueprints moved here after upload
├── output/                      # Final MP4s (long-form + shorts)
├── workspace/                    # Temp per-run assets (auto-cleaned)
├── assets/
│   └── ambient/                 # Optional ambient bed audio
├── cache/images/                 # Prompt-hash image cache (reused from Grafyte's images.py)
├── channel_data/
│   ├── citation_bank.md         # Pre-verified researcher/study facts — see Section 6
│   └── character_reference.png  # Generated once; the recurring stick-figure/prop sheet
├── scripts/                      # EXISTING — manual one-off chat-workflow scripts, untouched
├── master_prompt_splurj.txt      # EXISTING — source of all voice/visual rules below
├── splurj_engine.py              # Main orchestrator + CLI (was grafyte_engine.py)
├── splurj_draft.py               # CLI: topic → blueprint via Gemini (was grafyte_draft.py)
├── run_queue.py                  # Daily runner, called by Task Scheduler
├── scheduler/splurj_daily.ps1
├── .env / .env.example
└── requirements.txt
```

The existing chat-based Stage 1-4 workflow (`scripts/`, the master prompt conversation) is **not replaced**. It remains available for one-off, manually-directed videos. The new engine is an independent, second path used for the daily-automation channel operation.

## 4. Blueprint JSON schema

```json
{
  "day": 1,
  "format": "long",
  "metadata": {
    "title": "≤70 char scroll-stopping title, no clickbait the script doesn't deliver",
    "description": "hook + summary + CTA + disclaimer line + hashtags (see Section 9)",
    "tags": ["tag1", "tag2", "..."]
  },
  "voiceover": {
    "directive": "Calm, curious, a little conspiratorial. 2nd person. Unhurried.",
    "full_text": "Complete narration — concatenation of all segment `text` fields."
  },
  "timeline": [
    {
      "start": 0,
      "end": 15,
      "text": "Narration for this segment, ~30-40 words.",
      "prompt": "Full doodle-style image prompt for this segment's scene, including the style anchor and style lock strings from the master prompt.",
      "is_short_candidate": false
    }
  ]
}
```

**Format rules:**
- `format` is always `"long"` for this pipeline (kept for schema parity with Grafyte, which also supports `"short"`; Splurj does not draft standalone Shorts blueprints).
- Segment count: 45-56 segments × 15s ≈ 11.25-14 min, matching the 1,800-2,500 word script length already specified in Splurj's master prompt.
- `start`/`end` are advisory scene-structure timing; actual clip duration is derived from real TTS audio length, exactly as in Grafyte.
- When a scene is held across consecutive segments (per the master prompt's "hold scenes" rule), `prompt` must be **byte-identical** across those segments — this is what makes Grafyte's existing prompt-hash cache in `images.py` collapse them into one API call.
- `is_short_candidate: true` is set by the drafting model to mark **2-3 separate contiguous runs** of 3-4 segments each (45-60s per run — the hook segment(s) and one or two "counterintuitive twist" or standout-fact moments). The orchestrator groups each contiguous run of `true`-marked segments into its own Short, so a video yields 2-3 Shorts, not one per marked segment.
- `full_text` must equal the space-joined concatenation of all segment `text` fields (validated at load time, same as Grafyte's `_validate_blueprint`).

## 5. Content generation — `splurj_draft.py`

Same shape as Grafyte's `grafyte_draft.py`: one Gemini call (`gemini-3.5-flash`, temperature ~0.7) with a system instruction built from `master_prompt_splurj.txt`'s CONTENT & SCRIPT DNA and VISUAL STYLE DNA sections, given a topic + day number, returns the full blueprint JSON.

System instruction includes, verbatim from the master prompt:
- Hook formula, script rhythm, narrative arc (Hook → Reframe → Science → Twist → System → Modern Mirror → Closing)
- Content safety rules (no financial advice, no outcome promises, disclaimer requirement)
- Visual style DNA (art style, character description, color palette, background-color-by-tone rules)
- **The citation bank (Section 6), pasted in full** — the model is instructed it may only attribute a named claim to a researcher/study that appears in the bank, and must not introduce any other named researcher or specific statistic.

Output is validated (`_validate_draft`-equivalent) before being written to `queue/`:
- Segment count in 45-56 range
- `full_text` word count in 1,800-2,500
- Every named researcher/study mention in `full_text` cross-checked against `citation_bank.md` (see Section 6) — if an unlisted name appears, the draft is rejected and regenerated (up to 2 retries), then flagged in the log rather than silently discarded if retries also fail.
- Every segment `prompt` contains the style-lock closing string
- Exactly 2-3 distinct contiguous runs of `is_short_candidate: true` segments, each run 3-4 segments long

## 6. Citation safety — `channel_data/citation_bank.md`

Because the pipeline runs with no human review, this file is the single control against fabricated or misattributed research claims. It is pasted into the drafting system instruction and re-checked by the QA gate. Initial contents (pre-verified against the Content Safety & Credibility Rules in the master prompt):

```markdown
# Splurj Citation Bank
Only these researchers/studies may be named in a script. Do not introduce others.

1. Daniel Kahneman & Amos Tversky — Prospect Theory (1979, Econometrica). Losses
   loom roughly twice as large as equivalent gains ("loss aversion"); people
   evaluate outcomes relative to a reference point, not absolute wealth.

2. Richard Thaler — Mental accounting (Thaler, 1985/1999). People sort money
   into non-fungible mental buckets (rent, fun money, savings) rather than
   treating money as fully interchangeable. Also: nudge theory (Thaler &
   Sunstein), choice architecture shaping behavior without restricting options.

3. Dan Ariely — "Pain of paying" / decoupling of payment from consumption
   (discussed across Ariely's published work on irrational spending behavior).
   Also: anchoring on arbitrary numbers affecting willingness to pay (Ariely,
   Loewenstein & Prelec, "Coherent Arbitrariness," QJE 2003).

4. Drazen Prelec & Duncan Simester — "Always Leave Home Without It" (Marketing
   Letters, 2001). MIT Sloan auction for Boston Celtics tickets: bidders told
   they'd pay by credit card bid roughly double what cash-bidders bid for
   identical tickets.

5. Hal Hershfield — Future self-continuity research (e.g., Hershfield et al.,
   "Increasing Saving Behavior Through Age-Progressed Renderings of the Future
   Self," Journal of Marketing Research, 2011). People who feel more
   psychologically connected to their future self save more; age-progressed
   avatars increased hypothetical retirement allocations.

6. Brian Knutson et al. — "Neural Predictors of Purchases" (Knutson, Rick,
   Wimmer, Prelec, Loewenstein; Neuron, 2007). fMRI: nucleus accumbens
   activation tracks anticipated product pleasure; insula activation tracks
   price-related discomfort; the balance between the two predicted purchase
   decisions before conscious choice.

7. Elizabeth Dunn — Prosocial spending (Dunn, Aknin, Norton, "Spending Money
   on Others Promotes Happiness," Science, 2008). Spending on others produced
   greater happiness gains than spending the same amount on oneself,
   independent of income level.

8. Kathleen Vohs — Money priming (Vohs, Mead, Goode, "The Psychological
   Consequences of Money," Science, 2006). CONTESTED: several money-priming
   effects have had mixed/failed replications. Only usable if the script
   explicitly flags the contested replication status, per master prompt rules
   — otherwise cut this one.
```

This file is meant to grow over time as new verified studies are added by whoever maintains the channel — it is not meant to be exhaustive on day one, only to be a hard boundary the model cannot step outside of.

## 7. Image generation

`engine/images.py` is ported from Grafyte with two changes:
1. Default provider/model: `IMAGE_PROVIDER=gemini`, `GEMINI_IMAGE_MODEL=gemini-3.1-flash-image` (not `gemini-3-pro-image-preview`).
2. `_generate_gemini` gains an optional `reference_image_path` parameter. When set, `contents` becomes a list `[reference_image_part, prompt_text]` instead of a bare string, using Gemini's multi-image input support, so every scene generation is conditioned on the channel's locked character/prop sheet.

One-time setup step (not per-video): generate `channel_data/character_reference.png` — a neutral-pose sheet of the main stick-figure character plus recurring props (wallet, piggy bank, price tag) on a white background, using the master prompt's VISUAL STYLE DNA section as the prompt. This mirrors Grafyte's `CREW_ref.png` approach.

The existing prompt-hash cache in `images.py` (`cache/images/<md5>.png`) is unchanged and is what makes held-scene segments cheap — no code change needed there, only the drafting instruction to repeat identical prompts across held segments (Section 4/5).

## 8. Video assembly

`engine/video.py` ported from Grafyte with:
- Canvas changed from 1080×1920 (9:16) to **1920×1080 (16:9)** in both `create_segment_video`'s zoompan filter and `finalize`'s scale/pad filter.
- Ken Burns zoom (100%→104% over the clip) kept as-is — works fine for flat doodle stills, adds subtle life without motion-render cost.
- Ambient audio mixing kept as-is (optional, `-15dB` default).
- **New: Shorts auto-cut.** After the long-form final render, for each contiguous run of `is_short_candidate: true` segments: re-encode that sub-range's already-assembled clips as a 1080×1920 crop (`scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920`), burn in an ALL-CAPS hook caption (via `drawtext`) pulled from that segment's `text`, and render as its own MP4 in `output/`. No new TTS or image generation — these are cut from assets already produced for the long-form video.

## 9. Publishing — `engine/youtube.py`

Ported from Grafyte with:
- `DEFAULT_CATEGORY` changed from `"24"` (Entertainment) to `"27"` (Education).
- Description builder (currently inline in `grafyte_engine.py`'s upload step) always appends the line `"This video is for education and entertainment only. It is not financial advice."` as a hardcoded string — not sourced from the model's generated `metadata.description` — so no bad generation can ever drop it.
- The long-form upload uses `youtube.com/watch` framing in logs (not the hardcoded `/shorts/` URL Grafyte's logger assumes); the Shorts uploads use the `/shorts/` framing and get `#Shorts` appended to their own descriptions.
- **New:** a custom thumbnail generation step — one extra Gemini image call per video (concept-text-frame style: a large central prop + bold ALL-CAPS title words, per the master prompt's "Concept text frame" pattern), uploaded via `youtube.thumbnails().set()` after the video upload completes. Grafyte does not do this; it's a Splurj-specific addition for CTR.
- `YOUTUBE_PRIVACY` default kept at `"private"` for the very first manual test run per Grafyte's convention, then switched to whatever cadence privacy the user wants once OAuth and one full run are confirmed working (this is a config choice at setup time, not an architecture decision).

## 10. Scheduling & queue

Unchanged from Grafyte: `run_queue.py` picks the earliest file in `queue/*.json`, invokes `splurj_engine.py --input <file>`, and Windows Task Scheduler runs `scheduler/splurj_daily.ps1` daily. Blueprints are produced ahead of time by running `splurj_draft.py "<topic>" --day N` (which can itself be scheduled, or run manually to keep a rolling buffer in `queue/`).

## 11. Environment variables

```env
ELEVENLABS_API_KEY=
ELEVENLABS_VOICE_ID=          # calm/curious/2nd-person voice — pick at setup from elevenlabs.io/voice-library
ELEVENLABS_MODEL=eleven_turbo_v2

IMAGE_PROVIDER=gemini
GEMINI_IMAGE_MODEL=gemini-3.1-flash-image
GEMINI_API_KEY=

YOUTUBE_CLIENT_SECRET=./client_secret.json
YOUTUBE_PRIVACY=private
YOUTUBE_CATEGORY_ID=27

AMBIENT_DB=-15
SHORT_MIN_SEGMENTS=3          # min contiguous is_short_candidate segments to cut a Short (45s)
SHORT_MAX_SEGMENTS=4          # max contiguous segments per Short (60s)
```

## 12. Cost model (recap from prior discussion)

| Component | Rate | Per video | 
|---|---|---|
| ElevenLabs TTS (turbo) | $22/mo Creator plan (440k turbo chars) or $0.05/1k overage | ~$0.65-0.73 |
| Gemini scene images (~18 unique, flash-image, 1K) | $0.067/image | ~$1.20 |
| Gemini thumbnail image | $0.067/image | ~$0.07 |
| Gemini blueprint draft (Flash) | $1.50/$9 per 1M in/out | ~$0.08 |
| Gemini citation QA pass (Flash-Lite) | $0.25/$1.50 per 1M | ~$0.003 |
| YouTube upload (long-form + shorts) | Free (quota-limited) | $0 |
| **Total** | | **~$2.00-2.10/video, ~$80-90/month at daily cadence** |

## 13. Error handling / QA gates

- Blueprint structural validation (required keys, non-empty timeline) — ported from Grafyte's `_validate_blueprint`, unchanged.
- Citation QA gate (Section 5/6) — new, hard-fails/regenerates rather than warns, since there's no human reviewer to catch it downstream.
- All existing retry/backoff behavior in `audio.py`, `images.py`, `youtube.py` (429/5xx handling) is kept as-is.
- If a blueprint fails citation QA after 2 regeneration attempts, it is left in `queue/` with a `.rejected` suffix and logged loudly rather than silently deleted or force-published — this is a deliberate manual-intervention trapdoor for the one failure mode automation can't safely resolve on its own.

## 14. Out of scope / follow-ups (not part of this spec)

- No web dashboard for queue management (the existing Supabase/Resend MCP servers enabled in `.claude/settings.local.json` are not used by this spec — flagged as a possible future project, not this one).
- No automatic topic-selection/research step — topics are still supplied manually to `splurj_draft.py` (or a future scheduled job could pick from a backlog file, but that's not designed here).
- No A/B thumbnail testing or analytics feedback loop.
- ElevenLabs voice ID selection is a one-time manual setup task (browse elevenlabs.io/voice-library for a calm/curious/2nd-person voice), not an engineering decision — left as a setup checklist item, not a spec placeholder.
