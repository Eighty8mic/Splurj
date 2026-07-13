# Splurj Automation Engine — Design Spec

Status: Revised per user feedback (dashboard, auto-research, thumbnail variants added to scope), pending implementation plan
Date: 2026-07-06

## 1. Overview

A fully autonomous daily content pipeline for Splurj (@Splurj-it), ported from the proven Grafyte engine architecture (`C:\Users\School\Desktop\Grafyte`) and adapted to Splurj's format, brand rules, and YMYL (financial-content) safety requirements. The pipeline picks a topic, drafts a JSON "blueprint" for one video, generates all assets (narration audio, doodle-style images, thumbnail variants), assembles a 16:9 video with FFmpeg, and uploads a long-form video plus 2-3 auto-cut Shorts to YouTube. A companion web dashboard (Next.js + Supabase, deployed on Vercel) provides visibility into the queue and is the one place a human touches the system — approving newly-discovered research citations before they can be used in scripts.

This is a **port and adapt**, not a from-scratch build: Grafyte's `engine/audio.py`, `engine/youtube.py`, and orchestration patterns carry over with minimal changes. The genuinely new work is: content generation safety (citation bank + QA gate), image consistency (character reference), the 16:9/long-duration video path, the Shorts auto-cut, automatic topic research, and the dashboard.

## 2. Decisions made (from brainstorming)

- **Length**: keep Splurj's existing 10-14 minute long-form format. Segments scale from Grafyte's 13×15s to ~45-56×15s per video, using the "hold scenes across consecutive timestamps" rule already in Splurj's master prompt so most segments reuse a cached image rather than generating a new one every 15s.
- **Autonomy**: the video pipeline itself (draft → render → publish) is fully autonomous, no human checkpoint. The one exception, added in this revision: newly-discovered research citations require human approval before they enter the citation bank (Section 7) — this is the sole recurring human touchpoint in the whole system.
- **Scope**: long-form daily + auto-cut Shorts pulled from the same generation (not a separate Shorts pipeline). Shorts add no marginal API cost — they're re-encoded crops of already-generated segments.
- **Images**: Gemini image generation (`gemini-3.1-flash-image`) with a saved character/prop reference image attached to every scene prompt, for visual consistency and lower cost than the pro-tier model Grafyte defaults to.
- **Topic selection**: automatic. A weekly research step web-searches for new, real, named behavioral-finance/psychology studies, drafts a candidate citation-bank entry, and queues it for human approval in the dashboard. Daily topic selection then draws only from *approved* citations, combined per the master prompt's 5 viral-angle formulas.
- **Dashboard**: Supabase is the source of truth for queue/citation state (not just a mirror of local files). Personal-use only, single-user auth, deployed to Vercel.
- **Notifications**: Resend sends email only for citation-QA gate failures (the one failure mode needing a human). No publish confirmations or digests.
- **Thumbnails**: engine generates 2-3 candidate thumbnails per video; the dashboard surfaces them with a direct link to YouTube Studio's Test & Compare setup (no public API exists for that feature, so it stays a one-click manual step, not full automation).
- **Estimated cost**: ~$2.10-2.20/video blended (ElevenLabs' $22/mo flat plan amortized to ~$0.73/video + ~$1.40-1.48/video variable for images, thumbnail variants, and text-gen) — **roughly $64-67/month total** at daily (~30 videos/mo) cadence: $22 flat + ~$42-44 variable. Research-step grounding and Supabase/Vercel/Resend are all within free tiers at this scale. See Section 14.

## 3. Directory layout

```
Splurj/
├── engine/                        # Python — runs locally, scheduled via Task Scheduler
│   ├── __init__.py
│   ├── audio.py                   # ElevenLabs TTS — ported from Grafyte near-unchanged
│   ├── images.py                  # Gemini image gen — extended for reference-image input
│   ├── video.py                   # FFmpeg assembly — 16:9, + Shorts auto-cut
│   ├── youtube.py                  # YouTube Data API v3 upload — category/disclaimer changes, multi-thumbnail
│   ├── gemini_tools.py             # Script polish + prompt enhancement + citation QA gate
│   ├── research.py                 # NEW — weekly citation-candidate discovery (Gemini + Google Search grounding)
│   ├── topic_picker.py              # NEW — picks next topic from approved citations + angle formulas
│   └── supabase_client.py            # NEW — thin wrapper: queue/citation reads+writes against Supabase
├── queue_local/                   # Local staging for the blueprint currently being rendered (not authoritative)
├── output/                        # Final MP4s (long-form + shorts), also uploaded as Supabase Storage refs
├── workspace/                      # Temp per-run assets (auto-cleaned)
├── assets/ambient/                  # Optional ambient bed audio
├── cache/images/                    # Prompt-hash image cache (reused from Grafyte's images.py)
├── channel_data/
│   └── character_reference.png     # Generated once; the recurring stick-figure/prop sheet
├── dashboard/                      # NEW — Next.js app, deployed to Vercel, talks to the same Supabase project
│   ├── app/                        # queue view, citation approval queue, thumbnail picker
│   └── ...
├── scripts/                        # EXISTING — manual one-off chat-workflow scripts, untouched
├── master_prompt_splurj.txt         # EXISTING — source of all voice/visual rules below
├── splurj_engine.py                 # Main render orchestrator + CLI (was grafyte_engine.py)
├── splurj_draft.py                  # Drafts one blueprint (topic supplied by topic_picker.py or manually)
├── run_daily.py                      # Daily runner: topic_picker → splurj_draft → splurj_engine
├── scheduler/splurj_daily.ps1         # Runs run_daily.py
├── scheduler/splurj_research_weekly.ps1  # Runs research.py once/week
├── .env / .env.example
└── requirements.txt
```

The existing chat-based Stage 1-4 workflow (`scripts/`, the master prompt conversation) is **not replaced**. It remains available for one-off, manually-directed videos. The new engine is an independent, second path used for the daily-automation channel operation.

## 4. Data model (Supabase — source of truth)

```
citations
  id, researchers, study_ref, year, venue, summary, source_url,
  contested (bool), status ('pending' | 'approved' | 'rejected'),
  discovered_at, reviewed_at

used_topics                          -- dedup guard for topic_picker.py
  id, citation_ids (int[]), angle_formula, video_day, created_at

videos
  id, day, format ('long'), title, description, tags,
  status ('drafted' | 'citation_review' | 'rendering' | 'rendered'
          | 'uploaded' | 'rejected'),
  blueprint (jsonb), citation_ids_used (int[]),
  youtube_video_id, shorts_youtube_ids (text[]),
  thumbnail_urls (text[]), created_at, published_at

citation_qa_failures
  id, video_id, attempted_names (text[]), blueprint_snapshot (jsonb),
  created_at
```

`blueprint` jsonb follows the same schema as Section 5. Local files under `queue_local/` and `output/` are working storage for a single in-progress render; once a run completes, its blueprint and status live in Supabase and the local copy can be cleaned up (mirroring Grafyte's existing `--keep-workspace` cleanup behavior).

## 5. Blueprint JSON schema

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
- `is_short_candidate: true` marks **2-3 separate contiguous runs** of 3-4 segments each (45-60s per run — the hook segment(s) and one or two "counterintuitive twist" or standout-fact moments). The orchestrator groups each contiguous run into its own Short.
- `full_text` must equal the space-joined concatenation of all segment `text` fields (validated at load time, same as Grafyte's `_validate_blueprint`).

## 6. Automatic topic selection — `engine/topic_picker.py`

Runs at the start of each daily cycle (`run_daily.py`), before drafting:
1. Query Supabase for `citations` where `status = 'approved'`.
2. Query `used_topics` to see which citations/angle-formula combinations have already been used, and how recently.
3. Pick either a single citation applied to one of the master prompt's 5 angle formulas ("Why your brain ___", "The real reason you can't ___", etc.), or a pairing of two citations not yet paired, preferring combinations that haven't appeared in `used_topics`.
4. Emit a plain topic string + the chosen `citation_ids`, which `splurj_draft.py` then drafts into a full blueprint exactly as if the topic had been typed manually — the manual CLI path (`python splurj_draft.py "topic" --day N`) still works unchanged for one-off manual topics.
5. Record the pick in `used_topics` once the resulting blueprint passes citation QA (Section 8), so a rejected draft doesn't block that combination from being retried later.

With 8 seed citations this already yields dozens of non-repeating topic/angle combinations (5 angles × 8 citations = 40, plus pair combinations), so there's substantial runway before the bank must grow — the weekly research cadence (Section 7) is deliberately slower than the daily video cadence.

## 7. Citation research & approval — `engine/research.py` + dashboard

Because the pipeline runs with no per-video human review, the citation bank is the hard control against fabricated or misattributed research claims, and growing it is the one place a human stays in the loop.

**Weekly discovery run** (`scheduler/splurj_research_weekly.ps1` → `research.py`):
1. Calls Gemini with Google Search grounding, prompted to find a real, named, peer-reviewed or well-documented study in behavioral finance/consumer psychology, matching the channel's niche and not already in the `citations` table.
2. Drafts a candidate row: researchers, study reference, year/venue, one-paragraph factual summary, source URL, and a `contested` flag if the search turns up notable replication concerns (mirroring how the seed bank already flags Vohs' money-priming work as contested).
3. Inserts the candidate into Supabase `citations` with `status = 'pending'`. Nothing about it is usable by `topic_picker.py` or drafting until approved.
4. Sends **no** email for this step itself (only QA failures trigger email, per Section 2) — the dashboard's pending-approval list is the surface for this.

**Dashboard approval queue**: lists each `pending` citation with its summary and source link; a human clicks approve or reject. Approved rows become available to `topic_picker.py` immediately; rejected rows are kept (status `rejected`) so the same source doesn't get re-proposed.

**Seed content** (initial 8 approved rows, migrated into Supabase at setup time):

```markdown
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
   explicitly flags the contested replication status, per master prompt rules.
```

At draft-time, `splurj_draft.py` queries the `approved` rows it needs (per `topic_picker.py`'s selection) and renders them into the Gemini system instruction as a text block — functionally identical to pasting a static file, but always reflecting the latest approved set.

## 8. Content generation & QA gate — `splurj_draft.py`

Same shape as Grafyte's `grafyte_draft.py`: one Gemini call (`gemini-3.5-flash`, temperature ~0.7) with a system instruction built from `master_prompt_splurj.txt`'s CONTENT & SCRIPT DNA and VISUAL STYLE DNA sections, plus the relevant approved citation rows (Section 7), given the topic from `topic_picker.py` (or a manual CLI arg) and a day number. Returns the full blueprint JSON. The model is instructed it may only attribute a named claim to a researcher/study passed in the prompt, and must not introduce any other named researcher or specific statistic.

Output is validated before being written to Supabase as a `videos` row with `status = 'citation_review'`:
- Segment count in 45-56 range; `full_text` word count in 1,800-2,500
- Every named researcher/study mention in `full_text` cross-checked against the citation rows actually passed in — if an unlisted name appears, the draft is rejected and regenerated (up to 2 retries)
- Every segment `prompt` contains the style-lock closing string
- Exactly 2-3 distinct contiguous runs of `is_short_candidate: true` segments, each run 3-4 segments long

If a blueprint still fails citation QA after 2 regeneration attempts, its row moves to `status = 'rejected'`, a row is written to `citation_qa_failures`, and Resend sends the one email this system sends (Section 2) — a deliberate manual-intervention trapdoor for the one failure mode automation can't safely resolve alone. Otherwise `status` becomes `rendering` and `splurj_engine.py` picks it up.

## 9. Image generation

`engine/images.py` is ported from Grafyte with two changes:
1. Default provider/model: `IMAGE_PROVIDER=gemini`, `GEMINI_IMAGE_MODEL=gemini-3.1-flash-image`.
2. `_generate_gemini` gains an optional `reference_image_path` parameter. When set, `contents` becomes a list `[reference_image_part, prompt_text]` instead of a bare string, using Gemini's multi-image input support, so every scene generation is conditioned on the channel's locked character/prop sheet.

One-time setup step (not per-video): generate `channel_data/character_reference.png` — a neutral-pose sheet of the main stick-figure character plus recurring props (wallet, piggy bank, price tag) on a white background, using the master prompt's VISUAL STYLE DNA section as the prompt. Mirrors Grafyte's `CREW_ref.png` approach.

The existing prompt-hash cache in `images.py` (`cache/images/<md5>.png`) is unchanged and is what makes held-scene segments cheap.

## 10. Video assembly

`engine/video.py` ported from Grafyte with:
- Canvas changed from 1080×1920 (9:16) to **1920×1080 (16:9)** in both `create_segment_video`'s zoompan filter and `finalize`'s scale/pad filter.
- Ken Burns zoom (100%→104% over the clip) kept as-is.
- Ambient audio mixing kept as-is (optional, `-15dB` default).
- **Shorts auto-cut**: after the long-form final render, for each contiguous run of `is_short_candidate: true` segments, re-encode that sub-range as a 1080×1920 crop with a burned-in ALL-CAPS hook caption (via `drawtext`) pulled from that segment's `text`, rendered as its own MP4. No new TTS or image generation.

## 11. Publishing & thumbnails — `engine/youtube.py`

Ported from Grafyte with:
- `DEFAULT_CATEGORY` changed from `"24"` (Entertainment) to `"27"` (Education).
- Description builder always appends `"This video is for education and entertainment only. It is not financial advice."` as a hardcoded string, not sourced from the model's generated description, so no bad generation can ever drop it.
- Long-form upload logs use `youtube.com/watch` framing; Shorts uploads use `/shorts/` framing and get `#Shorts` appended to their own descriptions.
- **Thumbnail variants**: instead of one thumbnail image, generate **2-3 candidates** (concept-text-frame style, differing in which prop/phrase is foregrounded). Upload the first as the video's default thumbnail via `youtube.thumbnails().set()` so the video isn't left blank. Store all variant URLs in Supabase `videos.thumbnail_urls`.
- Dashboard surfaces all variants per video with a deep link to that video's YouTube Studio page, so the human can set up native Test & Compare (Section 2) in one click. No programmatic A/B logic — YouTube doesn't expose an API for it.
- `YOUTUBE_PRIVACY` default `"private"` for the first manual test run, then switched once OAuth and one full run are confirmed working.

## 12. Web dashboard — `dashboard/`

Next.js app deployed to Vercel, reading/writing the same Supabase project the engine uses. Single-user auth (Supabase Auth, one pre-approved account) — no public signup.

Pages:
- **Queue** — every `videos` row with its status, links to the rendered output (Supabase Storage or a signed YouTube link once uploaded), and the blueprint for inspection.
- **Citation approvals** — `pending` rows from `citations`, with approve/reject actions.
- **Thumbnails** — per-video variant gallery with a "open in YouTube Studio" deep link.
- **Rejected/QA failures** — `citation_qa_failures` rows, so the human can see why a draft was rejected and manually adjust the citation bank or master prompt if a pattern emerges.

No write path from the dashboard back into a video's blueprint content itself in this version — approving/rejecting citations and viewing status is the full interaction surface. Editing a blueprint's script text is out of scope (Section 13).

## 13. Out of scope / follow-ups (not part of this spec)

- Editing blueprint/script content from the dashboard — approvals are citation-only.
- Programmatic thumbnail A/B swapping via YouTube Analytics — rejected in favor of the manual Test & Compare hand-off (Section 2), since there's no API for the real feature and a custom proxy-metric swap was explicitly not chosen.
- Auto-approving citation candidates without human review — the one deliberate human checkpoint in an otherwise autonomous system; not being removed.
- Multi-channel support, mobile app, or public/multi-user dashboard access.
- ElevenLabs voice ID selection is a one-time manual setup task (browse elevenlabs.io/voice-library for a calm/curious/2nd-person voice), not an engineering decision.

## 14. Cost model (recap)

| Component | Rate | Per video / cadence |
|---|---|---|
| ElevenLabs TTS (turbo) | $22/mo Creator plan (440k turbo chars) or $0.05/1k overage | ~$0.65-0.73/video |
| Gemini scene images (~18 unique, flash-image, 1K) | $0.067/image | ~$1.20/video |
| Gemini thumbnail images (2-3 variants) | $0.067/image | ~$0.13-0.20/video |
| Gemini blueprint draft (Flash) | $1.50/$9 per 1M in/out | ~$0.08/video |
| Gemini citation QA pass (Flash-Lite) | $0.25/$1.50 per 1M | ~$0.003/video |
| Gemini research step w/ Google Search grounding | $14/1,000 grounded queries, 5,000 free/mo | ~$0 (weekly cadence, well under free tier) |
| YouTube upload (long-form + shorts) | Free (quota-limited) | $0 |
| Supabase / Vercel / Resend | Free tiers at this scale | $0 |
| FFmpeg assembly | Local compute | $0 |
| **Total (blended per-video)** | | **~$2.10-2.20/video** |
| **Total (monthly, ~30 videos)** | $22 flat (ElevenLabs) + ~$1.40-1.48/video variable × 30 | **~$64-67/month** |

## 15. Error handling / QA gates

- Blueprint structural validation (required keys, non-empty timeline) — ported from Grafyte's `_validate_blueprint`, unchanged.
- Citation QA gate (Section 8) — hard-fails/regenerates rather than warns, since there's no human reviewer downstream for video content itself.
- All existing retry/backoff behavior in `audio.py`, `images.py`, `youtube.py` (429/5xx handling) is kept as-is.
- A rejected blueprint's row and `citation_qa_failures` entry are retained (not deleted) so failure patterns are visible in the dashboard over time.
