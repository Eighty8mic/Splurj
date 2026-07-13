import subprocess
from pathlib import Path

import pytest

from engine.video import CAPTION_MAX_CHARS_PER_LINE, VideoAssembler, probe_video_resolution


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


def test_extract_shorts_preserves_apostrophes_and_percent_signs(tmp_path, fixture_image, fixture_audio):
    """Regression test: apostrophes and '%' must survive into the burned-in caption.

    The old implementation escaped apostrophes with a backslash inside a
    single-quoted drawtext text='...' value, which ffmpeg's drawtext parser
    interprets as *removing* the apostrophe rather than escaping it. It also
    passed raw '%' straight into the filter string, which ffmpeg's strftime-style
    expansion treats as a "stray %" and silently blanks the entire caption.
    Both are silent failures (exit code 0) with no caption text rendered.

    The fix writes the caption to a textfile= instead of embedding it directly,
    which fixes the apostrophe-escaping issue — but textfile= content still goes
    through drawtext's expansion parser by default, so the '%' issue is fixed
    separately by passing expansion=none.
    """
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")

    caption_text = "It's 50% off: don't wait!"
    segments = [
        {"text": caption_text, "is_short_candidate": True},
    ]
    clips = [
        assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "clip_00.mp4", 1.0),
    ]

    output_dir = tmp_path / "shorts"
    shorts = assembler.extract_shorts(clips, segments, output_dir)

    assert len(shorts) == 1
    short_path = shorts[0]
    assert short_path.exists()
    assert probe_video_resolution(short_path) == (1080, 1920)

    caption_file = tmp_path / "short_00_caption.txt"
    assert caption_file.exists()
    # Captions are wrapped to fit the frame; apostrophes and '%' must survive
    # the wrapping, so compare with line breaks collapsed back to spaces.
    assert caption_file.read_text(encoding="utf-8").replace("\n", " ") == caption_text.upper()

    # Real-render assertion: the textfile content above is the INPUT to ffmpeg's
    # drawtext filter, not proof anything was actually drawn. drawtext's
    # `expansion` option defaults to "normal", which applies strftime/expansion
    # syntax to textfile= content too — a bare '%' in the caption ("50%") then
    # triggers a "Stray %" warning and ffmpeg silently draws zero pixels while
    # still exiting 0. Extract a raw grayscale frame from the caption band and
    # confirm bright (white-text) pixels actually landed on screen, since the
    # fixture frame is a flat blue field with no bright content of its own.
    frame_path = tmp_path / "short_00_frame.raw"
    crop_w, crop_h = 1080, 300
    extract_cmd = [
        "ffmpeg", "-y",
        "-i", str(short_path),
        "-frames:v", "1",
        "-vf", f"crop={crop_w}:{crop_h}:0:0,format=gray",
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        str(frame_path),
    ]
    result = subprocess.run(extract_cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    frame_bytes = frame_path.read_bytes()
    assert len(frame_bytes) == crop_w * crop_h

    BRIGHT_THRESHOLD = 200
    bright_pixel_count = sum(1 for b in frame_bytes if b > BRIGHT_THRESHOLD)
    assert bright_pixel_count > 0, (
        "Expected bright white-caption-text pixels in the caption band but found "
        "none — the '%' in the caption likely triggered ffmpeg drawtext's "
        "'Stray %' silent-blank-render bug (expansion=normal by default)."
    )


def test_wrap_caption_breaks_long_captions_into_screen_width_lines():
    """Regression test: drawtext never wraps text, so any caption wider than the
    1080px frame is clipped off both edges (fontsize=64 fits only ~22 chars).
    The caption must be pre-wrapped into newline-separated lines before it is
    written to the drawtext textfile."""
    text = "You tapped a small piece of plastic and your brain felt no pain at all."

    wrapped = VideoAssembler._wrap_caption(text)
    lines = wrapped.split("\n")

    assert len(lines) > 1
    assert all(len(line) <= CAPTION_MAX_CHARS_PER_LINE for line in lines)
    assert " ".join(lines) == text.upper()


def test_wrap_caption_leaves_short_captions_on_one_line():
    assert VideoAssembler._wrap_caption("Save more.") == "SAVE MORE."


def test_extract_shorts_writes_wrapped_caption_textfile(tmp_path, fixture_image, fixture_audio):
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")

    long_text = "You tapped a small piece of plastic and your brain felt no pain at all."
    segments = [{"text": long_text, "is_short_candidate": True}]
    clips = [
        assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "clip_00.mp4", 1.0),
    ]

    shorts = assembler.extract_shorts(clips, segments, tmp_path / "shorts")

    assert len(shorts) == 1
    assert probe_video_resolution(shorts[0]) == (1080, 1920)

    caption_lines = (tmp_path / "short_00_caption.txt").read_text(encoding="utf-8").split("\n")
    assert len(caption_lines) > 1
    assert all(len(line) <= CAPTION_MAX_CHARS_PER_LINE for line in caption_lines)
    assert " ".join(caption_lines) == long_text.upper()


def test_extract_shorts_caption_textfile_uses_lf_line_endings(tmp_path, fixture_image, fixture_audio):
    """Regression test: Path.write_text translates '\\n' to os.linesep, so on
    Windows the caption textfile got CRLF endings. drawtext treats the stray
    '\\r' as one more line break and renders a phantom blank line between every
    caption line, double-spacing the on-screen caption. read_text() hid the
    bug by translating CRLF back, so assert on the raw bytes."""
    assembler = VideoAssembler(workspace=tmp_path, assets_dir=tmp_path / "assets")

    long_text = "You tapped a small piece of plastic and your brain felt no pain at all."
    segments = [{"text": long_text, "is_short_candidate": True}]
    clips = [
        assembler.create_segment_video(fixture_image, fixture_audio, tmp_path / "clip_00.mp4", 1.0),
    ]

    assembler.extract_shorts(clips, segments, tmp_path / "shorts")

    caption_bytes = (tmp_path / "short_00_caption.txt").read_bytes()
    assert b"\n" in caption_bytes, "expected a wrapped multi-line caption"
    assert b"\r" not in caption_bytes
