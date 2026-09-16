"""
FFmpeg video assembly module for Splurj — 16:9 long-form output.

Pipeline stages:
  1. create_segment_video  — pose images + audio -> MP4 clip (crossfade loop,
                              or a Ken Burns zoom when only one pose is given)
  2. concatenate_segments   — clips -> single timeline
  3. mix_ambient_audio      — overlay looping drone at -15 dB (optional)
  4. finalize               — quality encode, pad to 1920x1080, faststart
"""

import json
import logging
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

# drawtext never wraps text, so captions must be pre-wrapped to fit the
# 1080px-wide Shorts frame: Arial Bold at fontsize=64 averages ~44px per
# ALL-CAPS glyph, so 22 chars ≈ 970px, leaving a safe margin on each side.
CAPTION_MAX_CHARS_PER_LINE = 22

# Multi-pose crossfade tuning. A segment's pose sequence is a "boomerang" that
# starts and ends on pose 0 -- [0, 1, ..., N-1, ..., 1, 0] -- repeated enough
# times to fill the segment's real audio duration. Because every independently
# rendered segment in a held scene both starts and ends on the same pose-0
# frame, the hard cut at each segment boundary lands on two identical frames
# and is imperceptible, same as today's single-image segments.
XFADE_SECONDS = 0.5
XFADE_TRANSITION = "fade"
TARGET_LAP_SECONDS = 7.0
MIN_POSE_CLIP_SECONDS = 0.9
MAX_LAPS_PER_SEGMENT = 2


class PoseSequence(NamedTuple):
    pose_indices: List[int]   # which of the original pose images survive thinning
    node_indices: List[int]   # boomerang sequence, as LOCAL indices into pose_indices
    clip_seconds: float       # per-node hold duration
    xfade_seconds: float      # 0.0 for the single-pose static-zoom fallback


def _evenly_spaced_indices(total: int, count: int) -> List[int]:
    if count >= total:
        return list(range(total))
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def _boomerang_nodes(n: int, laps: int) -> List[int]:
    lap = list(range(1, n)) + list(range(n - 2, -1, -1))
    return [0] + lap * laps


def build_pose_sequence(
    n_poses: int,
    target_duration: float,
    xfade_seconds: float = XFADE_SECONDS,
    target_lap_seconds: float = TARGET_LAP_SECONDS,
) -> PoseSequence:
    """Work out how to animate through n_poses images over target_duration seconds.

    Tries the full pose count first, thinning down (keeping the first and last
    pose) whenever the per-node hold duration would be too short to read as a
    real transition, until it falls back to a single static pose.
    """
    for n in range(n_poses, 1, -1):
        lap_transitions = 2 * n - 2
        laps = min(max(round(target_duration / target_lap_seconds), 1), MAX_LAPS_PER_SEGMENT)
        transitions = laps * lap_transitions
        nodes = 1 + transitions
        clip_seconds = (target_duration + transitions * xfade_seconds) / nodes
        if clip_seconds >= MIN_POSE_CLIP_SECONDS:
            return PoseSequence(
                pose_indices=_evenly_spaced_indices(n_poses, n),
                node_indices=_boomerang_nodes(n, laps),
                clip_seconds=clip_seconds,
                xfade_seconds=xfade_seconds,
            )

    return PoseSequence(pose_indices=[0], node_indices=[0], clip_seconds=target_duration, xfade_seconds=0.0)


def build_xfade_filtergraph(image_paths: List[Path], sequence: PoseSequence) -> Tuple[List[Path], str]:
    """Build the ffmpeg filter_complex for a chained N-way crossfade of stills.

    Caller's responsibility: only call this when sequence.node_indices has 2+
    entries (a 1-node sequence is the static-zoom fallback, handled separately).
    """
    ordered_paths = [image_paths[sequence.pose_indices[node]] for node in sequence.node_indices]
    d = sequence.xfade_seconds
    m = len(ordered_paths)

    per_input = [
        f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,format=yuv420p,fps=30[v{i}]"
        for i in range(m)
    ]

    chain = []
    prev_label = "v0"
    for i in range(1, m):
        offset = i * (sequence.clip_seconds - d)
        out_label = f"x{i}" if i < m - 1 else "vout"
        chain.append(f"[{prev_label}][v{i}]xfade=transition={XFADE_TRANSITION}:duration={d}:offset={offset:.3f}[{out_label}]")
        prev_label = out_label

    return ordered_paths, ";".join(per_input + chain)


def _build_static_zoom_filter(duration: float) -> str:
    frames = max(1, int(duration * 30))
    return (
        f"scale=1920:1080:force_original_aspect_ratio=increase,"
        f"crop=1920:1080,"
        f"zoompan=z='min(zoom+0.0001,1.04)':d={frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30"
    )


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
        self, image_paths: List[Path], audio_path: Path, output_path: Path, duration: float
    ) -> Path:
        sequence = build_pose_sequence(len(image_paths), duration)

        if len(sequence.node_indices) == 1:
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1",
                "-framerate", "30",
                "-i", str(image_paths[sequence.pose_indices[0]]),
                "-i", str(audio_path),
                "-vf", _build_static_zoom_filter(duration),
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
            logger.info("Segment done (static): %s", output_path.name)
            return output_path

        ordered_paths, filter_complex = build_xfade_filtergraph(image_paths, sequence)
        audio_input_index = len(ordered_paths)

        cmd = ["ffmpeg", "-y"]
        for pose_path in ordered_paths:
            cmd += ["-loop", "1", "-framerate", "30", "-t", str(sequence.clip_seconds), "-i", str(pose_path)]
        cmd += [
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", f"{audio_input_index}:a",
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
        logger.info("Segment done (%d-pose crossfade): %s", len(sequence.pose_indices), output_path.name)
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

    def mix_ambient_audio(
        self,
        video_path: Path,
        output_path: Path,
        ambient_db: float = -15.0,
        ambient_track: Optional[Path] = None,
    ) -> Path:
        if ambient_track is None:
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

    @staticmethod
    def _wrap_caption(text: str) -> str:
        """Upper-case a caption and wrap it into lines that fit the 1080px frame."""
        return textwrap.fill(text.upper(), width=CAPTION_MAX_CHARS_PER_LINE)

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

            caption = self._wrap_caption(segments[start]["text"])
            caption_path = self.workspace / f"short_{i:02d}_caption.txt"
            # newline="\n": drawtext renders a CRLF as two line breaks, which
            # double-spaces the caption when Windows translates "\n" on write.
            caption_path.write_text(caption, encoding="utf-8", newline="\n")

            # Escape the colon in Windows font/text-file paths for ffmpeg filter syntax,
            # and use forward slashes so the path itself doesn't confuse the filter parser.
            escaped_font_path = font_path.replace(":", r"\:")
            escaped_caption_path = caption_path.as_posix().replace(":", r"\:")
            vf = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,"
                f"drawtext=fontfile='{escaped_font_path}':textfile='{escaped_caption_path}':fontcolor=white:"
                "fontsize=64:borderw=4:bordercolor=black:x=(w-text_w)/2:y=120:"
                "line_spacing=10:box=0:expansion=none"
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
