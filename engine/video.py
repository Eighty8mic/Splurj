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
            # Escape the colon in Windows font path for ffmpeg filter syntax
            escaped_font_path = font_path.replace(":", r"\:")
            vf = (
                "scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,"
                f"drawtext=fontfile='{escaped_font_path}':text='{caption}':fontcolor=white:"
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
