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
