from pathlib import Path

import cv2
import numpy as np


def _extract_frames(
    cap: cv2.VideoCapture,
    interval: int,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            frames.append(frame)

            if max_frames is not None and len(frames) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def sample_frames(
    video_path: str | Path,
    interval: int = 30,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Sample frames from a video file at a regular interval.

    Args:
        video_path: Path to the input video file.
        interval: Sample every Nth frame (default: every 30th frame).
        max_frames: Stop after saving this many frames. None means no limit.

    Returns:
        List of frames as numpy arrays (BGR format).
    """
    cap = load_video(video_path)
    return _extract_frames(cap, interval, max_frames)


def load_video(video_path: str | Path) -> cv2.VideoCapture:
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return cap


def get_video_info(video_path: str | Path) -> dict:
    """Get metadata about a video file.

    Returns:
        Dict with fps, total_frames, width, height, duration_seconds.
    """
    cap = load_video(video_path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_seconds"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    cap.release()
    return info


def extract_clip_frames(
    video_path: str | Path,
    start_frame: int,
    num_frames: int = 32,
) -> list[np.ndarray]:
    """Extract a dense clip of consecutive frames starting at start_frame.

    Args:
        video_path: Path to the input video file.
        start_frame: Frame index to start extraction from.
        num_frames: Number of consecutive frames to extract.

    Returns:
        List of frames as numpy arrays (BGR format).
    """
    cap = load_video(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def sample_frames_by_seconds(
    video_path: str | Path,
    every_n_seconds: float = 1.0,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Sample frames from a video at a time-based interval.

    Args:
        video_path: Path to the input video file.
        every_n_seconds: Sample one frame every N seconds (default: 1.0).
        max_frames: Stop after saving this many frames. None means no limit.

    Returns:
        List of frames as numpy arrays (BGR format).
    """
    cap = load_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, round(fps * every_n_seconds))
    return _extract_frames(cap, interval, max_frames)