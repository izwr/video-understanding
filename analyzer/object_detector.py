from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass
class FrameDetections:
    frame_index: int
    detections: list[Detection] = field(default_factory=list)

    @property
    def person_detections(self) -> list[Detection]:
        return [d for d in self.detections if d.class_id == PERSON_CLASS_ID]


class ObjectDetector:
    def __init__(self, model_name: str = "yolo11n.pt") -> None:
        self.model = YOLO(model_name)

    def detect_frames(self, frames: list[np.ndarray]) -> list[FrameDetections]:
        results = self.model(frames, verbose=False)
        all_detections: list[FrameDetections] = []
        for frame_idx, result in enumerate(results):
            boxes = result.boxes
            detections: list[Detection] = []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=result.names[cls_id],
                        confidence=float(boxes.conf[i].item()),
                        bbox_xyxy=tuple(boxes.xyxy[i].tolist()),
                    )
                )
            all_detections.append(FrameDetections(frame_index=frame_idx, detections=detections))
        return all_detections


def build_object_inventory(frame_detections: list[FrameDetections]) -> dict[str, int]:
    """Build a mapping of object name to the number of frames it appeared in."""
    counts: dict[str, set[int]] = defaultdict(set)
    for fd in frame_detections:
        for d in fd.detections:
            counts[d.class_name].add(fd.frame_index)
    return {name: len(frame_set) for name, frame_set in sorted(counts.items())}
