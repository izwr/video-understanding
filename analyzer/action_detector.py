from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from pytorchvideo.models.hub import slowfast_r50_detection
from torchvision.transforms import functional as F

# AVA action labels (subset of commonly detected actions)
AVA_ACTION_LABELS = {
    0: "bend/bow",
    1: "crouch/kneel",
    2: "dance",
    3: "fall down",
    4: "get up",
    5: "jump/leap",
    6: "lie/sleep",
    7: "martial art",
    8: "run/jog",
    9: "sit",
    10: "stand",
    11: "swim",
    12: "walk",
    13: "answer phone",
    14: "brush teeth",
    15: "carry/hold (an object)",
    16: "catch (an object)",
    17: "chop",
    18: "climb (e.g. a+ladder)",
    19: "clink glass",
    20: "close (e.g. a+door)",
    21: "cook",
    22: "cut",
    23: "dig",
    24: "dress/put on clothing",
    25: "drink",
    26: "drive (e.g. a+car)",
    27: "eat",
    28: "enter",
    29: "exit",
    30: "extract",
    31: "fishing",
    32: "hit (an object)",
    33: "kick (an object)",
    34: "lift/pick up",
    35: "listen (e.g. to+music)",
    36: "open (e.g. a+window)",
    37: "paint",
    38: "play musical instrument",
    39: "point to (an object)",
    40: "press/push (an object)",
    41: "pull (an object)",
    42: "push (another person)",
    43: "put down",
    44: "read",
    45: "ride (e.g. a+bike)",
    46: "row boat",
    47: "sail boat",
    48: "shoot",
    49: "shovel",
    50: "smoke",
    51: "stir",
    52: "take a photo",
    53: "text on/look at a cellphone",
    54: "throw",
    55: "touch (another person)",
    56: "turn (e.g. a+screwdriver)",
    57: "watch (e.g. TV)",
    58: "work on a computer",
    59: "write",
    60: "fight/hit (a person)",
    61: "give/serve (an object) to (a person)",
    62: "grab (a person)",
    63: "hand clap",
    64: "hand shake",
    65: "hand wave",
    66: "hug (a person)",
    67: "kick (a person)",
    68: "kiss (a person)",
    69: "lift (a person)",
    70: "listen to (a person)",
    71: "play with kids",
    72: "push (another person)",
    73: "sing to (e.g. self)",
    74: "take (an object) from (a person)",
    75: "talk to (e.g. self)",
    76: "watch (a person)",
}


@dataclass
class ActionResult:
    clip_start_frame: int
    person_bbox: tuple[float, float, float, float]
    actions: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ClipActionResults:
    clip_start_frame: int
    clip_end_frame: int
    results: list[ActionResult] = field(default_factory=list)


class ActionDetector:
    def __init__(self, device: str = "cpu", top_k: int = 3, threshold: float = 0.1) -> None:
        self.device = torch.device(device)
        self.top_k = top_k
        self.threshold = threshold
        self.model = slowfast_r50_detection(True)
        self.model = self.model.eval().to(self.device)

    def _preprocess_clip(self, clip_frames: list[np.ndarray]) -> list[torch.Tensor]:
        """Convert BGR frames to normalized slow/fast pathway tensors."""
        processed: list[torch.Tensor] = []
        for frame in clip_frames:
            rgb = frame[:, :, ::-1].copy()  # BGR -> RGB
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            processed.append(tensor)

        # Stack into (T, C, H, W) then resize short side to 256
        clip_tensor = torch.stack(processed)  # (T, C, H, W)
        t, c, h, w = clip_tensor.shape
        if h < w:
            new_h = 256
            new_w = int(w * 256 / h)
        else:
            new_w = 256
            new_h = int(h * 256 / w)
        resized = torch.stack([F.resize(clip_tensor[i], [new_h, new_w]) for i in range(t)])

        # Normalize with ImageNet stats
        mean = torch.tensor([0.45, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.225, 0.224, 0.225]).view(1, 3, 1, 1)
        resized = (resized - mean) / std

        # Rearrange to (1, C, T, H, W) for model input
        clip_cthw = resized.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

        # Split into slow (8 frames) and fast (32 frames) pathways
        # SlowFast R50 detection uses alpha=4: slow=T/4, fast=T
        num_frames = clip_cthw.shape[2]
        slow_idx = torch.linspace(0, num_frames - 1, num_frames // 4).long()
        fast_idx = torch.arange(num_frames)

        slow = clip_cthw[:, :, slow_idx].to(self.device)
        fast = clip_cthw[:, :, fast_idx].to(self.device)
        return [slow, fast]

    def _prepare_bboxes(
        self,
        person_bboxes: list[tuple[float, float, float, float]],
        frame_height: int,
        frame_width: int,
    ) -> torch.Tensor:
        """Convert absolute pixel bboxes to normalized [0,1] with batch index prepended."""
        if not person_bboxes:
            return torch.zeros((0, 5), device=self.device)
        boxes = []
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            boxes.append([
                0,  # batch index
                x1 / frame_width,
                y1 / frame_height,
                x2 / frame_width,
                y2 / frame_height,
            ])
        return torch.tensor(boxes, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def classify_clip(
        self,
        clip_frames: list[np.ndarray],
        person_bboxes: list[tuple[float, float, float, float]],
        clip_start_frame: int = 0,
    ) -> ClipActionResults:
        """Classify actions for detected persons in a clip.

        Args:
            clip_frames: List of BGR frames (ideally 32 frames).
            person_bboxes: List of person bounding boxes in (x1, y1, x2, y2) pixel coords.
            clip_start_frame: Frame index where this clip starts in the original video.

        Returns:
            ClipActionResults with per-person action predictions.
        """
        clip_end_frame = clip_start_frame + len(clip_frames) - 1
        result = ClipActionResults(
            clip_start_frame=clip_start_frame,
            clip_end_frame=clip_end_frame,
        )

        if not person_bboxes or len(clip_frames) < 8:
            return result

        # Pad clip to 32 frames if needed
        while len(clip_frames) < 32:
            clip_frames.append(clip_frames[-1])

        h, w = clip_frames[0].shape[:2]
        inputs = self._preprocess_clip(clip_frames[:32])
        boxes = self._prepare_bboxes(person_bboxes, h, w)

        if boxes.shape[0] == 0:
            return result

        preds = self.model(inputs, boxes)  # (num_persons, num_actions)
        preds = torch.sigmoid(preds)

        for i, bbox in enumerate(person_bboxes):
            scores = preds[i]
            top_indices = scores.topk(min(self.top_k, len(scores))).indices
            actions = []
            for idx in top_indices:
                idx_val = idx.item()
                conf = scores[idx_val].item()
                if conf >= self.threshold:
                    label = AVA_ACTION_LABELS.get(idx_val, f"action_{idx_val}")
                    actions.append((label, round(conf, 3)))
            result.results.append(
                ActionResult(
                    clip_start_frame=clip_start_frame,
                    person_bbox=bbox,
                    actions=actions,
                )
            )

        return result
