# src/keyframe/random_selector.py
# Random keyframe selector for comparison/testing purposes.

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence, Any

import os
import cv2
import numpy as np
import random


@dataclass(frozen=True)
class Keyframe:
    frame_idx: int
    score: float
    scene_id: int


class RandomSelector:
    """
    Random keyframe selector that picks keyframes randomly within each scene.
    Useful for baseline comparison or testing.
    """
    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _sample_indices(s: int, e: int, stride: int, cap: Optional[int]) -> List[int]:
        idxs = list(range(s, e + 1, max(1, stride)))
        if cap is not None and cap > 0 and len(idxs) > cap:
            sel = np.linspace(0, len(idxs) - 1, cap, dtype=int)
            idxs = [idxs[i] for i in sel]
        return idxs

    def select_for_scene(
        self,
        video_path: str,
        scene_range: Tuple[int, int],
        sample_stride: int = 10,
        max_frames_per_scene: int = 30,
        keyframes_per_scene: int = 1,
        nms_radius: int = 2,
        resize_to: Optional[Tuple[int, int]] = (320, 180),
        scene_id: int = -1,
        batch_pairs: int = 16,
    ) -> List[Keyframe]:
        s, e = scene_range
        idxs = self._sample_indices(s, e, sample_stride, max_frames_per_scene)
        if not idxs:
            return []

        # Randomly select keyframes_per_scene from available indices
        if len(idxs) <= keyframes_per_scene:
            selected_indices = idxs
        else:
            selected_indices = random.sample(idxs, keyframes_per_scene)

        # Create keyframes with random scores (for compatibility)
        keyframes = []
        for frame_idx in sorted(selected_indices):
            score = random.random()  # Random score between 0 and 1
            keyframes.append(Keyframe(frame_idx=frame_idx, score=score, scene_id=scene_id))

        return keyframes


# Convenience function to build + run selector
def select_keyframes_for_scenes(
    video_path: str,
    scenes: Sequence[Tuple[int, int]],
    sample_stride: int = 10,
    max_frames_per_scene: int = 30,
    keyframes_per_scene: int = 1,
    nms_radius: int = 2,
    resize_to: Optional[Tuple[int, int]] = (320, 180),
    batch_pairs: int = 16,
    seed: Optional[int] = None,
) -> List[Keyframe]:
    selector = RandomSelector(seed=seed)

    all_keys: List[Keyframe] = []
    for sid, (s, e) in enumerate(scenes):
        ks = selector.select_for_scene(
            video_path=video_path,
            scene_range=(s, e),
            sample_stride=sample_stride,
            max_frames_per_scene=max_frames_per_scene,
            keyframes_per_scene=keyframes_per_scene,
            nms_radius=nms_radius,
            resize_to=resize_to,
            scene_id=sid,
            batch_pairs=batch_pairs,
        )
        all_keys.extend(ks)
    return all_keys