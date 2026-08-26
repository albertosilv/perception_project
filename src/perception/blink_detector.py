from __future__ import annotations
import time
import numpy as np
from scipy.spatial import distance as dist
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)

RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX = list(range(42, 48))


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    a = dist.euclidean(eye_points[1], eye_points[5])
    b = dist.euclidean(eye_points[2], eye_points[4])
    c = dist.euclidean(eye_points[0], eye_points[3])
    return (a + b) / (2.0 * c)


class BlinkDetector(BaseDetector):
    def __init__(self, cfg: SimpleNamespace):
        self.ear_threshold = cfg.ear_threshold
        self.min_blink_duration_s = getattr(cfg, "min_blink_duration_s", 0.2)
        self._closed_since: dict[int, float | None] = {}

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        now = time.time()

        for i, face in enumerate(faces):
            if face.landmarks is None:
                continue

            right_eye = face.landmarks[RIGHT_EYE_IDX]
            left_eye = face.landmarks[LEFT_EYE_IDX]

            ear = (eye_aspect_ratio(right_eye) + eye_aspect_ratio(left_eye)) / 2.0
            face.extra["ear"] = round(float(ear), 3)

            closed_since = self._closed_since.get(i)
            blinked = False

            if ear < self.ear_threshold:
                if closed_since is None:
                    self._closed_since[i] = now
            else:
                if closed_since is not None:
                    duration = now - closed_since
                    if duration >= self.min_blink_duration_s:
                        blinked = True
                        # ÚNICO LOG: quando a piscada é contabilizada
                        log.info(f"Piscada detectada - Face {i} | Duração: {duration:.3f}s")
                self._closed_since[i] = None

            face.extra["blinked"] = blinked

        return faces