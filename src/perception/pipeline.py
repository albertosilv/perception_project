"""
Orquestra os detectores de percepção em sequência sobre um frame.

Este é o ponto central para adicionar novas features: crie seu detector
(herdando de BaseDetector), instancie aqui e adicione uma etapa em
`process()`.
"""

from __future__ import annotations
import time
import numpy as np
from types import SimpleNamespace

from src.perception.face_detector import FaceDetector
from src.perception.landmark_detector import LandmarkDetector
from src.perception.types import FrameResult
from src.perception.blink_detector import BlinkDetector
from src.perception.head_pose_detector import HeadPoseDetector
from src.core.logger import get_logger
from src.perception.yawn_detector import YawnDetector
from src.perception.closed_eyes_detector import ClosedEyesDetector
from src.perception.perclos_detector import PerclosDetector

log = get_logger(__name__)


class PerceptionPipeline:
    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.face_detector = FaceDetector(cfg.face_detection)

        self.landmarks_enabled = cfg.landmarks.enabled
        self.landmark_detector = (
            LandmarkDetector(cfg.landmarks) if self.landmarks_enabled else None
        )

        self._last_time = time.time()

        self.blink_enabled = getattr(cfg, "blink", None) is not None and cfg.blink.enabled
        self.blink_detector = BlinkDetector(cfg.blink) if self.blink_enabled else None

        self.head_pose_enabled = getattr(cfg, "head_pose", None) is not None and cfg.head_pose.enabled
        self.head_pose_detector = HeadPoseDetector(cfg.head_pose) if self.head_pose_enabled else None

        self.yawn_enabled = getattr(cfg, "yawn", None) is not None and cfg.yawn.enabled
        self.yawn_detector = YawnDetector(cfg.yawn) if self.yawn_enabled else None

        self.closed_eyes_enabled = getattr(cfg, "closed_eyes", None) is not None and cfg.closed_eyes.enabled
        self.closed_eyes_detector = ClosedEyesDetector(cfg.closed_eyes) if self.closed_eyes_enabled else None

        # PERCLOS depende de eye_closed_ratio (closed_eyes) e dos eventos de
        # blink/yawn — precisa rodar depois desses três no pipeline.
        self.perclos_enabled = getattr(cfg, "perclos", None) is not None and cfg.perclos.enabled
        self.perclos_detector = PerclosDetector(cfg.perclos) if self.perclos_enabled else None

        

    def process(self, frame: np.ndarray) -> FrameResult:
        faces = self.face_detector.detect(frame)

        if self.landmarks_enabled and faces:
            faces = self.landmark_detector.detect(frame, faces)

            if self.blink_enabled:
                faces = self.blink_detector.detect(frame, faces)
            if self.head_pose_enabled:
                faces = self.head_pose_detector.detect(frame, faces)
            if self.yawn_enabled:
                faces = self.yawn_detector.detect(frame, faces)
            if self.closed_eyes_enabled:
                faces = self.closed_eyes_detector.detect(frame, faces)
            if self.perclos_enabled:
                faces = self.perclos_detector.detect(frame, faces)

        fps = self._compute_fps()
        return FrameResult(faces=faces, fps=fps)

    def _compute_fps(self) -> float:
        now = time.time()
        elapsed = now - self._last_time
        self._last_time = now
        return 1.0 / elapsed if elapsed > 0 else 0.0
