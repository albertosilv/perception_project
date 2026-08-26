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
from src.core.logger import get_logger

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

        # -----------------------------------------------------------
        # Ponto de extensão: instancie aqui novos detectores, ex:
        # self.emotion_detector = EmotionDetector(cfg.emotion)
        # self.gaze_detector = GazeDetector(cfg.gaze)
        # -----------------------------------------------------------

    def process(self, frame: np.ndarray) -> FrameResult:
        """Executa todas as etapas de percepção habilitadas sobre um frame."""
        faces = self.face_detector.detect(frame)

        if self.landmarks_enabled and faces:
            faces = self.landmark_detector.detect(frame, faces)

        # -----------------------------------------------------------
        # Ponto de extensão: rode novos detectores sobre `faces`, ex:
        # faces = self.emotion_detector.detect(frame, faces)
        # faces = self.gaze_detector.detect(frame, faces)
        # -----------------------------------------------------------

        fps = self._compute_fps()
        return FrameResult(faces=faces, fps=fps)

    def _compute_fps(self) -> float:
        now = time.time()
        elapsed = now - self._last_time
        self._last_time = now
        return 1.0 / elapsed if elapsed > 0 else 0.0
