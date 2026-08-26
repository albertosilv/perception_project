"""
Detector de rostos com dois backends intercambiáveis:

- "haar": cv2.CascadeClassifier (mais rápido, menos preciso)
- "dlib_hog": dlib.get_frontal_face_detector (mais preciso, mais lento)

Trocar o backend é só uma linha no config.yaml — o resto do pipeline
não precisa saber qual está sendo usado.
"""

from __future__ import annotations
import cv2
import dlib
import numpy as np
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face, BoundingBox
from src.core.logger import get_logger

log = get_logger(__name__)


class FaceDetector(BaseDetector):
    def __init__(self, cfg: SimpleNamespace):
        self.backend = cfg.backend
        self.upsample = getattr(cfg, "upsample", 1)

        if self.backend == "haar":
            cascade_path = cfg.haar_cascade_path
            self._model = cv2.CascadeClassifier(cascade_path)
            if self._model.empty():
                raise RuntimeError(
                    f"Não foi possível carregar o Haar Cascade em: {cascade_path}"
                )
        elif self.backend == "dlib_hog":
            self._model = dlib.get_frontal_face_detector()
        else:
            raise ValueError(
                f"Backend de detecção facial desconhecido: '{self.backend}'. "
                "Use 'haar' ou 'dlib_hog'."
            )

        log.info(f"FaceDetector inicializado com backend='{self.backend}'")

    def detect(self, frame: np.ndarray) -> list[Face]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.backend == "haar":
            return self._detect_haar(gray)
        return self._detect_dlib(gray)

    def _detect_haar(self, gray: np.ndarray) -> list[Face]:
        detections = self._model.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return [
            Face(bbox=BoundingBox(x, y, w, h))
            for (x, y, w, h) in detections
        ]

    def _detect_dlib(self, gray: np.ndarray) -> list[Face]:
        detections = self._model(gray, self.upsample)
        faces = []
        for rect in detections:
            x, y = rect.left(), rect.top()
            w, h = rect.right() - x, rect.bottom() - y
            faces.append(Face(bbox=BoundingBox(x, y, w, h)))
        return faces
