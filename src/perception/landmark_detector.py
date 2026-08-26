"""
Extrai os 68 pontos de referência facial (landmarks) usando o
shape_predictor do dlib, dado um rosto já detectado.
"""

from __future__ import annotations
import dlib
import numpy as np
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)


class LandmarkDetector(BaseDetector):
    def __init__(self, cfg: SimpleNamespace):
        predictor_path = cfg.predictor_path
        try:
            self._predictor = dlib.shape_predictor(predictor_path)
        except RuntimeError as e:
            raise RuntimeError(
                f"Não foi possível carregar o shape_predictor em '{predictor_path}'.\n"
                "Baixe o modelo em: "
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2\n"
                "e extraia para a pasta 'models/'."
            ) from e

        log.info("LandmarkDetector inicializado (68 pontos).")

    def detect(self, frame: np.ndarray, faces: list[Face]) -> list[Face]:
        """
        Preenche o campo `landmarks` de cada Face detectada.

        Args:
            frame: frame BGR original
            faces: lista de Face já detectadas pelo FaceDetector

        Returns:
            A mesma lista de faces, com `landmarks` preenchido.
        """
        gray = frame if frame.ndim == 2 else _to_gray(frame)

        for face in faces:
            x, y, w, h = face.bbox.as_tuple()
            rect = dlib.rectangle(x, y, x + w, y + h)
            shape = self._predictor(gray, rect)
            face.landmarks = np.array(
                [[p.x, p.y] for p in shape.parts()], dtype=np.int32
            )

        return faces


def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
