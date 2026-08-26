# src/perception/fatigue_detectors.py

from __future__ import annotations
import numpy as np
from scipy.spatial import distance as dist
from types import SimpleNamespace
from collections import deque
import time

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)

# Índices dos pontos faciais (modelo 68 pontos)
RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX = list(range(42, 48))
MOUTH_IDX = list(range(48, 68))


def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Calcula a EAR (Eye Aspect Ratio) para detecção de fadiga.
    Quanto menor o valor, mais fechado o olho.
    
    Args:
        eye_points: Array com os 6 pontos do olho
        
    Returns:
        float: EAR (0 = olho fechado, ~0.25 = olho aberto)
    """
    a = dist.euclidean(eye_points[1], eye_points[5])
    b = dist.euclidean(eye_points[2], eye_points[4])
    c = dist.euclidean(eye_points[0], eye_points[3])
    
    if c == 0:
        return 0
    
    return (a + b) / (2.0 * c)


def eye_closed_ratio(eye_points: np.ndarray) -> float:
    """
    Calcula a proporção de fechamento do olho.
    
    Args:
        eye_points: Array com os pontos do olho
        
    Returns:
        float: 0 = completamente aberto, 1 = completamente fechado
    """
    ear = eye_aspect_ratio(eye_points)
    closed_ratio = max(0, min(1, (0.25 - ear) / 0.15))
    return closed_ratio


# ============================================================================
# DETECTOR DE OLHOS FECHADOS (SONOLÊNCIA)
# ============================================================================

class ClosedEyesDetector(BaseDetector):
    """
    Detector de olhos fechados por longo período (sonolência).
    Detecta quando a pessoa mantém os olhos fechados por muito tempo.
    """
    
    def __init__(self, cfg: SimpleNamespace):
        # cfg é um SimpleNamespace (não dict) — usar getattr, não .get()
        self.closed_threshold = getattr(cfg, "closed_eye_threshold", 0.3)
        # Duração mínima (segundos) com os olhos fechados para caracterizar
        # microssono — medida por relógio, não por nº de frames.
        self.min_duration_s = getattr(cfg, "min_duration_s", 3.0)
        self._closed_since: dict[int, float | None] = {}

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        now = time.time()
        for i, face in enumerate(faces):
            if face.landmarks is None:
                continue

            right_eye = face.landmarks[RIGHT_EYE_IDX]
            left_eye = face.landmarks[LEFT_EYE_IDX]

            closed_ratio = (eye_closed_ratio(right_eye) + eye_closed_ratio(left_eye)) / 2.0
            face.extra["eye_closed_ratio"] = round(float(closed_ratio), 3)

            closed_since = self._closed_since.get(i)
            eyes_closed = False
            
            if closed_ratio > self.closed_threshold:
                if closed_since is None:
                    self._closed_since[i] = now
                else:
                    duration = now - closed_since
                    if duration >= self.min_duration_s:
                        eyes_closed = True
                        # ÚNICO LOG: quando os olhos ficam fechados por tempo suficiente
                        log.warning(f"Olhos fechados por {duration:.1f}s - Face {i} | ALERTA DE SONOLÊNCIA")
            else:
                # Olhos abertos - reseta o contador
                if closed_since is not None:
                    duration = now - closed_since
                    if duration >= self.min_duration_s:
                        log.info(f"Olhos reabertos após {duration:.1f}s - Face {i}")
                self._closed_since[i] = None

            face.extra["eyes_closed"] = eyes_closed

        return faces