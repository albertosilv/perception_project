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

MOUTH_IDX = list(range(48, 68))


def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """
    Calcula a MAR (Mouth Aspect Ratio) para detecção de bocejos.
    Quanto maior o valor, mais aberta a boca.
    
    Args:
        mouth_points: Array com os pontos da boca
        
    Returns:
        float: MAR (0 = boca fechada, >0.6 = boca aberta/bocejo)
    """
    a = dist.euclidean(mouth_points[13], mouth_points[19])
    b = dist.euclidean(mouth_points[14], mouth_points[18])
    c = dist.euclidean(mouth_points[15], mouth_points[17])
    d = dist.euclidean(mouth_points[12], mouth_points[16])
    
    if d == 0:
        return 0
    
    return (a + b + c) / (3.0 * d)





# ============================================================================
# DETECTOR DE BOCEJO
# ============================================================================

class YawnDetector(BaseDetector):
    """
    Detector de bocejos usando MAR (Mouth Aspect Ratio).
    Detecta quando a pessoa abre a boca (bocejo/fala).
    """
    
    def __init__(self, cfg: SimpleNamespace):
        self.mar_threshold = getattr(cfg, "mar_threshold", 0.6)
        # Duração mínima (segundos) com a boca aberta acima do MAR para
        # confirmar bocejo — medida por relógio, não por nº de frames.
        self.min_duration_s = getattr(cfg, "min_duration_s", 3.0)
        self._open_since: dict[int, float | None] = {}

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        now = time.time()
        for i, face in enumerate(faces):
            if face.landmarks is None:
                continue

            mouth = face.landmarks[MOUTH_IDX]
            mar = mouth_aspect_ratio(mouth)
            face.extra["mar"] = round(float(mar), 3)

            open_since = self._open_since.get(i)
            yawn = False
            if mar > self.mar_threshold:
                if open_since is None:
                    open_since = now
                    self._open_since[i] = now
                duration = now - open_since
                if duration >= self.min_duration_s:
                    yawn = True
                    log.info(f"Bocejo detectado - Face {i} ({duration:.1f}s)")
            else:
                self._open_since[i] = None

            face.extra["yawn"] = yawn

        return faces


