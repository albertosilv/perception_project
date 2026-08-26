"""
Estruturas de dados compartilhadas entre os módulos de percepção.

Centralizar isso aqui evita que cada novo detector invente seu próprio
formato de retorno — todo mundo fala a mesma "língua".
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class Face:
    """Representa um rosto detectado em um frame."""
    bbox: BoundingBox
    landmarks: np.ndarray | None = None   # shape (68, 2) se landmarks habilitados
    confidence: float = 1.0
    # Espaço livre para suas features futuras, ex: emotion, identity, gaze...
    extra: dict = field(default_factory=dict)


@dataclass
class FrameResult:
    """Resultado da percepção sobre um único frame."""
    faces: list[Face] = field(default_factory=list)
    fps: float = 0.0
