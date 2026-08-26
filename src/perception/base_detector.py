"""
Interface base que todo detector de percepção deve seguir.

Ao criar uma nova feature (ex: detecção de emoção, reconhecimento facial,
estimativa de pose), herde de BaseDetector para manter o pipeline
plugável e consistente.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list:
        """
        Executa a detecção sobre um frame BGR (formato OpenCV).

        Deve retornar uma lista de resultados (ex: list[Face]).
        """
        raise NotImplementedError
