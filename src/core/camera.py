"""
Abstração da fonte de vídeo (webcam, arquivo de vídeo ou RTSP).

Mantém o resto do pipeline agnóstico em relação à origem do frame,
facilitando trocar de webcam para vídeo gravado ou câmera IP.
"""

from __future__ import annotations
import cv2
import numpy as np
from types import SimpleNamespace


class Camera:
    def __init__(self, cfg: SimpleNamespace):
        self.source = cfg.source
        self.width = cfg.width
        self.height = cfg.height
        self.flip_horizontal = cfg.flip_horizontal
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo: {self.source}"
            )

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is None:
            raise RuntimeError("Câmera não foi aberta. Chame .open() antes de .read().")

        ok, frame = self._cap.read()
        if not ok:
            return False, None

        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # Permite uso como context manager: `with Camera(cfg) as cam:`
    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
