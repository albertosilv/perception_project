"""
Funções auxiliares para desenhar os resultados da percepção sobre o frame.

Mantido separado do pipeline para que a lógica de detecção não se misture
com a lógica de "como exibir isso na tela".
"""

from __future__ import annotations
import cv2
import numpy as np
from types import SimpleNamespace

from src.perception.types import FrameResult

COLOR_BBOX = (0, 255, 0)
COLOR_LANDMARK = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)


def draw_result(frame: np.ndarray, result: FrameResult, cfg: SimpleNamespace) -> np.ndarray:
    """Desenha bounding boxes, landmarks e FPS sobre uma cópia do frame."""
    output = frame.copy()

    for face in result.faces:
        if cfg.show_bbox:
            x, y, w, h = face.bbox.as_tuple()
            cv2.rectangle(output, (x, y), (x + w, y + h), COLOR_BBOX, 2)

        if face.landmarks is not None:
            for idx, (px, py) in enumerate(face.landmarks):
                cv2.circle(output, (px, py), 1, COLOR_LANDMARK, -1)

        # --- Classificação (regressão logística), se habilitada ---
        if "drowsiness_label" in face.extra:
            x, y, w, h = face.bbox.as_tuple()
            label = face.extra["drowsiness_label"]
            confidence = face.extra["drowsiness_confidence"]
            cv2.putText(
                output,
                f"{label} ({confidence:.0%})",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                (0, 255, 255),
                2,
            )

        # --- Pico de fadiga (regressão linear sobre a janela), se habilitado ---
        if "fatigue_value" in face.extra:
            x, y, w, h = face.bbox.as_tuple()
            slope = face.extra.get("fatigue_trend_slope")
            n_spikes = face.extra.get("fatigue_spikes_in_window", 0)
            trend_arrow = "↑" if (slope or 0) > 0 else "↓" if (slope or 0) < 0 else "→"
            cv2.putText(
                output,
                f"fadiga: {face.extra['fatigue_value']:.2f} {trend_arrow}  picos/janela: {n_spikes}",
                (x, y + h + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                (255, 0, 255),
                2,
            )
            if face.extra.get("fatigue_spike"):
                cv2.putText(
                    output,
                    "!!! PICO DE FADIGA !!!",
                    (x, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cfg.font_scale + 0.1,
                    (0, 0, 255),
                    2,
                )

    if cfg.show_fps:
        cv2.putText(
            output,
            f"FPS: {result.fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            cfg.font_scale,
            COLOR_TEXT,
            2,
        )

    cv2.putText(
        output,
        f"Rostos: {len(result.faces)}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        cfg.font_scale,
        COLOR_TEXT,
        2,
    )

    return output
