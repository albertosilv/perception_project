# src/perception/fatigue_detectors.py

from __future__ import annotations

import time
import numpy as np
from scipy.spatial import distance as dist
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)

MOUTH_IDX = list(range(48, 68))


def mouth_aspect_ratio(mouth_points: np.ndarray) -> float:
    """
    Calcula o MAR (Mouth Aspect Ratio).

    Quanto maior o MAR, maior a abertura vertical da boca.

    MAR ≈ 0
        Boca fechada.

    MAR elevado
        Boca mais aberta.

    O MAR é uma feature geométrica contínua.
    """

    a = dist.euclidean(
        mouth_points[13],
        mouth_points[19],
    )

    b = dist.euclidean(
        mouth_points[14],
        mouth_points[18],
    )

    c = dist.euclidean(
        mouth_points[15],
        mouth_points[17],
    )

    d = dist.euclidean(
        mouth_points[12],
        mouth_points[16],
    )

    if d == 0:
        return 0.0

    return (a + b + c) / (3.0 * d)


# =========================================================
# DETECTOR DE BOCEJO
# =========================================================

class YawnDetector(BaseDetector):
    """
    Detecta abertura prolongada da boca utilizando MAR.

    Responsabilidade:

        MAR → abertura da boca → duração → bocejo

    Este detector NÃO calcula frequência de bocejos.

    Ele apenas informa se existe atualmente um bocejo
    confirmado.

    O cálculo de bocejos/minuto é responsabilidade do
    PerclosDetector.

    Estados:

        yawn = False
            Não existe bocejo confirmado.

        yawn = True
            A boca permaneceu aberta acima do limiar
            durante tempo suficiente para caracterizar
            um bocejo.
    """

    def __init__(self, cfg: SimpleNamespace):

        # =====================================================
        # LIMIAR DO MAR
        # =====================================================

        self.mar_threshold = getattr(
            cfg,
            "mar_threshold",
            0.60,
        )

        # =====================================================
        # DURAÇÃO MÍNIMA
        # =====================================================

        self.min_duration_s = getattr(
            cfg,
            "min_duration_s",
            3.0,
        )

        # Momento em que a boca ultrapassou o threshold.
        self._open_since: dict[
            int,
            float | None
        ] = {}

    def detect(
        self,
        frame,
        faces: list[Face],
    ) -> list[Face]:

        now = time.time()

        for i, face in enumerate(faces):

            if face.landmarks is None:
                continue

            # =================================================
            # LANDMARKS DA BOCA
            # =================================================

            mouth = face.landmarks[MOUTH_IDX]

            # =================================================
            # FEATURE: MAR
            # =================================================

            mar = mouth_aspect_ratio(mouth)

            face.extra["mar"] = round(
                float(mar),
                3,
            )

            # =================================================
            # ESTADO DE BOCA ABERTA
            # =================================================

            mouth_open = (
                mar >= self.mar_threshold
            )

            face.extra["mouth_open"] = mouth_open

            # =================================================
            # CONTROLE TEMPORAL
            # =================================================

            open_since = self._open_since.get(i)

            yawn = False
            yawn_duration = 0.0

            if mouth_open:

                # Primeiro instante em que a boca abriu.
                if open_since is None:

                    self._open_since[i] = now
                    open_since = now

                # Tempo acumulado com a boca aberta.
                yawn_duration = (
                    now -
                    open_since
                )

                # =================================================
                # BOCEJO CONFIRMADO
                # =================================================

                if yawn_duration >= self.min_duration_s:

                    yawn = True

            else:

                # Boca fechou → reinicia o estado temporal.
                self._open_since[i] = None

            # =================================================
            # RESULTADOS
            # =================================================

            face.extra["yawn"] = yawn

            face.extra["yawn_duration_s"] = round(
                yawn_duration,
                2,
            )

        return faces