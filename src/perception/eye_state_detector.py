# src/perception/fatigue_detectors.py

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial import distance as dist

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)


# ============================================================================
# ÍNDICES DOS LANDMARKS
# Modelo facial de 68 pontos do dlib
# ============================================================================

RIGHT_EYE_IDX = list(range(36, 42))
LEFT_EYE_IDX = list(range(42, 48))
MOUTH_IDX = list(range(48, 68))


# ============================================================================
# FEATURES OCULARES
# ============================================================================

def eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Calcula o Eye Aspect Ratio (EAR).

    EAR relaciona a altura do olho com sua largura.

    EAR alto:
        → olho mais aberto

    EAR baixo:
        → olho mais fechado
    """

    a = dist.euclidean(
        eye_points[1],
        eye_points[5],
    )

    b = dist.euclidean(
        eye_points[2],
        eye_points[4],
    )

    c = dist.euclidean(
        eye_points[0],
        eye_points[3],
    )

    # Evita divisão por zero.
    if c == 0:
        return 0.0

    return (a + b) / (2.0 * c)


def eye_closed_ratio(
    eye_points: np.ndarray,
    open_ear: float = 0.25,
    closed_ear: float = 0.10,
) -> float:
    """
    Converte EAR em uma escala normalizada de fechamento.

    Retorno:

        0.0 → olho aberto
        1.0 → olho muito fechado

    Fórmula:

        closed_ratio =
            (open_ear - EAR)
            -----------------
            (open_ear - closed_ear)

    O resultado é limitado ao intervalo [0, 1].
    """

    if open_ear <= closed_ear:
        raise ValueError(
            "open_ear deve ser maior que closed_ear."
        )

    ear = eye_aspect_ratio(eye_points)

    ratio = (
        open_ear - ear
    ) / (
        open_ear - closed_ear
    )

    return float(
        np.clip(ratio, 0.0, 1.0)
    )


# ============================================================================
# DETECTOR OCULAR
# ============================================================================

class EyeStateDetector(BaseDetector):
    """
    Detector de estado e eventos oculares.

    Responsabilidades:

    1. Calcular EAR.
    2. Calcular eye_closed_ratio.
    3. Determinar se os olhos estão fechados.
    4. Medir a duração do fechamento.
    5. Classificar o fechamento como:
       - fechamento curto/ruído;
       - piscada;
       - fechamento ocular prolongado.

    Todas as métricas temporais são calculadas utilizando
    timestamps (time.time()), e NÃO quantidade de frames.

    O detector não determina diretamente fadiga.
    Ele fornece features e eventos para os demais componentes.
    """

    def __init__(self, cfg: SimpleNamespace):

        # ====================================================================
        # NORMALIZAÇÃO DO EAR
        # ====================================================================

        self.open_ear = getattr(
            cfg,
            "open_ear",
            0.25,
        )

        self.closed_ear = getattr(
            cfg,
            "closed_ear",
            0.10,
        )

        if self.open_ear <= self.closed_ear:
            raise ValueError(
                "open_ear deve ser maior que closed_ear."
            )

        # ====================================================================
        # LIMIAR DE OLHO FECHADO
        # ====================================================================

        self.closed_ratio_threshold = getattr(
            cfg,
            "closed_ratio_threshold",
            0.60,
        )

        # ====================================================================
        # PISCADA
        # ====================================================================

        self.min_blink_duration_s = getattr(
            cfg,
            "min_blink_duration_s",
            0.20,
        )

        # ====================================================================
        # FECHAMENTO OCULAR PROLONGADO
        # ====================================================================

        self.prolonged_closure_duration_s = getattr(
            cfg,
            "prolonged_closure_duration_s",
            3.0,
        )

        # ====================================================================
        # ESTADO TEMPORAL
        # ====================================================================

        # Face → timestamp do início do fechamento.
        self._closed_since: dict[int, float | None] = {}

        # Face → indica se o fechamento prolongado já foi registrado.
        #
        # Isso evita gerar o mesmo log a cada frame durante
        # vários segundos de fechamento.
        self._prolonged_reported: dict[int, bool] = {}

    def detect(
        self,
        frame,
        faces: list[Face],
    ) -> list[Face]:

        # Timestamp atual.
        #
        # Todas as durações serão calculadas através da diferença
        # entre timestamps.
        now = time.time()

        for i, face in enumerate(faces):

            # =================================================================
            # LANDMARKS
            # =================================================================

            if face.landmarks is None:
                continue

            # =================================================================
            # SELEÇÃO DOS OLHOS
            # =================================================================

            right_eye = face.landmarks[
                RIGHT_EYE_IDX
            ]

            left_eye = face.landmarks[
                LEFT_EYE_IDX
            ]

            # =================================================================
            # EAR
            # =================================================================

            right_ear = eye_aspect_ratio(
                right_eye
            )

            left_ear = eye_aspect_ratio(
                left_eye
            )

            ear = (
                right_ear +
                left_ear
            ) / 2.0

            # =================================================================
            # CLOSED RATIO
            # =================================================================

            right_closed_ratio = eye_closed_ratio(
                right_eye,
                self.open_ear,
                self.closed_ear,
            )

            left_closed_ratio = eye_closed_ratio(
                left_eye,
                self.open_ear,
                self.closed_ear,
            )

            closed_ratio = (
                right_closed_ratio +
                left_closed_ratio
            ) / 2.0

            # =================================================================
            # SALVA FEATURES
            # =================================================================

            face.extra["ear"] = round(
                float(ear),
                3,
            )

            face.extra["right_ear"] = round(
                float(right_ear),
                3,
            )

            face.extra["left_ear"] = round(
                float(left_ear),
                3,
            )

            face.extra["eye_closed_ratio"] = round(
                float(closed_ratio),
                3,
            )

            # =================================================================
            # ESTADO DO FRAME
            # =================================================================

            blinked = False
            prolonged_closure = False

            # ================================================================
            # DETERMINAÇÃO DO ESTADO OCULAR
            # ================================================================

            eye_closed = (
                closed_ratio >=
                self.closed_ratio_threshold
            )

            # Timestamp do início do fechamento atual.
            closed_since = self._closed_since.get(i)

            # =================================================================
            # OLHO FECHADO
            # =================================================================

            if eye_closed:

                # -------------------------------------------------------------
                # INÍCIO DO FECHAMENTO
                # -------------------------------------------------------------

                if closed_since is None:

                    self._closed_since[i] = now

                    self._prolonged_reported[i] = False

                # -------------------------------------------------------------
                # FECHAMENTO CONTINUA
                # -------------------------------------------------------------

                else:

                    duration = (
                        now -
                        closed_since
                    )

                    # ---------------------------------------------------------
                    # FECHAMENTO PROLONGADO
                    # ---------------------------------------------------------

                    if (
                        duration >=
                        self.prolonged_closure_duration_s
                    ):

                        prolonged_closure = True

                        # Log apenas uma vez por fechamento.
                        if not self._prolonged_reported.get(
                            i,
                            False,
                        ):

                            log.warning(
                                f"Fechamento ocular prolongado - "
                                f"Face {i} | "
                                f"Duração: {duration:.2f}s"
                            )

                            self._prolonged_reported[i] = True

            # =================================================================
            # OLHO ABERTO
            # =================================================================

            else:

                # -------------------------------------------------------------
                # EXISTIA UM FECHAMENTO
                # -------------------------------------------------------------

                if closed_since is not None:

                    duration = (
                        now -
                        closed_since
                    )

                    # =========================================================
                    # PISCADA
                    # =========================================================
                    #
                    # A piscada só é confirmada quando o olho REABRE.
                    #
                    # Isso garante que sabemos a duração total do fechamento.
                    #
                    # Exemplo:
                    #
                    # 0.05s → não é piscada
                    # 0.20s → piscada
                    # 1.00s → piscada
                    # 3.00s → fechamento prolongado
                    # 5.00s → fechamento prolongado
                    #
                    # =========================================================

                    if (
                        self.min_blink_duration_s
                        <= duration
                        < self.prolonged_closure_duration_s
                    ):

                        blinked = True

                        log.info(
                            f"Piscada detectada - "
                            f"Face {i} | "
                            f"Duração: {duration:.3f}s"
                        )

                    # =========================================================
                    # REABERTURA APÓS FECHAMENTO PROLONGADO
                    # =========================================================

                    elif (
                        duration >=
                        self.prolonged_closure_duration_s
                    ):

                        log.info(
                            f"Olhos reabertos - "
                            f"Face {i} | "
                            f"Duração do fechamento: "
                            f"{duration:.2f}s"
                        )

                # -------------------------------------------------------------
                # RESET DO FECHAMENTO
                # -------------------------------------------------------------

                self._closed_since[i] = None
                self._prolonged_reported[i] = False

            # =================================================================
            # RESULTADOS
            # =================================================================

            face.extra["eye_closed"] = eye_closed

            face.extra["blinked"] = blinked

            face.extra["prolonged_closure"] = (
                prolonged_closure
            )

        return faces