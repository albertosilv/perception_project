# src/perception/perclos_detector.py

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)


class PerclosDetector(BaseDetector):
    """
    Calcula métricas temporais relacionadas à fadiga.

    Responsabilidades:

        - PERCLOS
        - frequência de piscadas
        - frequência de bocejos

    Este detector NÃO calcula novamente o estado dos olhos.

    O EyeStateDetector é a única fonte responsável por determinar:

        face.extra["eye_closed"]

    Dessa forma existe uma única definição de "olho fechado"
    em todo o pipeline.

    Dependências:

        EyeStateDetector:
            face.extra["eye_closed"]
            face.extra["blinked"]

        YawnDetector:
            face.extra["yawn"]

    Todas as métricas temporais são calculadas utilizando
    timestamps através de time.time().

    Não são utilizados contadores de frames para determinar:

        - duração;
        - frequência;
        - janela temporal.
    """

    def __init__(self, cfg: SimpleNamespace):

        # ====================================================================
        # JANELA TEMPORAL
        # ====================================================================

        self.window_s = getattr(
            cfg,
            "window_s",
            60.0,
        )

        if self.window_s <= 0:
            raise ValueError(
                "window_s deve ser maior que zero."
            )

        # ====================================================================
        # PERCLOS
        # ====================================================================

        self.perclos_alert_threshold = getattr(
            cfg,
            "perclos_alert_threshold",
            0.08,
        )

        # ====================================================================
        # BLINK RATE
        # ====================================================================

        self.blink_rate_alert_threshold = getattr(
            cfg,
            "blink_rate_alert_threshold",
            30.0,
        )

        # ====================================================================
        # YAWN RATE
        # ====================================================================

        self.yawn_rate_alert_threshold = getattr(
            cfg,
            "yawn_rate_alert_threshold",
            3.0,
        )

        # ====================================================================
        # AMOSTRAS OCULARES
        # ====================================================================
        #
        # Cada elemento:
        #
        #     (timestamp, eye_closed)
        #
        # Exemplo:
        #
        #     (100.00, False)
        #     (100.03, False)
        #     (100.06, True)
        #
        # O estado da amostra representa o intervalo
        # até a próxima amostra.
        # ====================================================================

        self._eye_samples: dict[int, deque] = {}

        # ====================================================================
        # EVENTOS DE PISCADA
        # ====================================================================
        #
        # Cada elemento é um timestamp.
        #
        # Uma entrada representa uma piscada completa
        # confirmada pelo EyeStateDetector.
        # ====================================================================

        self._blink_events: dict[int, deque] = {}

        # ====================================================================
        # EVENTOS DE BOCEJO
        # ====================================================================

        self._yawn_events: dict[int, deque] = {}

        # ====================================================================
        # ESTADO ANTERIOR DO BOCEJO
        # ====================================================================
        #
        # O YawnDetector pode manter:
        #
        #     yawn = True
        #
        # durante vários frames.
        #
        # Aqui transformamos:
        #
        #     False -> True
        #
        # em um único evento.
        # ====================================================================

        self._yawn_was_active: dict[int, bool] = {}

        # ====================================================================
        # ESTADO DOS ALERTAS
        # ====================================================================

        self._last_perclos_alert: dict[int, bool] = {}
        self._last_blink_alert: dict[int, bool] = {}
        self._last_yawn_alert: dict[int, bool] = {}

    # ========================================================================
    # LIMPEZA DA JANELA TEMPORAL
    # ========================================================================

    def _trim(
        self,
        dq: deque,
        now: float,
    ) -> None:
        """
        Remove elementos anteriores à janela temporal.

        Exemplo:

            window_s = 60

        Se:

            now = 120

        qualquer evento anterior a:

            60 segundos

        é removido.
        """

        while dq:

            first = dq[0]

            if isinstance(first, tuple):
                timestamp = first[0]
            else:
                timestamp = first

            if now - timestamp <= self.window_s:
                break

            dq.popleft()

    # ========================================================================
    # CÁLCULO DO PERCLOS
    # ========================================================================

    def _calculate_perclos(
        self,
        samples: deque,
    ) -> tuple[float, float]:
        """
        Calcula PERCLOS utilizando TEMPO.

        Fórmula:

            PERCLOS =
                tempo com olhos fechados
                ------------------------
                    tempo observado

        Retorna:

            perclos:
                proporção entre 0 e 1.

            observed_time:
                tempo efetivamente observado em segundos.

        Importante:

            O cálculo não depende da quantidade de frames.
        """

        if len(samples) < 2:
            return 0.0, 0.0

        closed_time = 0.0
        observed_time = 0.0

        previous_timestamp, previous_closed = samples[0]

        for timestamp, closed in list(samples)[1:]:

            # ---------------------------------------------------------------
            # Intervalo temporal entre duas amostras.
            # ---------------------------------------------------------------

            dt = max(
                timestamp - previous_timestamp,
                0.0,
            )

            # ---------------------------------------------------------------
            # O estado anterior representa o intervalo.
            # ---------------------------------------------------------------

            if previous_closed:
                closed_time += dt

            observed_time += dt

            previous_timestamp = timestamp
            previous_closed = closed

        # ---------------------------------------------------------------
        # Segurança.
        # ---------------------------------------------------------------

        observed_time = min(
            observed_time,
            self.window_s,
        )

        closed_time = min(
            closed_time,
            observed_time,
        )

        if observed_time <= 0.0:
            return 0.0, 0.0

        perclos = (
            closed_time /
            observed_time
        )

        return perclos, observed_time

    # ========================================================================
    # DETECTOR
    # ========================================================================

    def detect(
        self,
        frame,
        faces: list[Face],
    ) -> list[Face]:

        # Timestamp atual.
        #
        # Todas as métricas temporais utilizam esse relógio.
        now = time.time()

        for i, face in enumerate(faces):

            # =================================================================
            # LANDMARKS
            # =================================================================

            if face.landmarks is None:
                continue

            # =================================================================
            # ESTRUTURAS DA FACE
            # =================================================================

            eye_samples = self._eye_samples.setdefault(
                i,
                deque(),
            )

            blink_events = self._blink_events.setdefault(
                i,
                deque(),
            )

            yawn_events = self._yawn_events.setdefault(
                i,
                deque(),
            )

            # =================================================================
            # ESTADO DOS OLHOS
            # =================================================================
            #
            # IMPORTANTE:
            #
            # Não calculamos EAR novamente.
            #
            # Não calculamos closed_ratio novamente.
            #
            # Não aplicamos outro threshold.
            #
            # A decisão pertence ao EyeStateDetector.
            # =================================================================

            eye_closed = face.extra.get(
                "eye_closed"
            )

            if eye_closed is not None:

                eye_samples.append(
                    (
                        now,
                        bool(eye_closed),
                    )
                )

            self._trim(
                eye_samples,
                now,
            )

            # =================================================================
            # PISCADAS
            # =================================================================
            #
            # blinked=True significa que uma piscada completa
            # acabou de ser confirmada.
            #
            # Cada True representa UM evento.
            # =================================================================

            if face.extra.get("blinked", False):

                blink_events.append(
                    now
                )

            self._trim(
                blink_events,
                now,
            )

            # =================================================================
            # BOCEJOS
            # =================================================================

            yawn_active = bool(
                face.extra.get(
                    "yawn",
                    False,
                )
            )

            was_active = self._yawn_was_active.get(
                i,
                False,
            )

            # ---------------------------------------------------------------
            # Apenas a transição False -> True gera um evento.
            # ---------------------------------------------------------------

            if yawn_active and not was_active:

                yawn_events.append(
                    now
                )

            self._yawn_was_active[i] = yawn_active

            self._trim(
                yawn_events,
                now,
            )

            # =================================================================
            # PERCLOS
            # =================================================================

            perclos, observed_time = (
                self._calculate_perclos(
                    eye_samples
                )
            )

            # =================================================================
            # TAXAS TEMPORAIS
            # =================================================================

            if observed_time > 0.0:

                observed_minutes = (
                    observed_time / 60.0
                )

                blink_rate = (
                    len(blink_events) /
                    observed_minutes
                )

                yawn_rate = (
                    len(yawn_events) /
                    observed_minutes
                )

            else:

                blink_rate = 0.0
                yawn_rate = 0.0

            # =================================================================
            # DISPONIBILIDADE DA MÉTRICA
            # =================================================================
            #
            # Uma janela completa possui:
            #
            #     window_s = 60 s
            #
            # Antes disso o PERCLOS é uma estimativa parcial.
            #
            # Mantemos o valor calculado, mas não geramos alerta
            # até existir tempo suficiente de observação.
            # =================================================================

            perclos_window_complete = (
                observed_time >= self.window_s
            )

            # =================================================================
            # ALERTAS
            # =================================================================

            perclos_alert = (
                perclos_window_complete
                and
                perclos >= self.perclos_alert_threshold
            )

            blink_alert = (
                blink_rate >=
                self.blink_rate_alert_threshold
            )

            yawn_alert = (
                yawn_rate >=
                self.yawn_rate_alert_threshold
            )

            # =================================================================
            # RESULTADOS
            # =================================================================

            face.extra["perclos"] = round(
                perclos,
                3,
            )

            face.extra["perclos_percent"] = round(
                perclos * 100.0,
                1,
            )

            face.extra["perclos_observed_time_s"] = round(
                observed_time,
                2,
            )

            face.extra["perclos_window_complete"] = (
                perclos_window_complete
            )

            face.extra["perclos_alert"] = (
                perclos_alert
            )

            face.extra["blink_rate_per_min"] = round(
                blink_rate,
                1,
            )

            face.extra["blink_rate_alert"] = (
                blink_alert
            )

            face.extra["yawn_rate_per_min"] = round(
                yawn_rate,
                1,
            )

            face.extra["yawn_rate_alert"] = (
                yawn_alert
            )

            # =================================================================
            # LOG PERCLOS
            # =================================================================

            last_perclos = self._last_perclos_alert.get(
                i,
                False,
            )

            if perclos_alert and not last_perclos:

                log.warning(
                    f"PERCLOS elevado - "
                    f"Face {i} | "
                    f"{perclos * 100:.1f}% do tempo "
                    f"com olhos fechados | "
                    f"janela: {observed_time:.1f}s | "
                    f"threshold: "
                    f"{self.perclos_alert_threshold * 100:.0f}%"
                )

            elif not perclos_alert and last_perclos:

                log.info(
                    f"PERCLOS normalizado - "
                    f"Face {i} | "
                    f"{perclos * 100:.1f}% do tempo "
                    f"com olhos fechados"
                )

            self._last_perclos_alert[i] = (
                perclos_alert
            )

            # =================================================================
            # LOG BLINK RATE
            # =================================================================

            last_blink = self._last_blink_alert.get(
                i,
                False,
            )

            if blink_alert and not last_blink:

                log.warning(
                    f"Frequência de piscadas elevada - "
                    f"Face {i} | "
                    f"{blink_rate:.1f} piscadas/min | "
                    f"threshold: "
                    f"{self.blink_rate_alert_threshold:.0f}"
                )

            elif not blink_alert and last_blink:

                log.info(
                    f"Frequência de piscadas normalizada - "
                    f"Face {i} | "
                    f"{blink_rate:.1f} piscadas/min"
                )

            self._last_blink_alert[i] = (
                blink_alert
            )

            # =================================================================
            # LOG YAWN RATE
            # =================================================================

            last_yawn = self._last_yawn_alert.get(
                i,
                False,
            )

            if yawn_alert and not last_yawn:

                log.warning(
                    f"Frequência de bocejos elevada - "
                    f"Face {i} | "
                    f"{yawn_rate:.1f} bocejos/min | "
                    f"threshold: "
                    f"{self.yawn_rate_alert_threshold:.0f}"
                )

            elif not yawn_alert and last_yawn:

                log.info(
                    f"Frequência de bocejos normalizada - "
                    f"Face {i} | "
                    f"{yawn_rate:.1f} bocejos/min"
                )

            self._last_yawn_alert[i] = (
                yawn_alert
            )

        return faces