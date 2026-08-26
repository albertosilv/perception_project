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
    Calcula PERCLOS em uma janela deslizante.

    PERCLOS representa a proporção de TEMPO em que os olhos foram
    classificados como fechados dentro da janela de observação.

    O estado de olho fechado é obtido a partir de:
        closed_ratio = (0.25 - EAR) / 0.15

    Com closed_ratio_threshold = 0.6:
        EAR ~= 0.16

    Importante:
        - PERCLOS não é frequência de piscadas.
        - PERCLOS não é bocejo.
        - PERCLOS é calculado por tempo, e não por quantidade de frames.
        - Blink rate e yawn rate são indicadores complementares.

    Depende de rodar depois de:
        - ClosedEyesDetector
        - BlinkDetector
        - YawnDetector

    Esses detectores devem preencher:
        face.extra["eye_closed_ratio"]
        face.extra["blinked"]
        face.extra["yawn"]
    """

    def __init__(self, cfg: SimpleNamespace):
        self.window_s = getattr(cfg, "window_s", 60.0)

        # Razão utilizada para classificar o olho como fechado.
        #
        # closed_ratio = (0.25 - EAR) / 0.15
        #
        # threshold = 0.6  -> EAR ~= 0.16
        self.closed_ratio_threshold = getattr(
            cfg,
            "closed_ratio_threshold",
            0.6,
        )

        # Alerta de PERCLOS.
        # 0.08 = 8% do tempo com os olhos fechados.
        self.perclos_alert_threshold = getattr(
            cfg,
            "perclos_alert_threshold",
            0.08,
        )

        # Frequência de piscadas por minuto.
        self.blink_rate_alert_threshold = getattr(
            cfg,
            "blink_rate_alert_threshold",
            30.0,
        )

        # Frequência de bocejos por minuto.
        self.yawn_rate_alert_threshold = getattr(
            cfg,
            "yawn_rate_alert_threshold",
            3.0,
        )

        # ------------------------------------------------------------------
        # Amostras oculares.
        #
        # Cada item:
        #   (timestamp, closed)
        #
        # O timestamp permite calcular PERCLOS utilizando TEMPO.
        # ------------------------------------------------------------------
        self._eye_samples: dict[int, deque] = {}

        # Eventos discretos de piscada.
        self._blink_events: dict[int, deque] = {}

        # Eventos discretos de bocejo.
        self._yawn_events: dict[int, deque] = {}

        # Estado anterior do bocejo.
        #
        # Usado para transformar:
        #   False -> True
        #
        # em apenas um evento.
        self._yawn_was_active: dict[int, bool] = {}

        # Controle para evitar logs repetidos
        self._last_perclos_alert: dict[int, bool] = {}
        self._last_blink_alert: dict[int, bool] = {}
        self._last_yawn_alert: dict[int, bool] = {}

    def _trim(self, dq: deque, now: float) -> None:
        """
        Remove eventos/amostras que ficaram fora da janela temporal.
        """
        while dq:
            item = dq[0]

            timestamp = item[0] if isinstance(item, tuple) else item

            if now - timestamp <= self.window_s:
                break

            dq.popleft()

    def _calculate_perclos(
        self,
        samples: deque,
        now: float,
    ) -> tuple[float, float]:
        """
        Calcula PERCLOS utilizando tempo, e não número de frames.

        Retorna:
            (perclos, observed_time)

        PERCLOS:

            tempo_com_olho_fechado
            ----------------------
                tempo_observado
        """

        if len(samples) < 2:
            return 0.0, 0.0

        closed_time = 0.0
        observed_time = 0.0

        previous_timestamp, previous_closed = samples[0]

        for timestamp, closed in list(samples)[1:]:
            dt = max(timestamp - previous_timestamp, 0.0)

            # A amostra anterior representa o estado durante o intervalo
            # até a próxima amostra.
            if previous_closed:
                closed_time += dt

            observed_time += dt

            previous_timestamp = timestamp
            previous_closed = closed

        # Limita o tempo observado à janela configurada.
        observed_time = min(observed_time, self.window_s)
        closed_time = min(closed_time, observed_time)

        if observed_time <= 0.0:
            return 0.0, 0.0

        perclos = closed_time / observed_time

        return perclos, observed_time

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        now = time.time()

        for i, face in enumerate(faces):

            if face.landmarks is None:
                continue

            eye_samples = self._eye_samples.setdefault(i, deque())
            blink_events = self._blink_events.setdefault(i, deque())
            yawn_events = self._yawn_events.setdefault(i, deque())

            # ==============================================================
            # OLHOS FECHADOS
            # ==============================================================

            closed_ratio = face.extra.get("eye_closed_ratio")

            if closed_ratio is not None:
                is_closed = (
                    closed_ratio >= self.closed_ratio_threshold
                )

                eye_samples.append(
                    (now, is_closed)
                )

            self._trim(eye_samples, now)

            # ==============================================================
            # PISCADAS
            # ==============================================================

            if face.extra.get("blinked"):
                blink_events.append(now)

            self._trim(blink_events, now)

            # ==============================================================
            # BOCEJO
            #
            # yawn=True pode permanecer ativo durante vários frames.
            #
            # Contamos somente:
            #
            # False -> True
            #
            # como um novo bocejo.
            # ==============================================================

            yawn_active = bool(
                face.extra.get("yawn")
            )

            was_active = self._yawn_was_active.get(
                i,
                False,
            )

            if yawn_active and not was_active:
                yawn_events.append(now)

            self._yawn_was_active[i] = yawn_active

            self._trim(yawn_events, now)

            # ==============================================================
            # PERCLOS
            # ==============================================================

            perclos, observed_time = self._calculate_perclos(
                eye_samples,
                now,
            )

            # ==============================================================
            # TAXAS POR MINUTO
            # ==============================================================

            if observed_time > 0.0:
                span_min = observed_time / 60.0
            else:
                span_min = 0.0

            if span_min > 0.0:
                blink_rate = len(blink_events) / span_min
                yawn_rate = len(yawn_events) / span_min
            else:
                blink_rate = 0.0
                yawn_rate = 0.0

            # ==============================================================
            # RESULTADOS
            # ==============================================================

            face.extra["perclos"] = round(
                perclos,
                3,
            )

            face.extra["perclos_percent"] = round(
                perclos * 100.0,
                1,
            )

            perclos_alert = perclos >= self.perclos_alert_threshold
            face.extra["perclos_alert"] = perclos_alert

            face.extra["perclos_observed_time_s"] = round(
                observed_time,
                2,
            )

            face.extra["blink_rate_per_min"] = round(
                blink_rate,
                1,
            )

            blink_alert = blink_rate >= self.blink_rate_alert_threshold
            face.extra["blink_rate_alert"] = blink_alert

            face.extra["yawn_rate_per_min"] = round(
                yawn_rate,
                1,
            )

            yawn_alert = yawn_rate >= self.yawn_rate_alert_threshold
            face.extra["yawn_rate_alert"] = yawn_alert

            # ==============================================================
            # LOGS DE ALERTA (apenas quando o estado muda)
            # ==============================================================

            # PERCLOS
            last_perclos = self._last_perclos_alert.get(i, False)
            if perclos_alert and not last_perclos:
                log.warning(
                    f"PERCLOS elevado - Face {i} | "
                    f"{perclos*100:.1f}% do tempo com olhos fechados "
                    f"(threshold: {self.perclos_alert_threshold*100:.0f}%)"
                )
            elif not perclos_alert and last_perclos:
                log.info(
                    f"PERCLOS normalizado - Face {i} | "
                    f"{perclos*100:.1f}% do tempo com olhos fechados"
                )
            self._last_perclos_alert[i] = perclos_alert

            # BLINK RATE
            last_blink = self._last_blink_alert.get(i, False)
            if blink_alert and not last_blink:
                log.warning(
                    f"Frequência de piscadas elevada - Face {i} | "
                    f"{blink_rate:.1f} piscadas/min "
                    f"(threshold: {self.blink_rate_alert_threshold:.0f})"
                )
            elif not blink_alert and last_blink:
                log.info(
                    f"Frequência de piscadas normalizada - Face {i} | "
                    f"{blink_rate:.1f} piscadas/min"
                )
            self._last_blink_alert[i] = blink_alert

            # YAWN RATE
            last_yawn = self._last_yawn_alert.get(i, False)
            if yawn_alert and not last_yawn:
                log.warning(
                    f"Frequência de bocejos elevada - Face {i} | "
                    f"{yawn_rate:.1f} bocejos/min "
                    f"(threshold: {self.yawn_rate_alert_threshold:.0f})"
                )
            elif not yawn_alert and last_yawn:
                log.info(
                    f"Frequência de bocejos normalizada - Face {i} | "
                    f"{yawn_rate:.1f} bocejos/min"
                )
            self._last_yawn_alert[i] = yawn_alert

        return faces