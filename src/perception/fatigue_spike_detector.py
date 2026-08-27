"""
Detecta PICOS de fadiga ao longo de uma janela de tempo `x`, combinando:

  - a probabilidade contínua de "fadiga" já gerada pela regressão
    LOGÍSTICA (DrowsinessClassifier) a cada frame;
  - uma regressão LINEAR simples ajustada sobre essa série temporal,
    dentro da janela de `window_seconds`, para estimar a TENDÊNCIA
    (subindo, descendo, estável).

Um "pico" não é apenas "a probabilidade está alta" — é quando o valor
atual sobe muito acima do que a tendência da janela prevê (resíduo alto),
o que indica uma alteração súbita, e não uma fadiga já crônica/estável.

Depende de `face.extra["drowsiness_probabilities"]`, preenchido pelo
DrowsinessClassifier — por isso deve rodar depois dele no pipeline.
"""

from __future__ import annotations
import csv
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)


class FatigueSpikeDetector(BaseDetector):
    def __init__(self, cfg: SimpleNamespace):
        self.target_label = cfg.target_label
        self.window_seconds = cfg.window_seconds
        self.spike_threshold = cfg.spike_threshold
        self.min_confidence = cfg.min_confidence
        self.cooldown_seconds = cfg.cooldown_seconds

        self.log_events = getattr(cfg, "log_events", True)
        self.events_log_path = Path(getattr(cfg, "events_log_file", "data/output/fatigue_spikes.csv"))

        # histórico (timestamp, probabilidade) por rosto, só o necessário
        # para cobrir a janela de tempo configurada
        self._history: dict[int, deque[tuple[float, float]]] = {}
        self._last_spike_time: dict[int, float] = {}

        # todos os picos detectados na sessão, para salvar em CSV ao final
        self._events: list[dict] = []

        log.info(
            f"FatigueSpikeDetector inicializado | janela={self.window_seconds}s | "
            f"threshold de resíduo={self.spike_threshold} | classe-alvo='{self.target_label}'"
        )

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        now = time.time()

        for i, face in enumerate(faces):
            probabilities = face.extra.get("drowsiness_probabilities")
            if probabilities is None or self.target_label not in probabilities:
                continue

            value = probabilities[self.target_label]

            history = self._history.setdefault(i, deque())
            history.append((now, value))
            self._trim_window(history, now)

            slope, trend_prediction = self._fit_trend(history, now)
            residual = value - trend_prediction if trend_prediction is not None else 0.0

            face.extra["fatigue_value"] = round(value, 3)
            face.extra["fatigue_trend_slope"] = round(slope, 4) if slope is not None else None
            face.extra["fatigue_spike"] = False

            is_spike = (
                trend_prediction is not None
                and residual >= self.spike_threshold
                and value >= self.min_confidence
                and (now - self._last_spike_time.get(i, 0.0)) >= self.cooldown_seconds
            )

            if is_spike:
                face.extra["fatigue_spike"] = True
                self._last_spike_time[i] = now
                self._events.append({
                    "timestamp": now,
                    "face_index": i,
                    "value": round(value, 3),
                    "trend_prediction": round(trend_prediction, 3),
                    "residual": round(residual, 3),
                })
                log.info(
                    f"Pico de fadiga detectado | rosto={i} | valor={value:.2f} | "
                    f"tendência esperada={trend_prediction:.2f} | resíduo={residual:.2f}"
                )

            # nº de picos dentro da janela atual (útil para exibir/alertar
            # "N picos nos últimos X segundos")
            face.extra["fatigue_spikes_in_window"] = sum(
                1 for e in self._events
                if e["face_index"] == i and (now - e["timestamp"]) <= self.window_seconds
            )

        return faces

    def _trim_window(self, history: deque, now: float) -> None:
        """Remove do histórico tudo que já saiu da janela de `window_seconds`."""
        while history and (now - history[0][0]) > self.window_seconds:
            history.popleft()

    def _fit_trend(self, history: deque, now: float) -> tuple[float | None, float | None]:
        """
        Ajusta uma regressão linear (grau 1) sobre os pontos da janela e
        retorna (slope, valor previsto para "agora").

        Precisa de pelo menos 3 pontos para a tendência fazer sentido —
        com menos que isso, não há como distinguir pico de ruído.
        """
        if len(history) < 3:
            return None, None

        timestamps = np.array([t for t, _ in history], dtype=np.float64)
        values = np.array([v for _, v in history], dtype=np.float64)

        # tempo relativo ao início da janela, para estabilidade numérica
        t_relative = timestamps - timestamps[0]
        slope, intercept = np.polyfit(t_relative, values, deg=1)

        t_now_relative = now - timestamps[0]
        prediction = slope * t_now_relative + intercept
        return float(slope), float(prediction)

    def save_events(self) -> None:
        """
        Salva todos os picos detectados na sessão em CSV, na pasta
        definida no config. Chamado no `finally` do main.py, garantindo
        que os eventos sejam persistidos mesmo se o programa for
        encerrado abruptamente (Ctrl+C, fechar a janela, erro).
        """
        if not self.log_events:
            return

        self.events_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.events_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["timestamp", "face_index", "value", "trend_prediction", "residual"]
            )
            writer.writeheader()
            writer.writerows(self._events)

        log.info(
            f"{len(self._events)} picos de fadiga salvos em: "
            f"{self.events_log_path.resolve()}"
        )
