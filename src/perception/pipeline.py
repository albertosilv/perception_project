"""
Orquestra os detectores de percepção em sequência sobre um frame.

Este é o ponto central para adicionar novas features: crie seu detector
(herdando de BaseDetector), instancie aqui e adicione uma etapa em
`process()`.
"""

from __future__ import annotations
import time
import numpy as np
from types import SimpleNamespace

from src.perception.face_detector import FaceDetector
from src.perception.landmark_detector import LandmarkDetector
from src.perception.types import FrameResult
from src.perception.eye_state_detector import EyeStateDetector
from src.perception.head_pose_detector import HeadPoseDetector
from src.core.logger import get_logger
from src.perception.yawn_detector import YawnDetector
from src.perception.perclos_detector import PerclosDetector
from src.perception.drowsiness_classifier import DrowsinessClassifier
from src.perception.fatigue_spike_detector import FatigueSpikeDetector

log = get_logger(__name__)


class PerceptionPipeline:
    def __init__(self, cfg: SimpleNamespace):
        self.cfg = cfg
        self.face_detector = FaceDetector(cfg.face_detection)

        self.landmarks_enabled = cfg.landmarks.enabled
        self.landmark_detector = (
            LandmarkDetector(cfg.landmarks) if self.landmarks_enabled else None
        )

        self._last_time = time.time()

        self.eye_state_enabled = getattr(cfg, "eye_state", None) is not None and cfg.eye_state.enabled
        self.eye_state_detector = EyeStateDetector(cfg.eye_state) if self.eye_state_enabled else None

        self.head_pose_enabled = getattr(cfg, "head_pose", None) is not None and cfg.head_pose.enabled
        self.head_pose_detector = HeadPoseDetector(cfg.head_pose) if self.head_pose_enabled else None

        self.yawn_enabled = getattr(cfg, "yawn", None) is not None and cfg.yawn.enabled
        self.yawn_detector = YawnDetector(cfg.yawn) if self.yawn_enabled else None

        # PERCLOS depende de face.extra["eye_closed"], preenchido pelo
        # EyeStateDetector, e de face.extra["yawn"], preenchido pelo
        # YawnDetector — por isso roda por último no process().
        self.perclos_enabled = getattr(cfg, "perclos", None) is not None and cfg.perclos.enabled
        self.perclos_detector = PerclosDetector(cfg.perclos) if self.perclos_enabled else None

        if self.perclos_enabled and not self.eye_state_enabled:
            log.warning(
                "PERCLOS habilitado sem eye_state habilitado — "
                "'perclos' e 'blink_rate_per_min' ficarão sempre em 0, "
                "pois dependem de face.extra['eye_closed']/['blinked']."
            )

        # O classificador depende das features geradas pelos detectores
        # acima (EAR, MAR, PERCLOS, pose de cabeça) — por isso roda por
        # último. Só existe depois de rodar `python -m src.ml.train_model`.
        self.classifier_enabled = getattr(cfg, "classifier", None) is not None and cfg.classifier.enabled
        self.classifier = DrowsinessClassifier(cfg.classifier) if self.classifier_enabled else None

        # Depende da probabilidade de "fadiga" gerada pelo classifier —
        # roda por último e só faz sentido com o classifier habilitado.
        self.fatigue_spike_enabled = getattr(cfg, "fatigue_spike", None) is not None and cfg.fatigue_spike.enabled
        self.fatigue_spike_detector = (
            FatigueSpikeDetector(cfg.fatigue_spike) if self.fatigue_spike_enabled else None
        )

        if self.fatigue_spike_enabled and not self.classifier_enabled:
            log.warning(
                "fatigue_spike habilitado sem classifier habilitado — "
                "nenhum pico será detectado, pois depende de "
                "face.extra['drowsiness_probabilities']."
            )

    def process(self, frame: np.ndarray) -> FrameResult:
        faces = self.face_detector.detect(frame)

        if self.landmarks_enabled and faces:
            faces = self.landmark_detector.detect(frame, faces)

            if self.eye_state_enabled:
                faces = self.eye_state_detector.detect(frame, faces)
            if self.head_pose_enabled:
                faces = self.head_pose_detector.detect(frame, faces)
            if self.yawn_enabled:
                faces = self.yawn_detector.detect(frame, faces)
            if self.perclos_enabled:
                faces = self.perclos_detector.detect(frame, faces)
            if self.classifier_enabled:
                faces = self.classifier.detect(frame, faces)
            if self.fatigue_spike_enabled:
                faces = self.fatigue_spike_detector.detect(frame, faces)

        fps = self._compute_fps()
        return FrameResult(faces=faces, fps=fps)

    def finalize(self) -> None:
        """
        Chamado ao final da execução (main.py) para persistir qualquer
        estado acumulado durante a sessão — hoje, os picos de fadiga
        detectados. Seguro de chamar mesmo se a feature estiver desligada.
        """
        if self.fatigue_spike_enabled:
            self.fatigue_spike_detector.save_events()

    def _compute_fps(self) -> float:
        now = time.time()
        elapsed = now - self._last_time
        self._last_time = now
        return 1.0 / elapsed if elapsed > 0 else 0.0
