"""
Classifica o estado do usuário (ex: "fadiga"/"sem_fadiga") em tempo real,
usando o modelo de Regressão Logística treinado por src/ml/train_model.py.

Depende das features geradas pelos outros detectores do pipeline
(EyeStateDetector, YawnDetector, HeadPoseDetector, PerclosDetector) —
por isso deve rodar por último em PerceptionPipeline.process().
"""

from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.preprocessing.feature_extractor import FeatureExtractor
from src.core.logger import get_logger

log = get_logger(__name__)


class DrowsinessClassifier(BaseDetector):
    def __init__(self, cfg: SimpleNamespace):
        model_path = Path(cfg.model_path)
        scaler_path = Path(cfg.scaler_path)
        metadata_path = model_path.with_suffix(".json")

        for path, description in [
            (model_path, "modelo"),
            (scaler_path, "scaler"),
            (metadata_path, "metadados"),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Arquivo de {description} não encontrado: {path.resolve()}\n"
                    "Rode 'python -m src.ml.train_model' antes de habilitar "
                    "'classifier.enabled' no config.yaml."
                )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Usa a MESMA ordem de features salva no treino — nunca a do
        # config.yaml diretamente, para não haver risco de divergência
        # caso alguém edite o config depois de treinar o modelo.
        self.feature_extractor = FeatureExtractor(metadata["feature_names"])
        self.classes = metadata["classes"]

        self.decision_threshold = getattr(cfg, "decision_threshold", 0.5)

        log.info(
            f"DrowsinessClassifier carregado | classes={self.classes} | "
            f"features={metadata['feature_names']} | "
            f"acurácia no treino={metadata.get('test_accuracy', '?')}"
        )

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        for face in faces:
            if face.landmarks is None:
                continue

            x = self.feature_extractor.extract(face).reshape(1, -1)
            x_scaled = self.scaler.transform(x)

            probabilities = self.model.predict_proba(x_scaled)[0]
            predicted_idx = int(np.argmax(probabilities))
            predicted_label = self.model.classes_[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            face.extra["drowsiness_label"] = predicted_label
            face.extra["drowsiness_confidence"] = round(confidence, 3)
            face.extra["drowsiness_probabilities"] = {
                cls: round(float(p), 3) for cls, p in zip(self.model.classes_, probabilities)
            }

        return faces
