"""
Treina um classificador de Regressão Logística para classificar o estado
do motorista/usuário (ex: "fadiga" vs "sem_fadiga") a partir das features
geradas pelo PerceptionPipeline (EAR, MAR, PERCLOS, pose de cabeça, etc).

Uso:
    python -m src.ml.train_model
    python -m src.ml.train_model --config config/config.yaml

Espera vídeos organizados em `training.dataset_dir`, uma subpasta por
classe (veja src/ml/dataset_builder.py para o formato exato).

Ao final, salva:
    - o modelo treinado          -> classifier.model_path
    - o scaler de normalização   -> classifier.scaler_path
    - metadados (features/classes) -> ao lado do modelo, mesmo nome + .json
    - relatório de avaliação     -> training.output_dir/classification_report.txt
    - o log completo do treino   -> training.output_dir/train.log
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.core.config_loader import load_config
from src.core.logger import get_logger
from src.core.paths import ensure_parent_dir
from src.perception.pipeline import PerceptionPipeline
from src.preprocessing.feature_extractor import FeatureExtractor
from src.ml.dataset_builder import build_dataset_from_videos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o classificador de regressão logística")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(
        "train_model",
        level=cfg.logging.level,
        log_to_file=True,
        log_file=str(output_dir / "train.log"),
    )

    try:
        log.info("=" * 60)
        log.info("Iniciando treino do classificador")
        log.info(f"Dataset: {cfg.training.dataset_dir}")
        log.info(f"Features: {cfg.classifier.features}")
        log.info("=" * 60)

        # -----------------------------------------------------------
        # 1. Monta o pipeline de percepção (mesmo usado em produção)
        #    para extrair as features dos vídeos de treino.
        # -----------------------------------------------------------
        pipeline = PerceptionPipeline(cfg)
        feature_extractor = FeatureExtractor(cfg.classifier.features)

        csv_path = output_dir / "dataset_features.csv"
        X, y, feature_names = build_dataset_from_videos(
            dataset_dir=cfg.training.dataset_dir,
            pipeline=pipeline,
            feature_extractor=feature_extractor,
            frame_sample_rate=cfg.training.frame_sample_rate,
            csv_output_path=csv_path,
        )

        # -----------------------------------------------------------
        # 2. Split treino/teste
        # -----------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg.training.test_size,
            random_state=cfg.training.random_state,
            stratify=y,
        )
        log.info(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

        # -----------------------------------------------------------
        # 3. Pré-processamento: normalização (fit SÓ no treino)
        # -----------------------------------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # -----------------------------------------------------------
        # 4. Treino da Regressão Logística
        # -----------------------------------------------------------
        model = LogisticRegression(
            max_iter=cfg.training.max_iter,
            class_weight="balanced",  # ajuda se as classes forem desbalanceadas
            random_state=cfg.training.random_state,
        )
        model.fit(X_train_scaled, y_train)

        # -----------------------------------------------------------
        # 5. Avaliação
        # -----------------------------------------------------------
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        log.info(f"Acurácia no conjunto de teste: {accuracy:.3f}")
        log.info("Relatório de classificação:\n" + report)
        log.info(f"Matriz de confusão (labels={list(model.classes_)}):\n{cm}")

        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"Acurácia: {accuracy:.3f}\n\n")
            f.write(report)
            f.write(f"\nMatriz de confusão (labels={list(model.classes_)}):\n{cm}\n")
        log.info(f"Relatório salvo em: {report_path.resolve()}")

        # -----------------------------------------------------------
        # 6. Salva modelo + scaler + metadados
        # -----------------------------------------------------------
        model_path = Path(cfg.classifier.model_path)
        scaler_path = Path(cfg.classifier.scaler_path)
        ensure_parent_dir(model_path)
        ensure_parent_dir(scaler_path)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "feature_names": feature_names,
            "classes": model.classes_.tolist(),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "test_accuracy": accuracy,
        }
        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        log.info(f"Modelo salvo em: {model_path.resolve()}")
        log.info(f"Scaler salvo em: {scaler_path.resolve()}")
        log.info(f"Metadados salvos em: {metadata_path.resolve()}")
        log.info("Treino concluído com sucesso.")

    except Exception:
        log.exception("Treino interrompido por um erro.")
        raise
    finally:
        # -----------------------------------------------------------
        # Garante que todo o log gerado durante o treino seja
        # de fato gravado em disco antes do processo encerrar.
        # -----------------------------------------------------------
        logging.shutdown()


if __name__ == "__main__":
    main()
