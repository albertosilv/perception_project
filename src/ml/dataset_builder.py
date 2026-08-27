"""
Constrói um dataset supervisionado (X, y) a partir de vídeos rotulados.

Estrutura de pastas esperada (você organiza como quiser, é só isso que
importa — o nome da subpasta É o rótulo da classe):

    data/dataset/
    ├── fadiga/
    │   ├── video1.mp4
    │   └── video2.mp4
    └── sem_fadiga/
        ├── video3.mp4
        └── video4.mp4

Cada frame do vídeo passa pelo PerceptionPipeline (detecção de rosto,
landmarks, EAR, MAR, pose de cabeça, PERCLOS). Quando um rosto com
landmarks é encontrado, o vetor de features é extraído e associado ao
rótulo da pasta.

Frames sem rosto detectado são ignorados (não geram amostra), e é
processado 1 a cada `frame_sample_rate` frames para não gerar amostras
quase idênticas em sequência.
"""

from __future__ import annotations
import csv
from pathlib import Path

import cv2
import numpy as np

from src.perception.pipeline import PerceptionPipeline
from src.preprocessing.feature_extractor import FeatureExtractor
from src.core.logger import get_logger

log = get_logger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def _iter_class_folders(dataset_dir: Path):
    for folder in sorted(dataset_dir.iterdir()):
        if folder.is_dir():
            yield folder.name, folder


def _iter_videos(class_folder: Path):
    for path in sorted(class_folder.iterdir()):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def build_dataset_from_videos(
    dataset_dir: str | Path,
    pipeline: PerceptionPipeline,
    feature_extractor: FeatureExtractor,
    frame_sample_rate: int = 5,
    csv_output_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Percorre `dataset_dir`, roda o pipeline de percepção sobre cada vídeo
    e monta as features rotuladas.

    Returns:
        X: matriz de features, shape (n_amostras, n_features)
        y: vetor de rótulos (strings, ex: "fadiga"/"sem_fadiga"), shape (n_amostras,)
        feature_names: nomes das colunas de X, na mesma ordem
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Pasta do dataset não encontrada: {dataset_dir.resolve()}\n"
            "Organize seus vídeos em subpastas por classe, ex:\n"
            f"  {dataset_dir}/fadiga/*.mp4\n"
            f"  {dataset_dir}/sem_fadiga/*.mp4"
        )

    class_folders = list(_iter_class_folders(dataset_dir))
    if not class_folders:
        raise ValueError(
            f"Nenhuma subpasta de classe encontrada em {dataset_dir.resolve()}. "
            "Cada subpasta representa um rótulo (ex: 'fadiga', 'sem_fadiga')."
        )

    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []

    for label, class_folder in class_folders:
        videos = list(_iter_videos(class_folder))
        if not videos:
            log.warning(f"Nenhum vídeo encontrado em '{class_folder}' (classe '{label}').")
            continue

        for video_path in videos:
            log.info(f"Processando vídeo: {video_path} (classe='{label}')")
            n_samples_video = _process_video(
                video_path, pipeline, feature_extractor,
                frame_sample_rate, label, X_rows, y_rows,
            )
            log.info(f"  -> {n_samples_video} amostras extraídas.")

    if not X_rows:
        raise RuntimeError(
            "Nenhuma amostra foi extraída de nenhum vídeo. Verifique se os "
            "vídeos têm rostos visíveis e se 'landmarks.enabled: true' no config."
        )

    X = np.vstack(X_rows)
    y = np.array(y_rows)

    log.info(f"Dataset construído: {X.shape[0]} amostras, {X.shape[1]} features, "
              f"classes: {sorted(set(y_rows))}")

    if csv_output_path is not None:
        _save_dataset_csv(csv_output_path, X, y, feature_extractor.feature_keys)

    return X, y, feature_extractor.feature_keys


def _process_video(
    video_path: Path,
    pipeline: PerceptionPipeline,
    feature_extractor: FeatureExtractor,
    frame_sample_rate: int,
    label: str,
    X_rows: list[np.ndarray],
    y_rows: list[str],
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Não foi possível abrir o vídeo: {video_path}")
        return 0

    frame_idx = 0
    n_samples = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_sample_rate == 0:
            result = pipeline.process(frame)
            for face in result.faces:
                if face.landmarks is None:
                    continue
                X_rows.append(feature_extractor.extract(face))
                y_rows.append(label)
                n_samples += 1

        frame_idx += 1

    cap.release()
    return n_samples


def _save_dataset_csv(path: str | Path, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*feature_names, "label"])
        for row, label in zip(X, y):
            writer.writerow([*row.tolist(), label])

    log.info(f"Dataset salvo em CSV: {path.resolve()}")
