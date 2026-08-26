"""
Baixa e extrai o modelo de landmarks faciais do dlib (68 pontos)
para a pasta models/.

Uso:
    python scripts/download_models.py
"""

from __future__ import annotations
import bz2
import shutil
import urllib.request
from pathlib import Path

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
COMPRESSED_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat.bz2"
OUTPUT_PATH = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"


def download_landmark_model() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Modelo já existe em: {OUTPUT_PATH}")
        return

    print(f"Baixando modelo de: {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, COMPRESSED_PATH)

    print("Extraindo arquivo .bz2...")
    with bz2.open(COMPRESSED_PATH, "rb") as f_in, open(OUTPUT_PATH, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    COMPRESSED_PATH.unlink()
    print(f"Modelo pronto em: {OUTPUT_PATH}")


if __name__ == "__main__":
    download_landmark_model()
