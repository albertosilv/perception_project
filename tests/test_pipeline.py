"""
Testes básicos ("smoke tests") do pipeline de percepção.

Rodar com:
    pytest tests/
"""

import numpy as np
import pytest

from src.core.config_loader import load_config
from src.perception.face_detector import FaceDetector
from src.perception.types import Face, BoundingBox


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/config.yaml")


def test_config_loads(cfg):
    assert cfg.camera.width == 640
    assert cfg.face_detection.backend in ("haar", "dlib_hog")


def test_bounding_box_helpers():
    bbox = BoundingBox(10, 20, 100, 50)
    assert bbox.as_tuple() == (10, 20, 100, 50)
    assert bbox.center() == (60, 45)


def test_face_detector_runs_on_blank_frame(cfg):
    detector = FaceDetector(cfg.face_detection)
    blank_frame = np.zeros((cfg.camera.height, cfg.camera.width, 3), dtype=np.uint8)

    faces = detector.detect(blank_frame)

    # Frame preto não deve gerar nenhuma detecção falsa
    assert isinstance(faces, list)
    assert len(faces) == 0
