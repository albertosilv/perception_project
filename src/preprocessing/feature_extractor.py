"""
Converte o dicionário `face.extra` (preenchido pelos detectores do pipeline:
EyeStateDetector, YawnDetector, HeadPoseDetector, PerclosDetector) em um
vetor numérico de features, na mesma ordem sempre — usado tanto para
treinar o modelo quanto para classificar em tempo real.

Mantido em um único lugar para garantir que o vetor de features do treino
e o da inferência sejam construídos exatamente da mesma forma. Divergência
aqui é a causa mais comum de "o modelo treinou bem mas classifica errado
em produção".
"""

from __future__ import annotations
import numpy as np

from src.perception.types import Face


def _get_nested(extra: dict, dotted_key: str, default: float = 0.0) -> float:
    """
    Busca um valor em `face.extra`, suportando chaves aninhadas com ".".

    Exemplo:
        _get_nested(extra, "head_pose.yaw_adjusted")
        -> extra["head_pose"]["yaw_adjusted"]

    Retorna `default` se qualquer parte do caminho não existir, for None,
    ou não for numérica (ex: bool é aceito e convertido para 0/1).
    """
    keys = dotted_key.split(".")
    value = extra
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


class FeatureExtractor:
    """
    Constrói vetores de features a partir de objetos `Face`.

    Args:
        feature_keys: lista de chaves de `face.extra` a extrair, na ordem
            desejada. Suporta notação com ponto para valores aninhados
            (ex: "head_pose.yaw_adjusted"). Essa ordem é o "contrato" do
            modelo — precisa ser idêntica entre treino e inferência.
        default: valor usado quando uma feature não está presente
            (ex: detector correspondente estava desabilitado no frame).
    """

    def __init__(self, feature_keys: list[str], default: float = 0.0):
        if not feature_keys:
            raise ValueError("feature_keys não pode ser vazio.")
        self.feature_keys = list(feature_keys)
        self.default = default

    def extract(self, face: Face) -> np.ndarray:
        """Extrai o vetor de features (shape: (n_features,)) de um único rosto."""
        return np.array(
            [_get_nested(face.extra, key, self.default) for key in self.feature_keys],
            dtype=np.float64,
        )

    def extract_batch(self, faces: list[Face]) -> np.ndarray:
        """Extrai features de vários rostos de uma vez (shape: (n_faces, n_features))."""
        if not faces:
            return np.empty((0, len(self.feature_keys)), dtype=np.float64)
        return np.vstack([self.extract(face) for face in faces])
