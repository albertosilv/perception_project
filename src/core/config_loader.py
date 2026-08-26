"""
Carrega o config.yaml e expõe como objeto acessível por atributos.

Uso:
    from src.core.config_loader import load_config
    cfg = load_config()
    print(cfg.camera.width)
"""

from __future__ import annotations
import yaml
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Converte dicionários aninhados em SimpleNamespace, recursivamente."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            value = _dict_to_namespace(value)
        setattr(ns, key, value)
    return ns


def load_config(path: str | Path = "config/config.yaml") -> SimpleNamespace:
    """
    Carrega o arquivo de configuração YAML do projeto.

    Args:
        path: caminho para o arquivo config.yaml

    Returns:
        SimpleNamespace navegável (ex: cfg.camera.source)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)
