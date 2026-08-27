"""
Pequeno utilitário compartilhado para garantir que pastas de saída
(logs, modelos, datasets processados) sejam criadas onde o config.yaml
manda, de forma consistente entre main.py e os scripts de treino.
"""

from __future__ import annotations
from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """Garante que a pasta-mãe de `path` existe e retorna `path` como Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
