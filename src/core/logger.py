"""
Logger centralizado do projeto.

Uso:
    from src.core.logger import get_logger
    log = get_logger(__name__)
    log.info("mensagem")
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    level: str = "INFO",
    log_to_file: bool = False,
    log_file: str = "data/output/perception.log",
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        # evita adicionar handlers duplicados se o logger já foi criado
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
