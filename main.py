"""
Ponto de entrada do projeto de percepção.

Uso:
    python main.py
    python main.py --config config/config.yaml

Pressione 'q' para sair.
"""

from __future__ import annotations
import argparse
import logging
import cv2

from src.core.config_loader import load_config
from src.core.logger import get_logger
from src.core.camera import Camera
from src.perception.pipeline import PerceptionPipeline
from src.utils.visualization import draw_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline de percepção com OpenCV + dlib")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Caminho para o arquivo de configuração YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    log = get_logger(
        __name__,
        level=cfg.logging.level,
        log_to_file=cfg.logging.log_to_file,
        log_file=cfg.logging.log_file,
    )

    try:
        log.info("Inicializando pipeline de percepção...")
        pipeline = PerceptionPipeline(cfg)

        with Camera(cfg.camera) as camera:
            log.info("Câmera aberta. Pressione 'q' para sair.")

            while True:
                ok, frame = camera.read()
                if not ok:
                    log.warning("Falha ao ler frame da câmera. Encerrando.")
                    break

                result = pipeline.process(frame)
                output = draw_result(frame, result, cfg.display)

                cv2.imshow(cfg.display.window_name, output)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Encerrado pelo usuário.")
                    break

        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        log.info("Encerrado via Ctrl+C.")
    except Exception:
        log.exception("Encerrado por um erro inesperado.")
        raise
    finally:
        # Garante que o log seja gravado em disco (na pasta definida em
        # cfg.logging.log_file) mesmo em caso de erro, Ctrl+C, ou fechar
        # a janela abruptamente — sem isso, as últimas linhas do buffer
        # de log podem se perder. `pipeline.finalize()` salva também os
        # picos de fadiga detectados na sessão (fatigue_spike_detector).
        if "pipeline" in locals():
            pipeline.finalize()
        log.info(f"Log final salvo em: {cfg.logging.log_file}")
        logging.shutdown()


if __name__ == "__main__":
    main()
