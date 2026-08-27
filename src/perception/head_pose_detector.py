from __future__ import annotations

import time
import cv2
import numpy as np
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)


# =========================================================
# MODELO 3D DO ROSTO
# =========================================================

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),            # Nariz
    (0.0, -330.0, -65.0),       # Queixo
    (-225.0, 170.0, -135.0),    # Olho esquerdo
    (225.0, 170.0, -135.0),     # Olho direito
    (-150.0, -150.0, -125.0),  # Boca esquerda
    (150.0, -150.0, -125.0),   # Boca direita
], dtype=np.float64)


# Correspondência com os landmarks do modelo de 68 pontos.
LANDMARK_IDX = [30, 8, 36, 45, 48, 54]


class HeadPoseDetector(BaseDetector):
    """
    Estima a pose da cabeça utilizando solvePnP.

    Features produzidas:

        pitch
        yaw
        roll

    Também detecta estados temporais:

        head_tilt_alert
        nodding_alert
        distraction_alert

    A pose absoluta é corrigida utilizando uma baseline individual
    obtida durante a calibração.

    Importante:
        Esses eventos são indicadores observáveis de comportamento
        da cabeça. Eles não representam, isoladamente, fadiga ou
        sonolência.
    """

    def __init__(self, cfg: SimpleNamespace):

        # =====================================================
        # DISPLAY
        # =====================================================

        self.draw_axis = getattr(
            cfg,
            "draw_axis",
            True,
        )

        # =====================================================
        # ROLL
        # =====================================================

        self.roll_alert_threshold = getattr(
            cfg,
            "roll_alert_threshold",
            20.0,
        )

        self.roll_min_duration_s = getattr(
            cfg,
            "roll_min_duration_s",
            0.8,
        )

        # =====================================================
        # PITCH
        # =====================================================

        self.pitch_nodding_threshold = getattr(
            cfg,
            "pitch_nodding_threshold",
            12.0,
        )

        self.pitch_min_duration_s = getattr(
            cfg,
            "pitch_min_duration_s",
            0.8,
        )

        # =====================================================
        # YAW
        # =====================================================

        self.yaw_distraction_threshold = getattr(
            cfg,
            "yaw_distraction_threshold",
            20.0,
        )

        self.yaw_min_duration_s = getattr(
            cfg,
            "yaw_min_duration_s",
            0.8,
        )

        # =====================================================
        # CALIBRAÇÃO
        # =====================================================

        self.calibration_duration_s = getattr(
            cfg,
            "calibration_duration_s",
            1.5,
        )

        self.auto_recalibrate = getattr(
            cfg,
            "auto_recalibrate",
            True,
        )

        self.stable_threshold_s = getattr(
            cfg,
            "stable_threshold_s",
            5.0,
        )

        self.calibration_max_variation = getattr(
            cfg,
            "calibration_max_variation",
            5.0,
        )

        # =====================================================
        # ESTADO DOS EVENTOS
        # =====================================================

        self._roll_since: dict[int, float | None] = {}
        self._pitch_since: dict[int, float | None] = {}
        self._yaw_since: dict[int, float | None] = {}

        # =====================================================
        # BASELINE
        # =====================================================

        self._baseline: dict[int, dict[str, float]] = {}

        # =====================================================
        # ESTADO DA CALIBRAÇÃO
        # =====================================================

        self._calibration_start_time: dict[int, float] = {}

        self._calibration_samples: dict[
            int,
            list[dict[str, float]]
        ] = {}

        self._is_calibrating: dict[int, bool] = {}

        # =====================================================
        # RECALIBRAÇÃO AUTOMÁTICA
        # =====================================================

        self._last_stable_time: dict[int, float] = {}

        # Evita recalibrar repetidamente sem que a pessoa
        # realmente saia e volte para uma nova posição.
        self._stable_recalibrated: dict[int, bool] = {}

    # =========================================================
    # CALIBRAÇÃO
    # =========================================================

    def _start_calibration(
        self,
        face_id: int,
    ) -> None:

        self._calibration_samples[face_id] = []

        self._calibration_start_time[face_id] = time.time()

        self._is_calibrating[face_id] = True

        log.info(
            f"Iniciando calibração para Face {face_id} | "
            f"Duração mínima: "
            f"{self.calibration_duration_s:.1f}s"
        )

    def _add_calibration_sample(
        self,
        face_id: int,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> bool:

        now = time.time()

        if face_id not in self._calibration_samples:

            self._start_calibration(face_id)

        self._calibration_samples[face_id].append({
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "timestamp": now,
        })

        elapsed = (
            now -
            self._calibration_start_time[face_id]
        )

        # Ainda não atingiu o tempo mínimo.
        if elapsed < self.calibration_duration_s:
            return False

        samples = self._calibration_samples[face_id]

        pitches = [
            sample["pitch"]
            for sample in samples
        ]

        yaws = [
            sample["yaw"]
            for sample in samples
        ]

        rolls = [
            sample["roll"]
            for sample in samples
        ]

        # =====================================================
        # ESTABILIDADE
        # =====================================================

        pitch_std = np.std(pitches)
        yaw_std = np.std(yaws)
        roll_std = np.std(rolls)

        max_std = max(
            pitch_std,
            yaw_std,
            roll_std,
        )

        if max_std > self.calibration_max_variation:

            log.warning(
                f"Calibração da Face {face_id} "
                f"instável | "
                f"Variação: {max_std:.2f}°"
            )

            self._calibration_samples[face_id] = []

            self._calibration_start_time[face_id] = now

            return False

        # =====================================================
        # BASELINE
        # =====================================================

        self._baseline[face_id] = {
            "pitch": float(np.median(pitches)),
            "yaw": float(np.median(yaws)),
            "roll": float(np.median(rolls)),
        }

        log.info(
            f"Calibração concluída - Face {face_id} | "
            f"Pitch: {self._baseline[face_id]['pitch']:.1f}° | "
            f"Yaw: {self._baseline[face_id]['yaw']:.1f}° | "
            f"Roll: {self._baseline[face_id]['roll']:.1f}° | "
            f"Amostras: {len(samples)}"
        )

        del self._calibration_samples[face_id]
        del self._calibration_start_time[face_id]

        self._is_calibrating[face_id] = False

        return True

    # =========================================================
    # RECALIBRAÇÃO AUTOMÁTICA
    # =========================================================

    def _check_auto_recalibration(
        self,
        face_id: int,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> None:

        if not self.auto_recalibrate:
            return

        if face_id not in self._baseline:
            return

        baseline = self._baseline[face_id]

        pitch_diff = abs(
            pitch - baseline["pitch"]
        )

        yaw_diff = abs(
            yaw - baseline["yaw"]
        )

        roll_diff = abs(
            roll - baseline["roll"]
        )

        stable = (
            pitch_diff < 3.0
            and yaw_diff < 3.0
            and roll_diff < 3.0
        )

        now = time.time()

        if not stable:

            self._last_stable_time.pop(
                face_id,
                None,
            )

            self._stable_recalibrated[face_id] = False

            return

        if face_id not in self._last_stable_time:

            self._last_stable_time[face_id] = now

            return

        stable_duration = (
            now -
            self._last_stable_time[face_id]
        )

        # Já recalibrou durante este período de estabilidade.
        if self._stable_recalibrated.get(
            face_id,
            False,
        ):
            return

        if stable_duration >= self.stable_threshold_s:

            log.info(
                f"Recalibração automática - "
                f"Face {face_id} | "
                f"Estável por {stable_duration:.1f}s"
            )

            self._baseline[face_id] = {
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll,
            }

            self._stable_recalibrated[face_id] = True

            # Reinicia timers dos eventos.
            self._pitch_since[face_id] = None
            self._yaw_since[face_id] = None
            self._roll_since[face_id] = None

    # =========================================================
    # DETECT
    # =========================================================

    def detect(
        self,
        frame,
        faces: list[Face],
    ) -> list[Face]:

        h, w = frame.shape[:2]

        focal_length = w

        center = (
            w / 2,
            h / 2,
        )

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        now = time.time()

        for i, face in enumerate(faces):

            if face.landmarks is None:
                continue

            # =====================================================
            # PONTOS 2D
            # =====================================================

            image_points = (
                face.landmarks[LANDMARK_IDX]
                .astype(np.float64)
            )

            # =====================================================
            # SOLVE PNP
            # =====================================================

            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS_3D,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if not success:
                continue

            # =====================================================
            # ROTAÇÃO → ÂNGULOS DE EULER
            # =====================================================

            rotation_mat, _ = cv2.Rodrigues(
                rotation_vec
            )

            pose_mat = cv2.hconcat(
                (
                    rotation_mat,
                    translation_vec,
                )
            )

            (
                _,
                _,
                _,
                _,
                _,
                _,
                euler_angles,
            ) = cv2.decomposeProjectionMatrix(
                pose_mat
            )

            pitch, yaw, roll = [
                float(angle)
                for angle in euler_angles.flatten()
            ]

            # =====================================================
            # CALIBRAÇÃO INICIAL
            # =====================================================

            if i not in self._baseline:

                completed = self._add_calibration_sample(
                    i,
                    pitch,
                    yaw,
                    roll,
                )

                if not completed:

                    elapsed = (
                        now -
                        self._calibration_start_time.get(
                            i,
                            now,
                        )
                    )

                    face.extra["head_pose"] = {
                        "pitch": round(pitch, 1),
                        "yaw": round(yaw, 1),
                        "roll": round(roll, 1),
                        "calibrating": True,
                        "progress": (
                            f"{elapsed:.1f}/"
                            f"{self.calibration_duration_s:.1f}s"
                        ),
                        "samples": len(
                            self._calibration_samples.get(
                                i,
                                [],
                            )
                        ),
                    }

                    continue

            # =====================================================
            # RECALIBRAÇÃO
            #
            # Precisa rodar ANTES do cálculo de desvio abaixo, pois
            # pode atualizar a baseline no meio deste frame.
            # =====================================================

            self._check_auto_recalibration(
                i,
                pitch,
                yaw,
                roll,
            )

            # =====================================================
            # DESVIO DA BASELINE (já considerando recalibração acima)
            # =====================================================

            baseline = self._baseline[i]

            pitch_adjusted = (
                pitch -
                baseline["pitch"]
            )

            yaw_adjusted = (
                yaw -
                baseline["yaw"]
            )

            roll_adjusted = (
                roll -
                baseline["roll"]
            )

            # =====================================================
            # FEATURES
            # =====================================================

            face.extra["head_pose"] = {

                # Valores absolutos.
                "pitch": round(pitch, 1),
                "yaw": round(yaw, 1),
                "roll": round(roll, 1),

                # Valores relativos à posição neutra.
                "pitch_adjusted": round(
                    pitch_adjusted,
                    1,
                ),

                "yaw_adjusted": round(
                    yaw_adjusted,
                    1,
                ),

                "roll_adjusted": round(
                    roll_adjusted,
                    1,
                ),

                "calibrated": True,

                "baseline": {
                    "pitch": round(
                        baseline["pitch"],
                        1,
                    ),
                    "yaw": round(
                        baseline["yaw"],
                        1,
                    ),
                    "roll": round(
                        baseline["roll"],
                        1,
                    ),
                },
            }

            # =====================================================
            # ROLL — INCLINAÇÃO LATERAL
            # =====================================================

            roll_since = self._roll_since.get(i)

            roll_alert = False
            roll_duration = 0.0

            if abs(roll_adjusted) > self.roll_alert_threshold:

                if roll_since is None:

                    self._roll_since[i] = now

                else:

                    roll_duration = (
                        now -
                        roll_since
                    )

                    if (
                        roll_duration >=
                        self.roll_min_duration_s
                    ):

                        roll_alert = True

            else:

                self._roll_since[i] = None

            face.extra["head_tilt_alert"] = roll_alert

            face.extra["head_tilt_duration_s"] = round(
                roll_duration,
                2,
            )

            # =====================================================
            # PITCH — CABECEIO
            # =====================================================

            pitch_since = self._pitch_since.get(i)

            nodding_alert = False
            nodding_duration = 0.0

            if abs(pitch_adjusted) > self.pitch_nodding_threshold:

                if pitch_since is None:

                    self._pitch_since[i] = now

                else:

                    nodding_duration = (
                        now -
                        pitch_since
                    )

                    if (
                        nodding_duration >=
                        self.pitch_min_duration_s
                    ):

                        nodding_alert = True

            else:

                self._pitch_since[i] = None

            face.extra["nodding_alert"] = nodding_alert

            face.extra["nodding_duration_s"] = round(
                nodding_duration,
                2,
            )

            # =====================================================
            # YAW — DESVIO LATERAL
            # =====================================================

            yaw_since = self._yaw_since.get(i)

            distraction_alert = False
            distraction_duration = 0.0

            if abs(yaw_adjusted) > self.yaw_distraction_threshold:

                if yaw_since is None:

                    self._yaw_since[i] = now

                else:

                    distraction_duration = (
                        now -
                        yaw_since
                    )

                    if (
                        distraction_duration >=
                        self.yaw_min_duration_s
                    ):

                        distraction_alert = True

            else:

                self._yaw_since[i] = None

            face.extra["distraction_alert"] = (
                distraction_alert
            )

            face.extra["distraction_duration_s"] = round(
                distraction_duration,
                2,
            )

        return faces