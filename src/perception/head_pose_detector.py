from __future__ import annotations
import time
import cv2
import numpy as np
from types import SimpleNamespace

from src.perception.base_detector import BaseDetector
from src.perception.types import Face
from src.core.logger import get_logger

log = get_logger(__name__)

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          
    (0.0, -330.0, -65.0),     
    (-225.0, 170.0, -135.0),  
    (225.0, 170.0, -135.0),   
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),  
], dtype=np.float64)

LANDMARK_IDX = [30, 8, 36, 45, 48, 54]

class HeadPoseDetector(BaseDetector):
    """
    Detector de pose da cabeça com calibração dinâmica baseada em TEMPO.
    
    A calibração é essencial porque a posição da câmera (ex: notebook em mesa baixa)
    pode fazer com que a pessoa pareça estar com a cabeça inclinada mesmo em posição
    natural. A calibração estabelece uma "pose neutra" de referência.
    
    TODOS os parâmetros de tempo são baseados em segundos (time.time()),
    não em frames, para ser independente de FPS/hardware.
    """
    
    def __init__(self, cfg: SimpleNamespace):
        self.draw_axis = getattr(cfg, "draw_axis", True)

        # Limiares angulares (graus) para alertas - agora relativos à baseline
        self.roll_alert_threshold = getattr(cfg, "roll_alert_threshold", 20.0)
        self.roll_min_duration_s = getattr(cfg, "roll_min_duration_s", 0.8)

        self.pitch_nodding_threshold = getattr(cfg, "pitch_nodding_threshold", 12.0)
        self.pitch_min_duration_s = getattr(cfg, "pitch_min_duration_s", 0.8)

        self.yaw_distraction_threshold = getattr(cfg, "yaw_distraction_threshold", 20.0)
        self.yaw_min_duration_s = getattr(cfg, "yaw_min_duration_s", 0.8)

        # Configurações de calibração baseadas em TEMPO
        self.calibration_duration_s = getattr(cfg, "calibration_duration_s", 1.5)
        self.auto_recalibrate = getattr(cfg, "auto_recalibrate", True)
        self.stable_threshold_s = getattr(cfg, "stable_threshold_s", 5.0)
        self.calibration_max_variation = getattr(cfg, "calibration_max_variation", 5.0)

        # Estado interno
        self._roll_since: dict[int, float | None] = {}
        self._pitch_since: dict[int, float | None] = {}
        self._yaw_since: dict[int, float | None] = {}

        # Calibração baseada em TEMPO
        self._baseline: dict[int, dict[str, float]] = {}
        self._calibration_start_time: dict[int, float] = {}  # Quando começou a calibração
        self._calibration_samples: dict[int, list[dict[str, float]]] = {}
        self._last_stable_time: dict[int, float] = {}
        self._is_calibrating: dict[int, bool] = {}

    def _start_calibration(self, face_id: int) -> None:
        """Inicia o processo de calibração para uma face."""
        self._calibration_samples[face_id] = []
        self._calibration_start_time[face_id] = time.time()
        self._is_calibrating[face_id] = True
        log.info(f"Iniciando calibração para Face {face_id} - "
                f"Duração mínima: {self.calibration_duration_s}s")

    def _add_calibration_sample(self, face_id: int, pitch: float, yaw: float, roll: float) -> bool:
        """
        Adiciona uma amostra para calibração baseada em TEMPO.
        Retorna True quando a calibração é concluída.
        """
        now = time.time()
        
        if face_id not in self._calibration_samples:
            self._start_calibration(face_id)
        
        self._calibration_samples[face_id].append({
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "timestamp": now
        })
        
        # Verifica se já passou tempo suficiente
        elapsed = now - self._calibration_start_time[face_id]
        
        if elapsed >= self.calibration_duration_s:
            # Calcula a média e desvio padrão
            samples = self._calibration_samples[face_id]
            
            pitches = [s["pitch"] for s in samples]
            yaws = [s["yaw"] for s in samples]
            rolls = [s["roll"] for s in samples]
            
            # Verifica se a cabeça ficou estável durante a calibração
            pitch_std = np.std(pitches)
            yaw_std = np.std(yaws)
            roll_std = np.std(rolls)
            
            max_std = max(pitch_std, yaw_std, roll_std)
            
            if max_std > self.calibration_max_variation:
                log.warning(f"Calibração da Face {face_id} teve muita variação "
                          f"({max_std:.1f}° > {self.calibration_max_variation}°). "
                          f"Reiniciando calibração...")
                # Reinicia calibração
                self._calibration_samples[face_id] = []
                self._calibration_start_time[face_id] = now
                return False
            
            # Usa mediana para ser robusto a outliers
            avg_pitch = np.median(pitches)
            avg_yaw = np.median(yaws)
            avg_roll = np.median(rolls)
            
            self._baseline[face_id] = {
                "pitch": avg_pitch,
                "yaw": avg_yaw,
                "roll": avg_roll
            }
            
            # Log da calibração
            log.info(f"Calibração CONCLUÍDA para Face {face_id} | "
                    f"Pitch: {avg_pitch:.1f}°, Yaw: {avg_yaw:.1f}°, Roll: {avg_roll:.1f}° | "
                    f"Tempo: {elapsed:.2f}s, Amostras: {len(samples)} | "
                    f"Variação: {max_std:.2f}°")
            
            # Limpa dados temporários
            del self._calibration_samples[face_id]
            del self._calibration_start_time[face_id]
            self._is_calibrating[face_id] = False
            return True
        
        return False

    def _check_auto_recalibration(self, face_id: int, pitch: float, yaw: float, roll: float) -> None:
        """
        Verifica se a cabeça está estável por tempo suficiente para recalibrar.
        Baseado em TEMPO (segundos), não em frames.
        """
        if not self.auto_recalibrate or face_id not in self._baseline:
            return
        
        baseline = self._baseline[face_id]
        pitch_diff = abs(pitch - baseline["pitch"])
        yaw_diff = abs(yaw - baseline["yaw"])
        roll_diff = abs(roll - baseline["roll"])
        
        # Se os ângulos estiverem dentro de 3 graus da baseline, considera estável
        if pitch_diff < 3.0 and yaw_diff < 3.0 and roll_diff < 3.0:
            now = time.time()
            if face_id not in self._last_stable_time:
                self._last_stable_time[face_id] = now
            elif now - self._last_stable_time[face_id] >= self.stable_threshold_s:
                # Estável por tempo suficiente - recalibra
                log.info(f"Recalibração automática para Face {face_id} | "
                        f"Posição estável por {self.stable_threshold_s}s")
                
                # Atualiza baseline com posição atual
                self._baseline[face_id] = {
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll
                }
                
                # Reseta timers de alerta
                self._pitch_since[face_id] = None
                self._yaw_since[face_id] = None
                self._roll_since[face_id] = None
                
                self._last_stable_time[face_id] = now
        else:
            # Não está estável, reseta o timer
            if face_id in self._last_stable_time:
                del self._last_stable_time[face_id]

    def detect(self, frame, faces: list[Face]) -> list[Face]:
        h, w = frame.shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1)) 

        now = time.time()
        for i, face in enumerate(faces):
            if face.landmarks is None:
                continue

            image_points = face.landmarks[LANDMARK_IDX].astype(np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                continue

            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            pitch, yaw, roll = [float(a) for a in euler_angles.flatten()]

            # ==============================================================
            # CALIBRAÇÃO INICIAL (BASEADA EM TEMPO)
            # ==============================================================
            if i not in self._baseline:
                # Ainda em calibração
                if not self._add_calibration_sample(i, pitch, yaw, roll):
                    # Em calibração - não processa alertas ainda
                    elapsed = now - self._calibration_start_time.get(i, now)
                    face.extra["head_pose"] = {
                        "pitch": round(pitch, 1),
                        "yaw": round(yaw, 1),
                        "roll": round(roll, 1),
                        "calibrating": True,
                        "progress": f"{elapsed:.1f}/{self.calibration_duration_s}s",
                        "samples": len(self._calibration_samples.get(i, []))
                    }
                    continue
            
            # ==============================================================
            # APLICA BASELINE E CALCULA AJUSTE
            # ==============================================================
            baseline = self._baseline[i]
            pitch_adjusted = pitch - baseline["pitch"]
            yaw_adjusted = yaw - baseline["yaw"]
            roll_adjusted = roll - baseline["roll"]

            # ==============================================================
            # AUTO-RECALIBRAÇÃO (opcional)
            # ==============================================================
            if self.auto_recalibrate:
                self._check_auto_recalibration(i, pitch, yaw, roll)
                # Se recalibrou, recalcula ajustes
                baseline = self._baseline[i]
                pitch_adjusted = pitch - baseline["pitch"]
                yaw_adjusted = yaw - baseline["yaw"]
                roll_adjusted = roll - baseline["roll"]

            # ==============================================================
            # ARMAZENA RESULTADOS
            # ==============================================================
            face.extra["head_pose"] = {
                "pitch": round(pitch, 1),
                "yaw": round(yaw, 1),
                "roll": round(roll, 1),
                "pitch_adjusted": round(pitch_adjusted, 1),
                "yaw_adjusted": round(yaw_adjusted, 1),
                "roll_adjusted": round(roll_adjusted, 1),
                "calibrated": True,
                "baseline": {
                    "pitch": round(baseline["pitch"], 1),
                    "yaw": round(baseline["yaw"], 1),
                    "roll": round(baseline["roll"], 1)
                }
            }
            face.extra["_pnp"] = (rotation_vec, translation_vec, camera_matrix, dist_coeffs)

            # ==============================================================
            # ALERTAS (usando valores ajustados)
            # ==============================================================

            # --- Inclinação lateral (roll): sonolência / perda de tônus ---
            roll_since = self._roll_since.get(i)
            roll_alert = False
            if abs(roll_adjusted) > self.roll_alert_threshold:
                if roll_since is None:
                    self._roll_since[i] = now
                else:
                    duration = now - roll_since
                    if duration >= self.roll_min_duration_s:
                        roll_alert = True
                        log.warning(f"Cabeça inclinada (roll) por {duration:.1f}s - Face {i} | "
                                  f"Ângulo ajustado: {roll_adjusted:.1f}° "
                                  f"(original: {roll:.1f}°)")
            else:
                if roll_since is not None:
                    duration = now - roll_since
                    if duration >= self.roll_min_duration_s:
                        log.info(f"Cabeça corrigiu inclinação após {duration:.1f}s - Face {i}")
                self._roll_since[i] = None
            face.extra["head_tilt_alert"] = roll_alert

            # --- Inclinação frontal (pitch): cabeceio de sonolência ---
            pitch_since = self._pitch_since.get(i)
            nodding_alert = False
            if abs(pitch_adjusted) > self.pitch_nodding_threshold:
                if pitch_since is None:
                    self._pitch_since[i] = now
                else:
                    duration = now - pitch_since
                    if duration >= self.pitch_min_duration_s:
                        nodding_alert = True
                        log.warning(f"Cabeceio (pitch) por {duration:.1f}s - Face {i} | "
                                  f"Ângulo ajustado: {pitch_adjusted:.1f}° "
                                  f"(original: {pitch:.1f}°) | ALERTA DE SONOLÊNCIA")
            else:
                if pitch_since is not None:
                    duration = now - pitch_since
                    if duration >= self.pitch_min_duration_s:
                        log.info(f"Cabeceio interrompido após {duration:.1f}s - Face {i}")
                self._pitch_since[i] = None
            face.extra["nodding_alert"] = nodding_alert

            # --- Rotação (yaw): desvio lateral / distração ---
            yaw_since = self._yaw_since.get(i)
            distraction_alert = False
            if abs(yaw_adjusted) > self.yaw_distraction_threshold:
                if yaw_since is None:
                    self._yaw_since[i] = now
                else:
                    duration = now - yaw_since
                    if duration >= self.yaw_min_duration_s:
                        distraction_alert = True
                        log.warning(f"Distração lateral (yaw) por {duration:.1f}s - Face {i} | "
                                  f"Ângulo ajustado: {yaw_adjusted:.1f}° "
                                  f"(original: {yaw:.1f}°)")
            else:
                if yaw_since is not None:
                    duration = now - yaw_since
                    if duration >= self.yaw_min_duration_s:
                        log.info(f"Retorno ao foco após {duration:.1f}s - Face {i}")
                self._yaw_since[i] = None
            face.extra["distraction_alert"] = distraction_alert

        return faces