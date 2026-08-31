#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorori_debug_speed.py
DORORI 프로젝트: 실제 카메라/YOLO/모터 + 수동 속도 입력 디버그 서버.

파일 배치(기본값):
- 이 서버 파일
- dorori_debug_speed.html
- best_selection.pt
- best노면.pt

모터 기본값:
- BCM STEP=17, DIR=27, ENABLE=22
- 전개/수납 펄스 수=2400 (1.5회전, 540도)
- STEP 주파수=300 Hz
- 전개 1.5회전 -> 5초 유지 -> 반대 방향 1.5회전 자동 수납

환경변수로 주요 설정을 바꿀 수 있습니다. 자세한 내용은 README_KO.md를 참고하세요.
"""

from __future__ import annotations

import atexit
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, request, send_from_directory

try:
    from gpiozero import OutputDevice
except ImportError:  # MOTOR_DRY_RUN=1이면 GPIO 패키지 없이 UI/상태머신 시험 가능
    OutputDevice = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[assignment]




def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


# ───────────────────────────── 기본 설정 ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
RUN_MODE = "DEBUG_SPEED"
HTML_FILE_NAME = "dorori_debug_speed.html"
DOOR_ID = "RR"
PORT = int(os.environ.get("PORT", "5000"))

SEG_MODEL_PATH = Path(
    os.environ.get("SEG_MODEL_PATH", str(BASE_DIR / "best_selection.pt"))
).expanduser()
CLASSIFY_MODEL_PATH = Path(
    os.environ.get("CLASSIFY_MODEL_PATH", str(BASE_DIR / "best노면.pt"))
).expanduser()

CAMERA_SOURCE_RAW = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_SOURCE: int | str = (
    int(CAMERA_SOURCE_RAW)
    if CAMERA_SOURCE_RAW.lstrip("-").isdigit()
    else CAMERA_SOURCE_RAW
)
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "0"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "0"))
CAMERA_STALE_SEC = float(os.environ.get("CAMERA_STALE_SEC", "1.0"))
CAMERA_RETRY_SEC = float(os.environ.get("CAMERA_RETRY_SEC", "2.0"))

CREEP_MAX_KMH = float(os.environ.get("CREEP_MAX_KMH", "5.0"))
STOP_SPEED_EPSILON_KMH = float(os.environ.get("STOP_SPEED_EPSILON_KMH", "0.1"))
STOP_DEPLOY_DELAY_SEC = float(os.environ.get("STOP_DEPLOY_DELAY_SEC", "3.0"))
CONTROL_PERIOD_SEC = float(os.environ.get("CONTROL_PERIOD_SEC", "0.05"))
VISION_INTERVAL_SEC = float(os.environ.get("VISION_INTERVAL_SEC", "1.0"))
VISION_STALE_SEC = float(os.environ.get("VISION_STALE_SEC", "5.0"))

SEG_IMGSZ = int(os.environ.get("SEG_IMGSZ", "640"))
CLASSIFY_IMGSZ = int(os.environ.get("CLASSIFY_IMGSZ", "224"))
SEG_CONF = float(os.environ.get("SEG_CONF", "0.25"))
INFERENCE_DEVICE = os.environ.get("INFERENCE_DEVICE", "").strip()

# 정규화 좌표 (x1, y1, x2, y2). 현재는 전체 화면이며 실차 테스트 후 보정합니다.
VISION_ROI = (0.0, 0.0, 1.0, 1.0)
REACH_ZONE = (0.25, 0.55, 0.75, 0.95)

# A4988 / 스테퍼 모터
STEP_PIN = int(os.environ.get("STEP_PIN", "17"))
DIR_PIN = int(os.environ.get("DIR_PIN", "27"))
ENABLE_PIN = int(os.environ.get("ENABLE_PIN", "22"))
MOTOR_STEPS_PER_REV = int(os.environ.get("MOTOR_STEPS_PER_REV", "2400"))
MOTOR_FREQUENCY_HZ = float(os.environ.get("MOTOR_FREQUENCY_HZ", "300"))
MOTOR_HOLD_SEC = float(os.environ.get("MOTOR_HOLD_SEC", "5.0"))
MOTOR_DEPLOY_DIRECTION = env_bool("MOTOR_DEPLOY_DIRECTION", True)
MOTOR_HOLD_TORQUE = env_bool("MOTOR_HOLD_TORQUE", True)
MOTOR_DRY_RUN = env_bool("MOTOR_DRY_RUN", False)



app = Flask(__name__)
shutdown_event = threading.Event()

vehicle_lock = threading.Lock()
camera_lock = threading.Lock()
perception_lock = threading.Lock()
system_lock = threading.Lock()


# ───────────────────────────── 공유 상태 ─────────────────────────────
vehicle_state: dict[str, Any] = {
    "connected": True,
    "speed_kmh": 10.0,
    "updated_at": time.time(),
    "source": "manual_debug",
    "error": None,
}

camera_state: dict[str, Any] = {
    "connected": False,
    "updated_at": 0.0,
    "error": "카메라 초기화 전",
}
latest_frame: Any = None

perception_state: dict[str, Any] = {
    "model_ready": False,
    "curb": {"detected": False, "reachable": False, "confidence": None},
    "drainage": {"detected": False, "confidence": None},
    "manhole": {"detected": False, "confidence": None},
    "surface": {"class_name": None, "confidence": None, "risky": None},
    "decision": "WAITING",
    "reason_text": "비전 모델 초기화 대기",
    "updated_at": 0.0,
    "error": None,
}

system_state: dict[str, Any] = {
    "door": DOOR_ID,
    "active": False,
    "phase": "IDLE",
    "final_decision": "IDLE",
    "ui_mode": "NEUTRAL",
    "reason_text": "운전자 시작 버튼 대기",
    "stopped_since": None,
    "yellow_since": None,
    "deploy_latched": False,
    "last_motor_action": None,
    "updated_at": time.time(),
}


def decision_to_ui(decision: str) -> str:
    if decision in {"RED", "YELLOW", "GREEN", "FAULT"}:
        return decision
    return "NEUTRAL"


def surface_group(surface_class: str | None) -> str:
    """dry/wet/water는 SAFE, snow/ice는 DANGER, 나머지는 UNKNOWN."""
    if not surface_class:
        return "UNKNOWN"

    value = str(surface_class).strip().lower().replace("_", "-")
    if "snow" in value or "ice" in value:
        return "DANGER"

    if (
        value in {"dry", "wet", "water"}
        or value.startswith("dry-")
        or value.startswith("wet-")
        or value.startswith("water-")
    ):
        return "SAFE"

    return "UNKNOWN"


def decide(
    curb_detected: bool,
    drainage_detected: bool,
    manhole_detected: bool,
    surface_class: str | None,
) -> tuple[str, str]:
    """
    최종 판단 우선순위:
    1) curb -> YELLOW (다른 감지 결과보다 우선)
    2) curb 없음 + drainage -> RED
    3) curb 없음 + snow/ice/알 수 없는 노면 -> RED
    4) curb 없음 + drainage 없음 + 안전 노면 -> GREEN

    manhole은 UI 표시용이며 최종 판단에는 사용하지 않습니다.
    """
    if curb_detected:
        return "YELLOW", "연석이 감지되었습니다. 정차 확인 후 발판을 전개합니다."

    if drainage_detected:
        return "RED", "연석은 없지만 배수로가 감지되어 노면이 불량합니다."

    group = surface_group(surface_class)
    if group == "DANGER":
        return "RED", f"위험 노면이 감지되었습니다 ({surface_class})."

    if group == "UNKNOWN":
        return "RED", f"노면 상태를 안전으로 확인할 수 없습니다 ({surface_class})."

    return "GREEN", f"연석과 배수로가 없고 안전 노면입니다 ({surface_class})."


def crop_frame_to_roi(
    frame: Any, roi: tuple[float, float, float, float]
) -> Any:
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"잘못된 VISION_ROI: {roi}")

    frame_h, frame_w = frame.shape[:2]
    left, top = int(x1 * frame_w), int(y1 * frame_h)
    right = max(left + 1, min(frame_w, int(x2 * frame_w)))
    bottom = max(top + 1, min(frame_h, int(y2 * frame_h)))
    return frame[top:bottom, left:right]


def box_center_in_zone(
    box_xyxy: list[float], frame_w: int, frame_h: int, zone: tuple[float, float, float, float]
) -> bool:
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / frame_w
    cy = (y1 + y2) / 2.0 / frame_h
    zx1, zy1, zx2, zy2 = zone
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def names_to_dict(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def vehicle_snapshot() -> dict[str, Any]:
    with vehicle_lock:
        return dict(vehicle_state)


def camera_snapshot() -> dict[str, Any]:
    with camera_lock:
        return dict(camera_state)


def perception_snapshot() -> dict[str, Any]:
    with perception_lock:
        return {
            "model_ready": perception_state["model_ready"],
            "curb": dict(perception_state["curb"]),
            "drainage": dict(perception_state["drainage"]),
            "manhole": dict(perception_state["manhole"]),
            "surface": dict(perception_state["surface"]),
            "decision": perception_state["decision"],
            "reason_text": perception_state["reason_text"],
            "updated_at": perception_state["updated_at"],
            "error": perception_state["error"],
        }


def system_snapshot() -> dict[str, Any]:
    with system_lock:
        return dict(system_state)


def update_system(**changes: Any) -> None:
    with system_lock:
        system_state.update(changes)
        system_state["updated_at"] = time.time()


def speed_is_fresh(vehicle: dict[str, Any] | None = None) -> bool:
    if vehicle is None:
        vehicle = vehicle_snapshot()
    speed = vehicle.get("speed_kmh")
    return speed is not None and math.isfinite(float(speed)) and float(speed) >= 0.0


def stop_delay_remaining(stopped_since: float | None, now: float | None = None) -> float:
    if stopped_since is None:
        return STOP_DEPLOY_DELAY_SEC
    elapsed = (time.time() if now is None else now) - float(stopped_since)
    return max(0.0, STOP_DEPLOY_DELAY_SEC - elapsed)



def perception_is_fresh(perception: dict[str, Any] | None = None) -> bool:
    if perception is None:
        perception = perception_snapshot()
    if not perception["model_ready"] or perception["error"]:
        return False
    updated_at = float(perception["updated_at"] or 0.0)
    return updated_at > 0.0 and (time.time() - updated_at) <= VISION_STALE_SEC


# ───────────────────────────── 모터 제어 ─────────────────────────────
class MotorController:
    BUSY_STATES = {"STARTING", "DEPLOYING", "DEPLOYED_HOLD", "RETRACTING"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._early_retract = threading.Event()
        self._cycle_id = 0
        self._step_pin: Any = None
        self._dir_pin: Any = None
        self._enable_pin: Any = None
        self._state: dict[str, Any] = {
            "door": DOOR_ID,
            "ready": False,
            "dry_run": MOTOR_DRY_RUN,
            "state": "INITIALIZING",
            "progress": 0,
            "hold_remaining_sec": None,
            "hold_until": None,
            "source": None,
            "cycle_id": 0,
            "error": None,
            "last_completed_at": None,
            "updated_at": time.time(),
        }
        self._initialize_gpio()

    def _initialize_gpio(self) -> None:
        if MOTOR_STEPS_PER_REV <= 0:
            self._set_error("MOTOR_STEPS_PER_REV는 1 이상이어야 합니다.")
            return
        if MOTOR_FREQUENCY_HZ <= 0:
            self._set_error("MOTOR_FREQUENCY_HZ는 0보다 커야 합니다.")
            return
        if MOTOR_HOLD_SEC < 0:
            self._set_error("MOTOR_HOLD_SEC는 0 이상이어야 합니다.")
            return

        if MOTOR_DRY_RUN:
            with self._lock:
                self._state.update({
                    "ready": True,
                    "state": "IDLE",
                    "error": None,
                    "updated_at": time.time(),
                })
            print("[MOTOR] MOTOR_DRY_RUN=1: GPIO 없이 실제 시간으로 동작을 시뮬레이션합니다.")
            return

        if OutputDevice is None:
            self._set_error("gpiozero를 불러올 수 없습니다. python3-gpiozero를 설치하세요.")
            return

        try:
            # ENABLE을 가장 먼저 비활성 상태(HIGH)로 설정합니다.
            self._enable_pin = OutputDevice(
                ENABLE_PIN,
                active_high=False,
                initial_value=False,
            )
            self._step_pin = OutputDevice(STEP_PIN, initial_value=False)
            self._dir_pin = OutputDevice(DIR_PIN, initial_value=False)
            self._enable_pin.off()

            with self._lock:
                self._state.update({
                    "ready": True,
                    "state": "IDLE",
                    "error": None,
                    "updated_at": time.time(),
                })

            print(
                f"[MOTOR] GPIO 준비 완료: STEP={STEP_PIN}, DIR={DIR_PIN}, "
                f"ENABLE={ENABLE_PIN}, move_steps={MOTOR_STEPS_PER_REV}, "
                f"frequency={MOTOR_FREQUENCY_HZ:.1f}Hz"
            )
        except Exception as exc:
            self._set_error(f"GPIO 초기화 실패: {exc}")

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state.update({
                "ready": False,
                "state": "ERROR",
                "error": message,
                "updated_at": time.time(),
            })
        print(f"[MOTOR] 오류: {message}")

    def _update_state(self, **changes: Any) -> None:
        with self._lock:
            self._state.update(changes)
            self._state["updated_at"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = dict(self._state)

        hold_until = data.get("hold_until")
        if data.get("state") == "DEPLOYED_HOLD" and hold_until:
            data["hold_remaining_sec"] = round(max(0.0, float(hold_until) - time.time()), 2)
        else:
            data["hold_remaining_sec"] = None

        data["busy"] = data.get("state") in self.BUSY_STATES
        data.pop("hold_until", None)
        return data

    def request_cycle(self, source: str) -> tuple[bool, str]:
        with self._lock:
            if not self._state["ready"]:
                return False, self._state.get("error") or "motor_not_ready"
            if self._state["state"] in self.BUSY_STATES:
                return False, "motor_busy"
            if self._thread is not None and self._thread.is_alive():
                return False, "motor_thread_busy"

            self._cycle_id += 1
            cycle_id = self._cycle_id
            self._early_retract.clear()
            self._state.update({
                "state": "STARTING",
                "progress": 0,
                "source": source,
                "cycle_id": cycle_id,
                "error": None,
                "updated_at": time.time(),
            })
            self._thread = threading.Thread(
                target=self._cycle_worker,
                args=(cycle_id, source),
                name=f"motor-cycle-{cycle_id}",
                daemon=True,
            )
            self._thread.start()

        return True, "cycle_started"

    def request_retract_now(self) -> tuple[bool, str]:
        state = self.snapshot()["state"]
        if state in {"STARTING", "DEPLOYING", "DEPLOYED_HOLD"}:
            self._early_retract.set()
            return True, "early_retract_requested"
        if state == "RETRACTING":
            return True, "already_retracting"
        return False, "ramp_not_deployed_or_moving"

    def _set_driver_enabled(self, enabled: bool) -> None:
        if MOTOR_DRY_RUN:
            return
        if self._enable_pin is None:
            raise RuntimeError("ENABLE GPIO가 준비되지 않았습니다.")
        if enabled:
            self._enable_pin.on()   # active_high=False이므로 실제 GPIO LOW
        else:
            self._enable_pin.off()  # 실제 GPIO HIGH

    def _rotate_enabled(
        self,
        direction: bool,
        state_name: str,
        start_progress: float,
        end_progress: float,
        steps: int,
        stop_event: threading.Event | None = None,
    ) -> tuple[int, bool]:
        """
        이미 ENABLE된 상태에서 STEP 펄스를 출력합니다.

        Returns:
            (완료한 펄스 수, 중단 요청으로 조기 종료했는지 여부)
        """
        if steps <= 0:
            return 0, bool(stop_event and stop_event.is_set())

        half_period = 1.0 / (2.0 * MOTOR_FREQUENCY_HZ)

        if not MOTOR_DRY_RUN:
            if self._dir_pin is None or self._step_pin is None:
                raise RuntimeError("STEP/DIR GPIO가 준비되지 않았습니다.")
            if direction:
                self._dir_pin.on()
            else:
                self._dir_pin.off()

        # A4988 DIR setup 여유 시간
        time.sleep(0.001)
        report_every = max(1, steps // 100)

        for step_index in range(steps):
            # 전개 중 속도 증가/시스템 종료/수동 수납 요청이 들어오면
            # 다음 펄스를 내보내기 전에 멈추고, 실제 전개된 만큼만 되감습니다.
            if stop_event is not None and stop_event.is_set():
                if not MOTOR_DRY_RUN and self._step_pin is not None:
                    self._step_pin.off()
                return step_index, True

            if not MOTOR_DRY_RUN:
                self._step_pin.on()
            time.sleep(half_period)
            if not MOTOR_DRY_RUN:
                self._step_pin.off()
            time.sleep(half_period)

            if step_index % report_every == 0 or step_index == steps - 1:
                ratio = (step_index + 1) / steps
                progress = start_progress + (end_progress - start_progress) * ratio
                self._update_state(state=state_name, progress=round(progress, 1))

        return steps, False

    def _cycle_worker(self, cycle_id: int, source: str) -> None:
        try:
            self._update_state(
                state="DEPLOYING",
                progress=0,
                source=source,
                cycle_id=cycle_id,
                hold_until=None,
            )
            self._set_driver_enabled(True)
            time.sleep(0.01)

            deployed_steps, interrupted = self._rotate_enabled(
                direction=MOTOR_DEPLOY_DIRECTION,
                state_name="DEPLOYING",
                start_progress=0,
                end_progress=100,
                steps=MOTOR_STEPS_PER_REV,
                stop_event=self._early_retract,
            )
            deployed_progress = 100.0 * deployed_steps / MOTOR_STEPS_PER_REV

            # STARTING/DEPLOYING 직후 중단되어 실제로 한 펄스도 나가지 않은 경우
            if deployed_steps <= 0:
                self._update_state(
                    state="IDLE",
                    progress=0,
                    hold_until=None,
                    last_completed_at=time.time(),
                )
                print(f"[MOTOR] 전개 전 수납 요청으로 사이클 취소: cycle={cycle_id}")
                return

            if not interrupted:
                hold_until = time.time() + MOTOR_HOLD_SEC
                self._update_state(
                    state="DEPLOYED_HOLD",
                    progress=100,
                    hold_until=hold_until,
                )

                if not MOTOR_HOLD_TORQUE:
                    self._set_driver_enabled(False)

                # 설정된 유지 시간이 지나거나 /api/retract가 호출될 때까지 대기
                self._early_retract.wait(timeout=MOTOR_HOLD_SEC)

                if not MOTOR_HOLD_TORQUE:
                    self._set_driver_enabled(True)
                    time.sleep(0.01)
            else:
                print(
                    f"[MOTOR] 전개 중 조기 수납: cycle={cycle_id}, "
                    f"deployed_steps={deployed_steps}/{MOTOR_STEPS_PER_REV}"
                )

            # 정상 전개라면 1.5바퀴, 조기 중단이라면 실제 전개된 펄스 수만큼 역회전합니다.
            self._update_state(
                state="RETRACTING",
                progress=round(deployed_progress, 1),
                hold_until=None,
            )
            self._rotate_enabled(
                direction=not MOTOR_DEPLOY_DIRECTION,
                state_name="RETRACTING",
                start_progress=deployed_progress,
                end_progress=0,
                steps=deployed_steps,
            )

            self._update_state(
                state="IDLE",
                progress=0,
                hold_until=None,
                last_completed_at=time.time(),
            )
            print(f"[MOTOR] 자동 전개/수납 완료: cycle={cycle_id}, source={source}")

        except Exception as exc:
            self._set_error(f"모터 사이클 실패: {exc}")
        finally:
            try:
                if not MOTOR_DRY_RUN and self._step_pin is not None:
                    self._step_pin.off()
                self._set_driver_enabled(False)
            except Exception as exc:
                print(f"[MOTOR] 종료 처리 오류: {exc}")

    def close(self) -> None:
        self._early_retract.set()

        # 가능한 경우 현재 전개량만큼 되감는 모터 스레드가 끝날 시간을 줍니다.
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            estimated_retract_sec = MOTOR_STEPS_PER_REV / MOTOR_FREQUENCY_HZ
            thread.join(timeout=min(10.0, max(2.0, estimated_retract_sec + 2.0)))

        try:
            self._set_driver_enabled(False)
        except Exception:
            pass

        for pin in (self._step_pin, self._dir_pin, self._enable_pin):
            try:
                if pin is not None:
                    pin.close()
            except Exception:
                pass

motor = MotorController()


# ───────────────────────────── 카메라 / 비전 ─────────────────────────────
def camera_loop() -> None:
    global latest_frame

    while not shutdown_event.is_set():
        cap = None
        try:
            print(f"[CAMERA] 연결 시도: {CAMERA_SOURCE}")
            cap = cv2.VideoCapture(CAMERA_SOURCE)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if CAMERA_WIDTH > 0:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            if CAMERA_HEIGHT > 0:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

            if not cap.isOpened():
                raise RuntimeError(f"카메라를 열 수 없습니다: {CAMERA_SOURCE}")

            with camera_lock:
                camera_state.update({"connected": True, "error": None})

            print("[CAMERA] 연결 완료")
            fail_count = 0

            while not shutdown_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    fail_count += 1
                    if fail_count >= 5:
                        raise RuntimeError("카메라 프레임 획득이 연속 5회 실패했습니다.")
                    time.sleep(0.05)
                    continue

                fail_count = 0
                now = time.time()
                with camera_lock:
                    latest_frame = frame
                    camera_state.update({
                        "connected": True,
                        "updated_at": now,
                        "error": None,
                    })

        except Exception as exc:
            with camera_lock:
                camera_state.update({
                    "connected": False,
                    "error": str(exc),
                })
            print(f"[CAMERA] 오류: {exc}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        shutdown_event.wait(CAMERA_RETRY_SEC)


def latest_frame_snapshot() -> tuple[Any | None, float]:
    with camera_lock:
        frame = None if latest_frame is None else latest_frame.copy()
        updated_at = float(camera_state["updated_at"] or 0.0)
    return frame, updated_at


def prediction_kwargs(imgsz: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "verbose": False,
        "imgsz": imgsz,
    }
    if INFERENCE_DEVICE:
        kwargs["device"] = INFERENCE_DEVICE
    return kwargs


def vision_loop() -> None:
    if YOLO is None:
        with perception_lock:
            perception_state.update({
                "model_ready": False,
                "decision": "FAULT",
                "reason_text": "ultralytics 패키지가 설치되지 않았습니다.",
                "error": "ultralytics import 실패",
            })
        return

    try:
        if not SEG_MODEL_PATH.is_file():
            raise FileNotFoundError(f"SEG 가중치 없음: {SEG_MODEL_PATH}")
        if not CLASSIFY_MODEL_PATH.is_file():
            raise FileNotFoundError(f"노면 가중치 없음: {CLASSIFY_MODEL_PATH}")

        print(f"[VISION] SEG 모델 로딩: {SEG_MODEL_PATH}")
        seg_model = YOLO(str(SEG_MODEL_PATH))
        seg_names = names_to_dict(seg_model.names)

        print(f"[VISION] CLASSIFY 모델 로딩: {CLASSIFY_MODEL_PATH}")
        classify_model = YOLO(str(CLASSIFY_MODEL_PATH))

        curb_ids = {idx for idx, name in seg_names.items() if "curb" in name.lower()}
        drainage_ids = {idx for idx, name in seg_names.items() if "drain" in name.lower()}
        manhole_ids = {idx for idx, name in seg_names.items() if "manhole" in name.lower()}

        if not curb_ids:
            raise RuntimeError(f"SEG 모델에서 curb 클래스를 찾지 못했습니다: {seg_names}")

        print(f"[VISION] SEG classes={seg_names}")
        print(
            f"[VISION] curb={sorted(curb_ids)}, drainage={sorted(drainage_ids)}, "
            f"manhole={sorted(manhole_ids)}"
        )

        with perception_lock:
            perception_state.update({
                "model_ready": True,
                "reason_text": "모델 준비 완료. 판단 활성화 대기",
                "error": None,
            })

    except Exception as exc:
        with perception_lock:
            perception_state.update({
                "model_ready": False,
                "decision": "FAULT",
                "reason_text": f"비전 모델 초기화 실패: {exc}",
                "error": str(exc),
            })
        print(f"[VISION] 모델 초기화 오류: {exc}")
        return

    next_inference_mono = 0.0
    previous_should_run = False
    previous_stopped = False
    previous_motor_pulsing = False

    while not shutdown_event.is_set():
        system = system_snapshot()
        vehicle = vehicle_snapshot()
        fresh_speed = speed_is_fresh(vehicle)
        speed = vehicle.get("speed_kmh")

        should_run = (
            bool(system["active"])
            and fresh_speed
            and speed is not None
            and float(speed) <= CREEP_MAX_KMH
        )
        stopped = should_run and float(speed) <= STOP_SPEED_EPSILON_KMH

        if not should_run:
            next_inference_mono = 0.0
            previous_should_run = False
            previous_stopped = False
            previous_motor_pulsing = False
            shutdown_event.wait(0.05)
            continue

        # Python sleep 기반 STEP 펄스의 지터를 줄이기 위해 실제 펄스 출력 중에는
        # 무거운 YOLO 추론을 잠시 멈춥니다. 전개 완료/수납 완료 직후 즉시 재추론합니다.
        motor_pulsing = motor.snapshot()["state"] in {"STARTING", "DEPLOYING", "RETRACTING"}
        if motor_pulsing:
            previous_motor_pulsing = True
            shutdown_event.wait(0.05)
            continue
        if previous_motor_pulsing:
            next_inference_mono = 0.0
        previous_motor_pulsing = False

        if not previous_should_run or (stopped and not previous_stopped):
            # 5 km/h 이하 진입 직후 및 정차 직후 즉시 한 번 추론
            next_inference_mono = 0.0

        previous_should_run = True
        previous_stopped = stopped
        now_mono = time.monotonic()

        if now_mono < next_inference_mono:
            shutdown_event.wait(min(0.05, next_inference_mono - now_mono))
            continue

        frame, frame_updated_at = latest_frame_snapshot()
        if (
            frame is None
            or frame_updated_at <= 0
            or (time.time() - frame_updated_at) > CAMERA_STALE_SEC
        ):
            camera = camera_snapshot()
            with perception_lock:
                perception_state.update({
                    "decision": "FAULT",
                    "reason_text": "최신 카메라 프레임이 없어 판단할 수 없습니다.",
                    "error": camera.get("error") or "camera_frame_stale",
                })
            next_inference_mono = time.monotonic() + 0.5
            continue

        try:
            roi_frame = crop_frame_to_roi(frame, VISION_ROI)
            frame_h, frame_w = roi_frame.shape[:2]

            seg_kwargs = prediction_kwargs(SEG_IMGSZ)
            seg_kwargs["conf"] = SEG_CONF
            seg_result = seg_model.predict(source=roi_frame, **seg_kwargs)[0]

            curb_detected = False
            curb_reachable = False
            drainage_detected = False
            manhole_detected = False
            curb_conf: float | None = None
            drainage_conf: float | None = None
            manhole_conf: float | None = None

            if seg_result.boxes is not None:
                for box in seg_result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0]) if box.conf is not None else None
                    xyxy = [float(value) for value in box.xyxy[0].tolist()]

                    if cls_id in curb_ids:
                        curb_detected = True
                        curb_conf = max(curb_conf or 0.0, confidence or 0.0)
                        if box_center_in_zone(xyxy, frame_w, frame_h, REACH_ZONE):
                            curb_reachable = True
                    elif cls_id in drainage_ids:
                        drainage_detected = True
                        drainage_conf = max(drainage_conf or 0.0, confidence or 0.0)
                    elif cls_id in manhole_ids:
                        manhole_detected = True
                        manhole_conf = max(manhole_conf or 0.0, confidence or 0.0)

            cls_result = classify_model.predict(
                source=roi_frame,
                **prediction_kwargs(CLASSIFY_IMGSZ),
            )[0]
            if cls_result.probs is None:
                raise RuntimeError("노면 분류 모델에서 probs 결과가 없습니다.")

            top1 = int(cls_result.probs.top1)
            surface_class = str(cls_result.names[top1])
            surface_conf = float(cls_result.probs.top1conf.item())
            surface_risky = surface_group(surface_class) != "SAFE"

            decision, reason = decide(
                curb_detected=curb_detected,
                drainage_detected=drainage_detected,
                manhole_detected=manhole_detected,
                surface_class=surface_class,
            )
            now = time.time()

            with perception_lock:
                perception_state.update({
                    "curb": {
                        "detected": curb_detected,
                        "reachable": curb_reachable,
                        "confidence": round(curb_conf, 3) if curb_conf is not None else None,
                    },
                    "drainage": {
                        "detected": drainage_detected,
                        "confidence": round(drainage_conf, 3) if drainage_conf is not None else None,
                    },
                    "manhole": {
                        "detected": manhole_detected,
                        "confidence": round(manhole_conf, 3) if manhole_conf is not None else None,
                    },
                    "surface": {
                        "class_name": surface_class,
                        "confidence": round(surface_conf, 3),
                        "risky": surface_risky,
                    },
                    "decision": decision,
                    "reason_text": reason,
                    "updated_at": now,
                    "error": None,
                })

            print(
                f"[VISION] speed={float(speed):.1f} "
                f"curb={curb_detected} drainage={drainage_detected} "
                f"manhole={manhole_detected} "
                f"surface={surface_class}({surface_conf:.2f}) => {decision}"
            )

        except Exception as exc:
            with perception_lock:
                perception_state.update({
                    "decision": "FAULT",
                    "reason_text": f"비전 추론 오류: {exc}",
                    "error": str(exc),
                })
            print(f"[VISION] 추론 오류: {exc}")

        next_inference_mono = time.monotonic() + VISION_INTERVAL_SEC


# ───────────────────────────── 상태머신 ─────────────────────────────
def control_loop() -> None:
    while not shutdown_event.is_set():
        now = time.time()
        system = system_snapshot()

        if not system["active"]:
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        vehicle = vehicle_snapshot()
        perception = perception_snapshot()
        motor_state = motor.snapshot()

        if not speed_is_fresh(vehicle):
            if motor_state["busy"]:
                motor.request_retract_now()
            update_system(
                phase="SPEED_SOURCE_FAULT",
                final_decision="FAULT",
                ui_mode="FAULT",
                reason_text=vehicle.get("error") or "속도 데이터가 유효하지 않습니다.",
                stopped_since=None,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        speed = float(vehicle["speed_kmh"])

        if not motor_state["ready"] or motor_state["state"] == "ERROR":
            update_system(
                phase="MOTOR_FAULT",
                final_decision="FAULT",
                ui_mode="FAULT",
                reason_text=motor_state.get("error") or "모터가 준비되지 않았습니다.",
                stopped_since=None,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        if speed > CREEP_MAX_KMH:
            if motor_state["busy"]:
                motor.request_retract_now()
            update_system(
                phase="WAITING_FOR_SLOW",
                final_decision="WAITING",
                ui_mode="NEUTRAL",
                reason_text=f"차량 속도 {speed:.1f} km/h: {CREEP_MAX_KMH:.1f} km/h 이하 대기",
                stopped_since=None,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        if perception["error"]:
            raw_decision = "FAULT"
            raw_reason = perception["reason_text"]
        elif not perception["model_ready"]:
            raw_decision = "WAITING"
            raw_reason = "비전 모델 준비 중"
        elif not perception_is_fresh(perception):
            raw_decision = "WAITING"
            raw_reason = "최신 비전 판단을 기다리고 있습니다."
        else:
            raw_decision = str(perception["decision"])
            raw_reason = str(perception["reason_text"])

        # 0 < speed <= 5: 판단은 표시하지만 모터는 절대 전개하지 않음
        if speed > STOP_SPEED_EPSILON_KMH:
            if motor_state["busy"]:
                motor.request_retract_now()
            update_system(
                phase="JUDGING_WHILE_CREEPING",
                final_decision=raw_decision,
                ui_mode=decision_to_ui(raw_decision),
                reason_text=raw_reason,
                stopped_since=None,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        # 정차 상태
        stopped_since = system.get("stopped_since")
        if stopped_since is None:
            stopped_since = now

        # 비전 오류는 과거 결과의 freshness와 무관하게 즉시 fail-safe 처리합니다.
        if raw_decision == "FAULT":
            if motor_state["busy"]:
                motor.request_retract_now()
            update_system(
                phase="VISION_FAULT",
                final_decision="FAULT",
                ui_mode="FAULT",
                reason_text=raw_reason,
                stopped_since=stopped_since,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        # 반드시 정차 이후 새로 얻은 비전 결과를 사용합니다.
        if raw_decision == "WAITING" or float(perception["updated_at"] or 0.0) < float(stopped_since):
            update_system(
                phase="STOPPED_WAITING_FRESH_VISION",
                final_decision="WAITING",
                ui_mode="NEUTRAL",
                reason_text="정차 이후의 새 카메라 판단을 기다리고 있습니다.",
                stopped_since=stopped_since,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        if raw_decision != "YELLOW":
            # 전개 후 판단이 RED/GREEN으로 바뀌면 유지 시간을 기다리지 않고 수납합니다.
            if motor_state["busy"]:
                motor.request_retract_now()
            update_system(
                phase="STOPPED_NO_DEPLOY",
                final_decision=raw_decision,
                ui_mode=decision_to_ui(raw_decision),
                reason_text=raw_reason,
                stopped_since=stopped_since,
                yellow_since=None,
            )
            shutdown_event.wait(CONTROL_PERIOD_SEC)
            continue

        # 정차 + YELLOW. 전개 지연은 YELLOW 시작이 아니라 연속 정차 시작부터 계산합니다.
        yellow_since = system.get("yellow_since")
        if yellow_since is None:
            yellow_since = now
        remaining = stop_delay_remaining(stopped_since, now)

        if motor_state["busy"]:
            phase = f"MOTOR_{motor_state['state']}"
            reason = "발판 자동 전개/수납 사이클이 진행 중입니다."
        elif system.get("deploy_latched"):
            phase = "STOPPED_YELLOW_CYCLE_DONE"
            reason = "이 판단 세션의 발판 사이클이 이미 실행되었습니다."
        else:
            phase = "STOPPED_YELLOW_DELAY" if remaining > 0 else "STOPPED_YELLOW_READY"
            reason = (
                f"정차 유지 중: {remaining:.2f}초 후 발판 자동 전개"
                if remaining > 0
                else f"{STOP_DEPLOY_DELAY_SEC:g}초 연속 정차 + YELLOW 조건 충족: 발판 자동 전개 준비"
            )

        update_system(
            phase=phase,
            final_decision="YELLOW",
            ui_mode="YELLOW",
            reason_text=reason,
            stopped_since=stopped_since,
            yellow_since=yellow_since,
        )

        if remaining <= 0.0 and not system.get("deploy_latched"):
            if not motor_state["ready"]:
                update_system(
                    phase="MOTOR_FAULT",
                    final_decision="FAULT",
                    ui_mode="FAULT",
                    reason_text=motor_state.get("error") or "모터가 준비되지 않았습니다.",
                    yellow_since=None,
                )
            else:
                ok, motor_reason = motor.request_cycle("auto_yellow_stop")
                if ok:
                    update_system(
                        phase="AUTO_CYCLE_TRIGGERED",
                        deploy_latched=True,
                        last_motor_action="AUTO_CYCLE:auto_yellow_stop",
                        reason_text="발판 1.5회전(540도) 전개 후 5초 유지, 자동 수납을 시작했습니다.",
                    )
                elif motor_reason not in {"motor_busy", "motor_thread_busy"}:
                    update_system(
                        phase="MOTOR_FAULT",
                        final_decision="FAULT",
                        ui_mode="FAULT",
                        reason_text=motor_reason,
                        yellow_since=None,
                    )

        shutdown_event.wait(CONTROL_PERIOD_SEC)


# ───────────────────────────── HTTP API ─────────────────────────────
@app.route("/api/start", methods=["POST"])
def start_assessment():
    motor_state = motor.snapshot()
    update_system(
        active=True,
        phase="ARMED_WAIT_SPEED",
        final_decision="WAITING",
        ui_mode="NEUTRAL",
        reason_text=f"판단 활성화됨. {CREEP_MAX_KMH:.1f} km/h 이하 대기",
        stopped_since=None,
        yellow_since=None,
        # 전개 중 페이지를 다시 켠 경우 중복 사이클 방지
        deploy_latched=bool(motor_state["busy"]),
        last_motor_action=None,
    )
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_assessment():
    # 시스템을 끄면 남은 5초 대기를 건너뛰고 가능한 즉시 수납 단계로 이동합니다.
    motor.request_retract_now()
    update_system(
        active=False,
        phase="IDLE",
        final_decision="IDLE",
        ui_mode="NEUTRAL",
        reason_text="판단 종료",
        stopped_since=None,
        yellow_since=None,
        deploy_latched=False,
    )
    return jsonify({"ok": True})


@app.route("/api/debug", methods=["GET", "POST"])
def debug_speed():
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        if "speed_kmh" not in payload:
            return jsonify({"ok": False, "reason": "speed_kmh_required"}), 400

        try:
            speed = float(payload["speed_kmh"])
        except (TypeError, ValueError):
            return jsonify({"ok": False, "reason": "invalid_speed"}), 400

        if not math.isfinite(speed) or speed < 0.0 or speed > 300.0:
            return jsonify({"ok": False, "reason": "speed_out_of_range"}), 400

        with vehicle_lock:
            vehicle_state.update({
                "connected": True,
                "speed_kmh": speed,
                "updated_at": time.time(),
                "source": "manual_debug",
                "error": None,
            })

    vehicle = vehicle_snapshot()
    return jsonify({
        "ok": True,
        "speed_kmh": vehicle["speed_kmh"],
        "source": vehicle["source"],
    })



@app.route("/api/vehicle", methods=["GET"])
def get_vehicle():
    vehicle = vehicle_snapshot()
    vehicle["fresh"] = speed_is_fresh(vehicle)
    if vehicle.get("updated_at"):
        vehicle["data_age_sec"] = round(time.time() - float(vehicle["updated_at"]), 3)
    else:
        vehicle["data_age_sec"] = None
    return jsonify(vehicle)


@app.route("/api/vision", methods=["GET"])
def get_vision():
    vehicle = vehicle_snapshot()
    perception = perception_snapshot()
    system = system_snapshot()
    motor_state = motor.snapshot()
    camera = camera_snapshot()

    return jsonify({
        "door": DOOR_ID,
        "mode": RUN_MODE,
        "active": system["active"],
        "phase": system["phase"],
        "final_decision": system["final_decision"],
        "ui_mode": system["ui_mode"],
        "reason_text": system["reason_text"],
        "curb": perception["curb"],
        "drainage": perception["drainage"],
        "manhole": perception["manhole"],
        "surface": perception["surface"],
        "perception_error": perception["error"],
        "updated_at": perception["updated_at"],
        "speed_kmh": vehicle.get("speed_kmh"),
        "speed_source": vehicle.get("source"),
        "speed_fresh": speed_is_fresh(vehicle),
        "obd_connected": vehicle.get("connected") if RUN_MODE == "OBD" else None,
        "vehicle_error": vehicle.get("error"),
        "deploy_latched": system["deploy_latched"],
        "motor": motor_state,
        "camera": camera,
    })


@app.route("/api/system", methods=["GET"])
def get_system():
    vehicle = vehicle_snapshot()
    perception = perception_snapshot()
    return jsonify({
        "mode": RUN_MODE,
        "system": system_snapshot(),
        "vehicle": {**vehicle, "fresh": speed_is_fresh(vehicle)},
        "perception": {
            **perception,
            "fresh": perception_is_fresh(perception),
        },
        "camera": camera_snapshot(),
        "motor": motor.snapshot(),
    })


@app.route("/api/status", methods=["GET"])
def get_ramp_status():
    return jsonify(motor.snapshot())


@app.route("/api/deploy", methods=["POST"])
def deploy_manual():
    system = system_snapshot()
    vehicle = vehicle_snapshot()
    perception = perception_snapshot()

    if not system["active"]:
        return jsonify({"ok": False, "reason": "assessment_not_active"}), 409
    if not speed_is_fresh(vehicle):
        return jsonify({"ok": False, "reason": "speed_not_fresh"}), 503
    if float(vehicle["speed_kmh"]) > STOP_SPEED_EPSILON_KMH:
        return jsonify({
            "ok": False,
            "reason": "vehicle_not_stopped",
            "speed_kmh": vehicle["speed_kmh"],
        }), 409
    if not perception_is_fresh(perception):
        return jsonify({"ok": False, "reason": "vision_not_fresh"}), 409
    if system.get("stopped_since") is None or float(perception["updated_at"]) < float(system["stopped_since"]):
        return jsonify({"ok": False, "reason": "post_stop_vision_not_ready"}), 409
    remaining = stop_delay_remaining(system.get("stopped_since"))
    if remaining > 0.0:
        return jsonify({
            "ok": False,
            "reason": "stop_delay_not_elapsed",
            "remaining_sec": round(remaining, 3),
        }), 409
    if system["final_decision"] != "YELLOW":
        return jsonify({
            "ok": False,
            "reason": "decision_not_yellow",
            "decision": system["final_decision"],
        }), 409
    if system.get("deploy_latched"):
        return jsonify({"ok": False, "reason": "cycle_already_triggered_this_session"}), 409

    ok, reason = motor.request_cycle("manual_api")
    if not ok:
        return jsonify({"ok": False, "reason": reason}), 409

    update_system(
        deploy_latched=True,
        phase="MANUAL_CYCLE_TRIGGERED",
        last_motor_action="AUTO_CYCLE:manual_api",
        reason_text="수동 버튼으로 자동 전개/수납 사이클을 시작했습니다.",
    )
    return jsonify({"ok": True})


@app.route("/api/retract", methods=["POST"])
def retract_early():
    ok, reason = motor.request_retract_now()
    if not ok:
        return jsonify({"ok": False, "reason": reason}), 409
    update_system(
        last_motor_action="EARLY_RETRACT:manual_api",
        reason_text="즉시 수납 요청을 전달했습니다.",
    )
    return jsonify({"ok": True, "reason": reason})


@app.route("/api/health", methods=["GET"])
def health():
    vehicle = vehicle_snapshot()
    perception = perception_snapshot()
    motor_state = motor.snapshot()
    return jsonify({
        "ok": bool(motor_state["ready"] and perception["model_ready"]),
        "mode": RUN_MODE,
        "speed_fresh": speed_is_fresh(vehicle),
        "model_ready": perception["model_ready"],
        "motor_ready": motor_state["ready"],
        "camera_connected": camera_snapshot()["connected"],
    })


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), HTML_FILE_NAME)


def shutdown() -> None:
    shutdown_event.set()
    motor.close()


atexit.register(shutdown)


def main() -> None:
    print(f"[SERVER] mode={RUN_MODE}, html={HTML_FILE_NAME}")
    print(f"[SERVER] SEG={SEG_MODEL_PATH}")
    print(f"[SERVER] CLASSIFY={CLASSIFY_MODEL_PATH}")

    threads = [
        threading.Thread(target=camera_loop, name="camera-loop", daemon=True),
        threading.Thread(target=vision_loop, name="vision-loop", daemon=True),
        threading.Thread(target=control_loop, name="control-loop", daemon=True),

    ]
    for thread in threads:
        thread.start()

    app.run(host="0.0.0.0", port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
