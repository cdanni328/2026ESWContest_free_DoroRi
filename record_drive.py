#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_drive.py
================
Raspberry Pi 5 + USB webcam + USB ELM327 OBD-II 주행 데이터 녹화기.

저장 항목
--------
records/YYYYMMDD_HHMMSS/
├── drive.mp4          # 우측 상단에 OBD 속도가 합성된 주행영상
├── frame_sync.csv     # 각 영상 프레임의 timestamp + 당시 사용한 OBD 속도
├── obd_raw.csv        # 실제 OBD SPEED PID 응답 원본 로그
└── metadata.json      # 녹화 설정/카메라/OBD 정보

핵심 동기화 방식
---------------
- 카메라 프레임과 OBD 응답 모두 time.monotonic_ns()를 사용합니다.
- OBD는 별도 스레드에서 계속 SPEED PID를 읽습니다.
- 영상 프레임을 획득한 순간, 그 시점 이전의 "가장 최근 OBD 속도"를 사용합니다.
- OBD 조회가 느리거나 잠시 끊겨도 카메라 녹화 루프는 멈추지 않습니다.

기본 실행
--------
    python3 record_drive.py

OBD 포트 지정
-------------
    python3 record_drive.py --obd-port /dev/ttyUSB0

카메라/해상도/FPS 지정
---------------------
    python3 record_drive.py --camera 0 --width 1280 --height 720 --fps 30

미리보기 없이 녹화
-----------------
    python3 record_drive.py --no-preview

종료
----
- 미리보기 창에서 q 또는 ESC
- 또는 터미널에서 Ctrl+C
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV가 없습니다.\n"
        "Raspberry Pi OS/Ubuntu 예: sudo apt install python3-opencv"
    ) from exc

try:
    import obd
except ImportError as exc:
    raise SystemExit(
        "python-OBD가 없습니다.\n"
        "설치: python3 -m pip install obd"
    ) from exc


# ───────────────────────────── 공유 OBD 상태 ─────────────────────────────

@dataclass
class ObdSnapshot:
    connected: bool = False
    speed_kmh: Optional[float] = None
    sample_mono_ns: Optional[int] = None
    status: str = "INIT"
    error: Optional[str] = None
    port: Optional[str] = None


class SharedObdState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = ObdSnapshot()

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)

    def snapshot(self) -> ObdSnapshot:
        with self._lock:
            return ObdSnapshot(
                connected=self._state.connected,
                speed_kmh=self._state.speed_kmh,
                sample_mono_ns=self._state.sample_mono_ns,
                status=self._state.status,
                error=self._state.error,
                port=self._state.port,
            )


# ───────────────────────────── 보조 함수 ─────────────────────────────

def parse_camera_source(value: str) -> Union[int, str]:
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def safe_status_text(value: object) -> str:
    try:
        return str(value)
    except Exception:
        return "UNKNOWN"


def draw_speed_overlay(
    frame,
    speed_kmh: Optional[float],
    valid: bool,
    age_ms: Optional[float],
) -> None:
    """영상 우측 상단에 현재 OBD 속도를 합성합니다."""
    h, w = frame.shape[:2]

    if valid and speed_kmh is not None:
        main_text = f"{speed_kmh:.1f} km/h"
        sub_text = "OBD-II SPEED"
    else:
        main_text = "--.- km/h"
        sub_text = "OBD-II NO DATA"

    font = cv2.FONT_HERSHEY_SIMPLEX
    main_scale = max(0.75, min(1.15, w / 1100.0))
    sub_scale = max(0.40, min(0.58, w / 2200.0))
    main_thickness = max(2, int(round(main_scale * 2)))
    sub_thickness = 1

    (main_w, main_h), main_base = cv2.getTextSize(
        main_text, font, main_scale, main_thickness
    )
    (sub_w, sub_h), sub_base = cv2.getTextSize(
        sub_text, font, sub_scale, sub_thickness
    )

    margin = max(12, int(w * 0.015))
    pad_x = max(12, int(w * 0.012))
    pad_y = max(9, int(h * 0.012))
    gap = max(5, int(h * 0.007))

    box_w = max(main_w, sub_w) + 2 * pad_x
    box_h = (
        pad_y
        + sub_h
        + sub_base
        + gap
        + main_h
        + main_base
        + pad_y
    )

    x1 = max(0, w - margin - box_w)
    y1 = margin
    x2 = min(w - 1, x1 + box_w)
    y2 = min(h - 1, y1 + box_h)

    # 가독성을 위한 반투명 검정 배경
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)

    text_x = x1 + pad_x
    sub_y = y1 + pad_y + sub_h
    main_y = sub_y + sub_base + gap + main_h

    # 정상 데이터는 흰색, 끊긴 데이터는 밝은 회색으로 표시
    main_color = (255, 255, 255) if valid else (190, 190, 190)
    sub_color = (220, 220, 220)

    cv2.putText(
        frame,
        sub_text,
        (text_x, sub_y),
        font,
        sub_scale,
        sub_color,
        sub_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        main_text,
        (text_x, main_y),
        font,
        main_scale,
        main_color,
        main_thickness,
        cv2.LINE_AA,
    )

    # 디버깅에 유용하지만 영상 화면을 복잡하게 만들지 않도록
    # 유효하지 않을 때만 데이터 age를 작게 표시합니다.
    if not valid and age_ms is not None:
        age_text = f"last: {age_ms / 1000.0:.1f}s ago"
        age_scale = max(0.35, min(0.48, w / 2500.0))
        cv2.putText(
            frame,
            age_text,
            (text_x, min(h - 5, y2 + int(18 * max(1.0, h / 720.0)))),
            font,
            age_scale,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )


def open_camera(
    source: Union[int, str],
    width: int,
    height: int,
    fps: float,
):
    """Linux에서는 V4L2를 우선 사용하고 실패하면 일반 backend로 재시도합니다."""
    cap = None

    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = None

    if cap is None:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {source}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def obd_worker(
    *,
    stop_event: threading.Event,
    state: SharedObdState,
    raw_csv_writer,
    raw_csv_file,
    csv_lock: threading.Lock,
    start_mono_ns: int,
    port: Optional[str],
    baudrate: Optional[int],
    fast: bool,
    timeout_sec: float,
    poll_sec: float,
    retry_sec: float,
) -> None:
    """
    OBD SPEED PID를 별도 스레드에서 읽습니다.

    query()가 카메라 녹화를 블로킹하지 않도록 완전히 분리되어 있습니다.
    """
    while not stop_event.is_set():
        connection = None

        try:
            kwargs = {
                "fast": fast,
                "timeout": timeout_sec,
            }
            if baudrate is not None:
                kwargs["baudrate"] = baudrate

            print(f"[OBD] 연결 시도: {port or 'AUTO'}")
            connection = obd.OBD(port, **kwargs)

            if not connection.is_connected():
                raise RuntimeError(
                    f"차량 OBD 연결 실패: {safe_status_text(connection.status())}"
                )

            status = safe_status_text(connection.status())
            state.update(
                connected=True,
                status=status,
                error=None,
                port=port,
            )
            print(f"[OBD] 연결 완료: {status}")

            null_count = 0

            while not stop_event.is_set():
                # 표준 Vehicle Speed PID = Mode 01 PID 0D.
                # 일부 ELM327 clone은 지원 PID 테이블 판별이 틀릴 수 있어 force=True.
                response = connection.query(obd.commands.SPEED, force=True)
                sample_ns = time.monotonic_ns()

                if response.is_null() or response.value is None:
                    null_count += 1
                    error = "SPEED response is null"

                    with csv_lock:
                        raw_csv_writer.writerow([
                            sample_ns,
                            f"{(sample_ns - start_mono_ns) / 1e9:.9f}",
                            iso_now(),
                            "",
                            "NULL",
                            safe_status_text(connection.status()),
                            error,
                        ])
                        raw_csv_file.flush()

                    state.update(
                        connected=True,
                        status=safe_status_text(connection.status()),
                        error=error,
                    )

                    if null_count >= 3:
                        raise RuntimeError("Vehicle Speed 응답이 연속 3회 없습니다.")

                    stop_event.wait(poll_sec)
                    continue

                null_count = 0
                speed_kmh = float(response.value.to("kph").magnitude)

                if not math.isfinite(speed_kmh) or speed_kmh < 0.0:
                    raise RuntimeError(f"비정상 Vehicle Speed 값: {speed_kmh}")

                status = safe_status_text(connection.status())
                state.update(
                    connected=True,
                    speed_kmh=speed_kmh,
                    sample_mono_ns=sample_ns,
                    status=status,
                    error=None,
                    port=port,
                )

                with csv_lock:
                    raw_csv_writer.writerow([
                        sample_ns,
                        f"{(sample_ns - start_mono_ns) / 1e9:.9f}",
                        iso_now(),
                        f"{speed_kmh:.6f}",
                        "OK",
                        status,
                        "",
                    ])
                    raw_csv_file.flush()

                stop_event.wait(poll_sec)

        except Exception as exc:
            message = str(exc)
            print(f"[OBD] 오류: {message}")
            state.update(
                connected=False,
                speed_kmh=None,
                sample_mono_ns=None,
                status="ERROR",
                error=message,
            )

            now_ns = time.monotonic_ns()
            with csv_lock:
                raw_csv_writer.writerow([
                    now_ns,
                    f"{(now_ns - start_mono_ns) / 1e9:.9f}",
                    iso_now(),
                    "",
                    "ERROR",
                    "",
                    message,
                ])
                raw_csv_file.flush()

        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

        stop_event.wait(retry_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="USB webcam 영상 + OBD-II 차량 속도 동기화 녹화"
    )

    parser.add_argument(
        "--camera",
        default="0",
        help="카메라 번호 또는 경로. 기본값: 0 (예: 0, /dev/video0)",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)

    parser.add_argument(
        "--obd-port",
        default=None,
        help="예: /dev/ttyUSB0. 생략하면 python-OBD 자동 탐색",
    )
    parser.add_argument(
        "--obd-baudrate",
        type=int,
        default=None,
        help="보통 생략. 필요할 때만 ELM327 baudrate 강제 지정",
    )
    parser.add_argument(
        "--obd-poll-sec",
        type=float,
        default=0.2,
        help="OBD SPEED 질의 간격. 기본 0.2초",
    )
    parser.add_argument(
        "--obd-timeout-sec",
        type=float,
        default=0.5,
        help="OBD 명령 timeout. 기본 0.5초",
    )
    parser.add_argument(
        "--obd-retry-sec",
        type=float,
        default=2.0,
        help="OBD 재연결 간격. 기본 2초",
    )
    parser.add_argument(
        "--obd-stale-sec",
        type=float,
        default=1.5,
        help="마지막 속도 표본이 이 시간보다 오래되면 영상에 NO DATA 표시",
    )
    parser.add_argument(
        "--obd-slow",
        action="store_true",
        help="python-OBD fast=False 사용. clone ELM327 통신이 불안정할 때 사용",
    )

    parser.add_argument(
        "--output-root",
        default="records",
        help="녹화 폴더 상위 경로. 기본: ./records",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="녹화 세션 폴더명. 생략하면 현재 날짜/시간",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="OpenCV 미리보기 창을 띄우지 않음",
    )

    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width와 --height는 1 이상이어야 합니다.")
    if args.fps <= 0:
        raise SystemExit("--fps는 0보다 커야 합니다.")
    if args.obd_poll_sec <= 0:
        raise SystemExit("--obd-poll-sec는 0보다 커야 합니다.")
    if args.obd_stale_sec <= 0:
        raise SystemExit("--obd-stale-sec는 0보다 커야 합니다.")

    session_name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_root).expanduser().resolve() / session_name
    session_dir.mkdir(parents=True, exist_ok=False)

    video_path = session_dir / "drive.mp4"
    frame_csv_path = session_dir / "frame_sync.csv"
    obd_csv_path = session_dir / "obd_raw.csv"
    metadata_path = session_dir / "metadata.json"

    print(f"[SAVE] {session_dir}")

    stop_event = threading.Event()
    obd_state = SharedObdState()
    csv_lock = threading.Lock()

    def request_stop(*_args) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    cap = None
    writer = None
    frame_csv_file = None
    obd_csv_file = None
    obd_thread = None

    start_wall_iso = iso_now()
    start_mono_ns = time.monotonic_ns()

    frame_count = 0
    first_frame_ns: Optional[int] = None
    last_frame_ns: Optional[int] = None

    try:
        camera_source = parse_camera_source(args.camera)
        print(
            f"[CAMERA] source={camera_source}, "
            f"request={args.width}x{args.height}@{args.fps:.3f}"
        )

        cap = open_camera(
            camera_source,
            args.width,
            args.height,
            args.fps,
        )

        # 실제 프레임을 하나 받아 VideoWriter 크기를 확정합니다.
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("첫 카메라 프레임을 읽지 못했습니다.")

        actual_h, actual_w = frame.shape[:2]
        camera_reported_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        # 카메라 드라이버가 0이나 비정상 FPS를 보고하면 요청 FPS를 사용합니다.
        if (
            math.isfinite(camera_reported_fps)
            and 1.0 <= camera_reported_fps <= 240.0
        ):
            writer_fps = camera_reported_fps
        else:
            writer_fps = args.fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            writer_fps,
            (actual_w, actual_h),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"VideoWriter를 열 수 없습니다: {video_path}\n"
                "OpenCV FFmpeg/GStreamer codec 구성을 확인하세요."
            )

        print(
            f"[CAMERA] actual={actual_w}x{actual_h}, "
            f"reported_fps={camera_reported_fps:.3f}, "
            f"video_fps={writer_fps:.3f}"
        )

        frame_csv_file = frame_csv_path.open(
            "w", newline="", encoding="utf-8"
        )
        frame_csv_writer = csv.writer(frame_csv_file)
        frame_csv_writer.writerow([
            "frame_index",
            "frame_mono_ns",
            "elapsed_s",
            "wall_time_iso",
            "speed_kmh",
            "obd_valid",
            "obd_age_ms",
            "obd_connected",
            "obd_status",
            "obd_error",
        ])
        frame_csv_file.flush()

        obd_csv_file = obd_csv_path.open(
            "w", newline="", encoding="utf-8"
        )
        obd_csv_writer = csv.writer(obd_csv_file)
        obd_csv_writer.writerow([
            "sample_mono_ns",
            "elapsed_s",
            "wall_time_iso",
            "speed_kmh",
            "result",
            "obd_status",
            "error",
        ])
        obd_csv_file.flush()

        obd_thread = threading.Thread(
            target=obd_worker,
            kwargs={
                "stop_event": stop_event,
                "state": obd_state,
                "raw_csv_writer": obd_csv_writer,
                "raw_csv_file": obd_csv_file,
                "csv_lock": csv_lock,
                "start_mono_ns": start_mono_ns,
                "port": args.obd_port,
                "baudrate": args.obd_baudrate,
                "fast": not args.obd_slow,
                "timeout_sec": args.obd_timeout_sec,
                "poll_sec": args.obd_poll_sec,
                "retry_sec": args.obd_retry_sec,
            },
            name="obd-speed-reader",
            daemon=True,
        )
        obd_thread.start()

        metadata = {
            "format_version": 1,
            "created_at": start_wall_iso,
            "clock": "time.monotonic_ns",
            "files": {
                "video": video_path.name,
                "frame_sync": frame_csv_path.name,
                "obd_raw": obd_csv_path.name,
            },
            "camera": {
                "source": str(camera_source),
                "requested_width": args.width,
                "requested_height": args.height,
                "requested_fps": args.fps,
                "actual_width": actual_w,
                "actual_height": actual_h,
                "reported_fps": camera_reported_fps,
                "video_writer_fps": writer_fps,
                "codec": "mp4v",
            },
            "obd": {
                "port_requested": args.obd_port,
                "baudrate": args.obd_baudrate,
                "fast": not args.obd_slow,
                "poll_sec": args.obd_poll_sec,
                "timeout_sec": args.obd_timeout_sec,
                "retry_sec": args.obd_retry_sec,
                "stale_sec": args.obd_stale_sec,
                "command": "SPEED",
                "standard_pid": "01 0D",
            },
            "note": (
                "frame_sync.csv의 각 프레임은 해당 프레임 획득 시점 이전에 "
                "수신된 가장 최근 OBD SPEED 표본과 연결되어 있습니다."
            ),
        }
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("[REC] 녹화 시작")
        if not args.no_preview:
            print("[REC] q 또는 ESC: 종료 / Ctrl+C: 종료")
        else:
            print("[REC] Ctrl+C: 종료")

        # 첫 read에서 얻은 frame도 버리지 않고 기록합니다.
        pending_frame = frame

        while not stop_event.is_set():
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                ok = True
            else:
                ok, frame = cap.read()

            frame_ns = time.monotonic_ns()

            if not ok or frame is None:
                print("[CAMERA] 프레임 획득 실패")
                stop_event.wait(0.02)
                continue

            if frame.shape[1] != actual_w or frame.shape[0] != actual_h:
                frame = cv2.resize(
                    frame,
                    (actual_w, actual_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            if first_frame_ns is None:
                first_frame_ns = frame_ns
            last_frame_ns = frame_ns

            snapshot = obd_state.snapshot()

            obd_age_ms: Optional[float] = None
            if snapshot.sample_mono_ns is not None:
                obd_age_ms = max(
                    0.0,
                    (frame_ns - snapshot.sample_mono_ns) / 1e6,
                )

            obd_valid = (
                snapshot.connected
                and snapshot.speed_kmh is not None
                and snapshot.sample_mono_ns is not None
                and obd_age_ms is not None
                and obd_age_ms <= args.obd_stale_sec * 1000.0
            )

            speed_for_frame = snapshot.speed_kmh if obd_valid else None

            # 녹화본에 속도를 직접 합성합니다.
            output_frame = frame.copy()
            draw_speed_overlay(
                output_frame,
                speed_for_frame,
                obd_valid,
                obd_age_ms,
            )

            writer.write(output_frame)

            elapsed_s = (frame_ns - start_mono_ns) / 1e9

            frame_csv_writer.writerow([
                frame_count,
                frame_ns,
                f"{elapsed_s:.9f}",
                iso_now(),
                "" if speed_for_frame is None else f"{speed_for_frame:.6f}",
                1 if obd_valid else 0,
                "" if obd_age_ms is None else f"{obd_age_ms:.3f}",
                1 if snapshot.connected else 0,
                snapshot.status,
                snapshot.error or "",
            ])

            # 프레임마다 디스크 flush할 필요는 없으므로 약 1초마다 수행합니다.
            if frame_count % max(1, int(round(writer_fps))) == 0:
                frame_csv_file.flush()

            if not args.no_preview:
                cv2.imshow("DORORI Drive Recorder", output_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop_event.set()
                    break

            frame_count += 1

        print("[REC] 종료 처리 중")

    finally:
        stop_event.set()

        if obd_thread is not None:
            obd_thread.join(timeout=3.0)

        if writer is not None:
            writer.release()

        if cap is not None:
            cap.release()

        if not args.no_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        if frame_csv_file is not None:
            frame_csv_file.flush()
            frame_csv_file.close()

        if obd_csv_file is not None:
            with csv_lock:
                obd_csv_file.flush()
                obd_csv_file.close()

        # 녹화 완료 후 metadata에 실제 결과를 추가합니다.
        try:
            metadata = {}
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            duration_s = None
            measured_capture_fps = None

            if (
                first_frame_ns is not None
                and last_frame_ns is not None
                and last_frame_ns > first_frame_ns
                and frame_count > 1
            ):
                duration_s = (last_frame_ns - first_frame_ns) / 1e9
                measured_capture_fps = (frame_count - 1) / duration_s

            metadata["finished_at"] = iso_now()
            metadata["result"] = {
                "frames_written": frame_count,
                "capture_duration_s": duration_s,
                "measured_capture_fps": measured_capture_fps,
            }

            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"[WARN] metadata 최종 저장 실패: {exc}")

    print(f"[DONE] 영상:       {video_path}")
    print(f"[DONE] 프레임 로그: {frame_csv_path}")
    print(f"[DONE] OBD 원본:    {obd_csv_path}")
    print(f"[DONE] 메타데이터:  {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
