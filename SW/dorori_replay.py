#!/usr/bin/env python3
"""렌트카 없이 디버깅용: record_drive 녹화본으로 DORORI 웹·판단·모터 상태머신을 재실행한다."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


REQUIRED_COLUMNS = {
    "frame_index",
    "elapsed_s",
    "speed_kmh",
    "obd_valid",
    "obd_connected",
    "obd_status",
    "obd_error",
}


class ReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayRow:
    frame_index: int
    elapsed_s: float
    speed_kmh: float | None
    obd_valid: bool
    obd_connected: bool
    obd_status: str
    obd_error: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="record_drive 영상·OBD 속도 동기 재생 서버"
    )
    parser.add_argument("record_dir", type=Path, help="drive.mp4가 있는 녹화 폴더")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="파일과 동기화 데이터만 검사하고 서버는 실행하지 않음",
    )
    parser.add_argument(
        "--allow-real-motor",
        action="store_true",
        help="위험: 확인 문구 입력 후 실제 GPIO 모터 동작 허용",
    )
    return parser


def parse_flag(value: str, name: str, line_number: int) -> bool:
    if value not in {"0", "1"}:
        raise ReplayError(
            f"CSV {line_number}행의 {name}은 0 또는 1이어야 합니다: {value!r}"
        )
    return value == "1"


def load_rows(csv_path: Path) -> list[ReplayRow]:
    try:
        csv_file = csv_path.open(newline="", encoding="utf-8-sig")
    except OSError as exc:
        raise ReplayError(f"CSV를 열 수 없습니다: {csv_path}: {exc}") from exc

    rows: list[ReplayRow] = []
    previous_elapsed: float | None = None
    with csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ReplayError(f"frame_sync.csv 필수 열이 없습니다: {', '.join(missing)}")

        for expected_index, raw in enumerate(reader):
            line_number = expected_index + 2
            try:
                frame_index = int(raw["frame_index"])
                elapsed_s = float(raw["elapsed_s"])
            except (TypeError, ValueError) as exc:
                raise ReplayError(
                    f"CSV {line_number}행의 프레임 번호/시간이 잘못되었습니다."
                ) from exc

            if frame_index != expected_index:
                raise ReplayError(
                    f"CSV {line_number}행 frame_index 불연속: "
                    f"expected={expected_index}, actual={frame_index}"
                )
            if not math.isfinite(elapsed_s):
                raise ReplayError(f"CSV {line_number}행 elapsed_s가 유한수가 아닙니다.")
            if previous_elapsed is not None and elapsed_s <= previous_elapsed:
                raise ReplayError(
                    f"CSV {line_number}행 elapsed_s가 증가하지 않습니다: {elapsed_s}"
                )

            obd_valid = parse_flag(raw["obd_valid"], "obd_valid", line_number)
            obd_connected = parse_flag(
                raw["obd_connected"], "obd_connected", line_number
            )
            speed_kmh: float | None = None
            if obd_valid:
                try:
                    speed_kmh = float(raw["speed_kmh"])
                except (TypeError, ValueError) as exc:
                    raise ReplayError(
                        f"CSV {line_number}행의 유효 속도 값이 없습니다."
                    ) from exc
                if not math.isfinite(speed_kmh) or speed_kmh < 0.0:
                    raise ReplayError(
                        f"CSV {line_number}행의 속도가 비정상입니다: {speed_kmh}"
                    )

            rows.append(
                ReplayRow(
                    frame_index=frame_index,
                    elapsed_s=elapsed_s,
                    speed_kmh=speed_kmh,
                    obd_valid=obd_valid,
                    obd_connected=obd_connected,
                    obd_status=raw["obd_status"],
                    obd_error=raw["obd_error"],
                )
            )
            previous_elapsed = elapsed_s

    if not rows:
        raise ReplayError("frame_sync.csv에 프레임 데이터가 없습니다.")
    return rows


def inspect_video(video_path: Path, *, decode_all: bool) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ReplayError(f"영상을 열 수 없습니다: {video_path}")

    try:
        if not decode_all:
            reported = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            return max(0, reported)

        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                raise ReplayError(f"영상 {count}번 프레임이 비어 있습니다.")
            count += 1
        return count
    finally:
        cap.release()


def validate_recording(
    record_dir: Path,
    *,
    decode_all: bool,
) -> tuple[Path, Path, list[ReplayRow], int]:
    record_dir = record_dir.expanduser().resolve()
    if not record_dir.is_dir():
        raise ReplayError(f"녹화 폴더가 없습니다: {record_dir}")

    video_path = record_dir / "drive.mp4"
    csv_path = record_dir / "frame_sync.csv"
    for path in (video_path, csv_path):
        if not path.is_file():
            raise ReplayError(f"필수 파일이 없습니다: {path}")

    rows = load_rows(csv_path)
    video_frames = inspect_video(video_path, decode_all=decode_all)
    if video_frames > 0 and video_frames != len(rows):
        raise ReplayError(
            f"영상/CSV 프레임 수 불일치: video={video_frames}, csv={len(rows)}"
        )
    return video_path, csv_path, rows, video_frames


def print_validation(
    video_path: Path,
    csv_path: Path,
    rows: list[ReplayRow],
    video_frames: int,
) -> None:
    duration_s = rows[-1].elapsed_s - rows[0].elapsed_s
    valid_count = sum(row.obd_valid for row in rows)
    print(f"[VALID] video={video_path}")
    print(f"[VALID] frame_sync={csv_path}")
    print(f"[VALID] video_frames={video_frames or 'unknown'}")
    print(f"[VALID] csv_rows={len(rows)}")
    print(f"[VALID] replay_duration_s={duration_s:.6f}")
    print(f"[VALID] obd_valid={valid_count}, obd_invalid={len(rows) - valid_count}")


def confirm_real_motor() -> bool:
    if not sys.stdin.isatty():
        print(
            "실제 모터 허용은 대화형 터미널에서만 가능합니다.",
            file=sys.stderr,
        )
        return False
    try:
        answer = input(
            "[DANGER] Replay 판단으로 실제 모터가 움직입니다. "
            "계속하려면 REAL MOTOR 입력: "
        )
    except (EOFError, KeyboardInterrupt):
        print("\n실제 모터 허용을 취소했습니다.", file=sys.stderr)
        return False
    if answer != "REAL MOTOR":
        print(
            "확인 문구가 일치하지 않아 실제 모터 허용을 취소했습니다.",
            file=sys.stderr,
        )
        return False
    return True


def run_server(video_path: Path, csv_path: Path, rows: list[ReplayRow]) -> None:
    # 모터 설정은 dorori_obd import 전에 main()에서 확정됩니다.
    import dorori_obd as core

    core.RUN_MODE = "REPLAY"

    with core.vehicle_lock:
        core.vehicle_state.update(
            {
                "connected": False,
                "speed_kmh": None,
                "updated_at": 0.0,
                "source": "replay",
                "port": str(csv_path),
                "status": "REPLAY_READY",
                "error": None,
            }
        )
    with core.camera_lock:
        core.camera_state.update(
            {
                "connected": False,
                "updated_at": 0.0,
                "error": "Replay 시작 버튼 대기",
            }
        )

    def finish_replay(message: str, *, failed: bool = False) -> None:
        core.motor.request_retract_now()
        core.update_system(
            active=False,
            phase="REPLAY_ERROR" if failed else "REPLAY_FINISHED",
            final_decision="FAULT" if failed else core.system_snapshot()["final_decision"],
            ui_mode="FAULT" if failed else core.system_snapshot()["ui_mode"],
            reason_text=message,
            stopped_since=None,
            yellow_since=None,
        )
        with core.vehicle_lock:
            core.vehicle_state.update(
                {
                    "connected": False,
                    "speed_kmh": None,
                    "source": "replay",
                    "status": "REPLAY_ERROR" if failed else "REPLAY_FINISHED",
                    "error": message,
                }
            )
        with core.camera_lock:
            core.camera_state.update({"connected": False, "error": message})

    def stop_replay() -> None:
        print("[REPLAY] 웹 중지 요청으로 재생을 종료합니다.")
        with core.vehicle_lock:
            core.vehicle_state.update(
                {
                    "connected": False,
                    "speed_kmh": None,
                    "status": "REPLAY_STOPPED",
                    "error": "웹 중지 요청",
                }
            )
        with core.camera_lock:
            core.camera_state.update(
                {"connected": False, "error": "웹 중지 요청"}
            )

    def replay_loop() -> None:
        cap = None
        try:
            print("[REPLAY] 모델 준비와 웹 시작 버튼을 기다립니다.")
            while not core.shutdown_event.is_set():
                perception = core.perception_snapshot()
                if perception["error"] and not perception["model_ready"]:
                    raise ReplayError(f"비전 모델 준비 실패: {perception['error']}")
                if perception["model_ready"] and core.system_snapshot()["active"]:
                    break
                core.shutdown_event.wait(0.05)
            else:
                return

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ReplayError(f"영상을 열 수 없습니다: {video_path}")

            first_elapsed = rows[0].elapsed_s
            replay_started_at = time.monotonic()
            last_reported_second = -1
            print(
                f"[REPLAY] 시작: frames={len(rows)}, "
                f"duration={rows[-1].elapsed_s - first_elapsed:.3f}s"
            )

            for row in rows:
                if core.shutdown_event.is_set():
                    return
                if not core.system_snapshot()["active"]:
                    stop_replay()
                    return

                ok, frame = cap.read()
                if not ok or frame is None:
                    raise ReplayError(
                        f"영상이 CSV보다 먼저 끝났습니다: frame_index={row.frame_index}"
                    )

                target = replay_started_at + (row.elapsed_s - first_elapsed)
                remaining = target - time.monotonic()
                if remaining > 0 and core.shutdown_event.wait(remaining):
                    return
                if not core.system_snapshot()["active"]:
                    stop_replay()
                    return

                now = time.time()
                with core.vehicle_lock, core.camera_lock:
                    core.vehicle_state.update(
                        {
                            "connected": row.obd_connected,
                            "speed_kmh": row.speed_kmh if row.obd_valid else None,
                            "updated_at": now,
                            "source": "replay",
                            "port": str(csv_path),
                            "status": row.obd_status,
                            "error": row.obd_error or None,
                            "replay_frame_index": row.frame_index,
                            "replay_elapsed_s": row.elapsed_s - first_elapsed,
                        }
                    )
                    core.latest_frame = frame
                    core.camera_state.update(
                        {
                            "connected": True,
                            "updated_at": now,
                            "error": None,
                            "replay_frame_index": row.frame_index,
                            "replay_elapsed_s": row.elapsed_s - first_elapsed,
                        }
                    )

                replay_second = int(row.elapsed_s - first_elapsed)
                if replay_second != last_reported_second:
                    speed_text = (
                        f"{row.speed_kmh:.1f} km/h" if row.obd_valid else "NO DATA"
                    )
                    print(
                        f"[REPLAY] {row.elapsed_s - first_elapsed:7.2f}s "
                        f"frame={row.frame_index} speed={speed_text}"
                    )
                    last_reported_second = replay_second

            extra_ok, _ = cap.read()
            if extra_ok:
                raise ReplayError(
                    f"CSV 종료 후 영상 프레임이 남았습니다: csv_frames={len(rows)}"
                )

            print("[REPLAY] 재생 완료")
            finish_replay("녹화본 재생이 완료되었습니다.")
        except Exception as exc:
            message = f"Replay 오류: {exc}"
            print(f"[REPLAY] {message}", file=sys.stderr)
            finish_replay(message, failed=True)
        finally:
            if cap is not None:
                cap.release()

    def replay_obd_stub() -> None:
        core.shutdown_event.wait()

    # 기존 웹·비전·제어 스레드는 유지하고 입력 두 개만 교체합니다.
    core.camera_loop = replay_loop
    core.obd_loop = replay_obd_stub
    core.main()


def main() -> int:
    args = build_parser().parse_args()
    try:
        video_path, csv_path, rows, video_frames = validate_recording(
            args.record_dir,
            decode_all=args.validate_only,
        )
        print_validation(video_path, csv_path, rows, video_frames)
    except ReplayError as exc:
        print(f"Replay 데이터 오류: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        return 0

    if args.allow_real_motor:
        if not confirm_real_motor():
            return 2
        os.environ["MOTOR_DRY_RUN"] = "0"
    else:
        os.environ["MOTOR_DRY_RUN"] = "1"

    run_server(video_path, csv_path, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
