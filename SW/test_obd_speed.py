#!/usr/bin/env python3
"""USB ELM327에서 차량 속도를 읽어 터미널에 표시한다."""

import argparse
import sys
import time

import obd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OBD-II 차량 속도(PID 01 0D) 시험")
    parser.add_argument("--port", help="예: /dev/ttyUSB0 (생략하면 자동 검색)")
    parser.add_argument("--baudrate", type=int, help="예: 38400 (생략하면 자동 감지)")
    parser.add_argument("--interval", type=float, default=0.2, help="조회 주기(초, 기본 0.2)")
    parser.add_argument("--timeout", type=float, default=1.0, help="응답 제한 시간(초, 기본 1.0)")
    parser.add_argument("--slow", action="store_true", help="복제 ELM327용 fast 모드 끄기")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval <= 0 or args.timeout <= 0:
        print("--interval과 --timeout은 0보다 커야 합니다.", file=sys.stderr)
        return 2

    connection = None
    try:
        print(f"OBD 연결 중: {args.port or '자동 검색'}")
        connection = obd.OBD(
            args.port,
            baudrate=args.baudrate,
            fast=not args.slow,
            timeout=args.timeout,
        )
        if not connection.is_connected():
            print(f"차량 연결 실패: {connection.status()}", file=sys.stderr)
            print("시동(IGN ON), 포트, dialout 권한을 확인하세요.", file=sys.stderr)
            return 1

        print(f"연결 완료: {connection.port_name()} ({connection.status()})")
        print("종료: Ctrl+C")
        while True:
            response = connection.query(obd.commands.SPEED, force=True)
            if response.is_null() or response.value is None:
                text = "응답 없음"
            else:
                text = f"{float(response.value.to('kph').magnitude):6.1f} km/h"
            print(f"\r차량 속도: {text:<20}", end="", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n종료합니다.")
        return 0
    except Exception as exc:
        print(f"\nOBD 오류: {exc}", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
