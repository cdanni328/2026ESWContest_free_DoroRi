#!/usr/bin/env python3
"""DORORI 최종 판단, 전체 화면 ROI, 라우트 등록의 최소 회귀 검사."""

import os
import sys

os.environ["MOTOR_DRY_RUN"] = "1"
sys.modules["obd"] = None

import dorori_obd


CASES = (
    ((True, True, False, "dry"), "YELLOW"),
    ((True, False, True, "snow"), "YELLOW"),
    ((False, True, True, "dry"), "RED"),
    ((False, False, True, "dry"), "GREEN"),
    ((False, False, False, "wet"), "GREEN"),
    ((False, False, False, "ice"), "RED"),
    ((False, False, False, None), "RED"),
)

ROUTES = (
    "/",
    "/raw",
    "/yolo",
    "/video/raw",
    "/video/yolo",
    "/api/vision",
    "/api/status",
    "/api/health",
)


class DummyFrame:
    shape = (100, 200, 3)

    def __getitem__(self, key):
        self.last_slice = key
        return self


def main() -> None:
    for arguments, expected in CASES:
        assert dorori_obd.decide(*arguments)[0] == expected, arguments

    assert dorori_obd.VISION_ROI == (0.0, 0.0, 1.0, 1.0)
    assert dorori_obd.STOP_DEPLOY_DELAY_SEC == 3.0
    assert dorori_obd.stop_delay_remaining(None, 100.0) == 3.0
    assert 0.09 < dorori_obd.stop_delay_remaining(100.0, 102.9) < 0.11
    assert dorori_obd.stop_delay_remaining(100.0, 103.0) == 0.0
    frame = DummyFrame()
    assert dorori_obd.crop_frame_to_roi(frame, dorori_obd.VISION_ROI) is frame
    assert frame.last_slice == (slice(0, 100), slice(0, 200))

    registered = {rule.rule for rule in dorori_obd.app.url_map.iter_rules()}
    missing = [route for route in ROUTES if route not in registered]
    assert not missing, f"빠진 라우트: {missing}"

    dorori_obd.shutdown()
    print("decision, ROI, route checks passed")


if __name__ == "__main__":
    main()
