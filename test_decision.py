#!/usr/bin/env python3
"""DORORI 최종 판단과 전체 화면 ROI의 최소 회귀 검사."""

import os
import sys

os.environ["MOTOR_DRY_RUN"] = "1"
sys.modules["obd"] = None

import dorori_debug_speed
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


class DummyFrame:
    shape = (100, 200, 3)

    def __getitem__(self, key):
        self.last_slice = key
        return self


def main() -> None:
    for server in (dorori_debug_speed, dorori_obd):
        for arguments, expected in CASES:
            assert server.decide(*arguments)[0] == expected
        assert server.VISION_ROI == (0.0, 0.0, 1.0, 1.0)
        frame = DummyFrame()
        assert server.crop_frame_to_roi(frame, server.VISION_ROI) is frame
        assert frame.last_slice == (slice(0, 100), slice(0, 200))

    dorori_debug_speed.shutdown()
    dorori_obd.shutdown()
    print("decision and ROI checks passed")


if __name__ == "__main__":
    main()
