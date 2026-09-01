#!/usr/bin/env python3
"""DORORI 판단, 음성-모터 순서, 라우트 등록의 최소 회귀 검사."""

import os
import sys
import time
from pathlib import Path

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
    "/api/announcement",
    "/api/health",
    "/audio/<path:filename>",
)

ANNOUNCEMENTS = ("DEPLOY", "DEPLOYED", "RETRACT", "RETRACTED")
ANNOUNCEMENT_PROGRESS = {
    "DEPLOY": 0,
    "DEPLOYED": 100,
    "RETRACT": 100,
    "RETRACTED": 0,
}
AUDIO_FILES = (
    "announce_deploy.mp3",
    "announce_deployed.mp3",
    "announce_retract.mp3",
    "announce_retracted.mp3",
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
    assert dorori_obd.STOP_DEPLOY_DELAY_SEC == 0.1
    assert dorori_obd.stop_delay_remaining(None, 100.0) == 0.1
    assert 0.04 < dorori_obd.stop_delay_remaining(100.0, 100.05) < 0.06
    assert dorori_obd.stop_delay_remaining(100.0, 100.11) == 0.0
    frame = DummyFrame()
    assert dorori_obd.crop_frame_to_roi(frame, dorori_obd.VISION_ROI) is frame
    assert frame.last_slice == (slice(0, 100), slice(0, 200))

    registered = {rule.rule for rule in dorori_obd.app.url_map.iter_rules()}
    missing = [route for route in ROUTES if route not in registered]
    assert not missing, f"빠진 라우트: {missing}"
    audio_response = dorori_obd.app.test_client().get("/audio/announce_deploy.mp3")
    assert audio_response.status_code == 200
    assert audio_response.mimetype == "audio/mpeg"

    for html_name in ("dorori_obd.html", "dorori_debug_speed.html"):
        html = (Path(__file__).parent / html_name).read_text(encoding="utf-8")
        assert all(html.count(name) == 1 for name in AUDIO_FILES), html_name
        assert "function warnBeep()" in html, html_name
        assert "SpeechSynthesisUtterance" not in html, html_name

    for audio_name in AUDIO_FILES:
        assert (Path(__file__).parent / "audio" / audio_name).stat().st_size > 5_000

    dorori_obd.MOTOR_STEPS_PER_REV = 2
    dorori_obd.MOTOR_FREQUENCY_HZ = 50.0
    dorori_obd.MOTOR_HOLD_SEC = 0.02
    dorori_obd.CYCLE_MAX_SEC = 2.0
    dorori_obd.MOTOR_MOVE_BUFFER_SEC = 0.01
    dorori_obd.ANNOUNCEMENT_RESERVE_SEC = dict.fromkeys(ANNOUNCEMENTS, 0.05)
    cycle_started_at = time.time()
    ok, reason = dorori_obd.motor.request_cycle("test", cycle_started_at)
    assert ok, reason

    heard: list[str] = []
    deadline = time.time() + 3.0
    while time.time() < deadline:
        state = dorori_obd.motor.snapshot()
        announcement = state.get("announcement")
        if announcement and announcement not in heard:
            heard.append(announcement)
            assert round(state["progress"]) == ANNOUNCEMENT_PROGRESS[announcement]
            time.sleep(0.01)
            assert dorori_obd.motor.snapshot()["state"] == state["state"]
            acknowledged, ack_reason = dorori_obd.motor.finish_announcement(
                state["cycle_id"], announcement, True
            )
            assert acknowledged, ack_reason
        if state["state"] == "IDLE" and state.get("last_completed_at"):
            break
        time.sleep(0.002)

    final_state = dorori_obd.motor.snapshot()
    assert heard == list(ANNOUNCEMENTS), heard
    assert final_state["state"] == "IDLE", final_state
    assert final_state["retracted_at"] - cycle_started_at <= dorori_obd.CYCLE_MAX_SEC
    assert final_state["last_completed_at"] - cycle_started_at <= dorori_obd.CYCLE_MAX_SEC

    dorori_obd.shutdown()
    print("decision, audio sequence, deadline, route checks passed")


if __name__ == "__main__":
    main()
