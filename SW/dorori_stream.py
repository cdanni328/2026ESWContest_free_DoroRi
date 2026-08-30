#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorori_stream.py
DORORI 3분할 시연용 MJPEG 스트림 모듈.

dorori_obd.py 와 같은 폴더에 두면 됩니다.
카메라는 서버가 이미 한 번만 열고 있고, 이 모듈은 그 프레임을 받아
두 갈래로 그려서 내보내기만 합니다. 카메라를 다시 열지 않고
YOLO도 다시 돌리지 않습니다.

브라우저 창 3개
    http://<파이IP>:5000/         기존 UI
    http://<파이IP>:5000/raw      카메라 원본 + 속도
    http://<파이IP>:5000/yolo     카메라 원본 + YOLO 오버레이 + 속도

환경변수
    STREAM_WIDTH            송출 가로 해상도 (기본 800, 0이면 원본 유지)
    STREAM_FPS              송출 상한 fps (기본 12)
    STREAM_JPEG_QUALITY     JPEG 품질 1~100 (기본 75)
    STREAM_MASK_FILL        마스크 반투명 채우기 (기본 1)
    STREAM_KOREAN_LABELS    한글 라벨 사용 (기본 1, 폰트 없으면 자동으로 영문)
    STREAM_YOLO_SMOOTH      1이면 최신 프레임에 직전 폴리곤을 얹어 부드럽게 송출
                            (기본 0 = 추론에 사용한 프레임 그대로. 윤곽이 정확함)

한글 라벨이 깨지면
    sudo apt install fonts-nanum
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from flask import Response, send_from_directory

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


STREAM_WIDTH = int(os.environ.get("STREAM_WIDTH", "800"))
STREAM_FPS = float(os.environ.get("STREAM_FPS", "12"))
STREAM_JPEG_QUALITY = int(os.environ.get("STREAM_JPEG_QUALITY", "75"))
STREAM_MASK_FILL = _env_bool("STREAM_MASK_FILL", True)
STREAM_KOREAN_LABELS = _env_bool("STREAM_KOREAN_LABELS", True)
STREAM_YOLO_SMOOTH = _env_bool("STREAM_YOLO_SMOOTH", False)

VISION_IDLE_SEC = 2.0
BOUNDARY = "dororiframe"

# 뷰어 페이지. 이 파일들이 없으면 아래 VIEWER_FALLBACK 을 대신 씁니다.
BASE_DIR = Path(__file__).resolve().parent
RAW_HTML_FILE_NAME = "dorori_raw.html"
YOLO_HTML_FILE_NAME = "dorori_yolo.html"

# BGR. live_seg.py 와 같은 색 계열을 씁니다.
SEG_COLORS = {
    "curb": (170, 220, 90),
    "drain": (60, 150, 245),
    "manhole": (245, 190, 90),
    "etc": (200, 200, 200),
}
LABEL_KO = {"curb": "연석", "drain": "배수구", "manhole": "맨홀", "etc": "기타"}
LABEL_EN = {"curb": "curb", "drain": "drain", "manhole": "manhole", "etc": "obj"}

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)

_font_cache: dict[int, Any] = {}
_font_lock = threading.Lock()


def load_font(size: int = 18):
    """한글 폰트를 한 번만 찾아 캐시합니다. 못 찾으면 None."""
    if not (HAVE_PIL and STREAM_KOREAN_LABELS):
        return None

    with _font_lock:
        if size in _font_cache:
            return _font_cache[size]

        font = None
        for path in FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue

        if font is None:
            print("[STREAM] 한글 폰트를 찾지 못해 영문 라벨을 사용합니다.")
            print("[STREAM] sudo apt install fonts-nanum")
        _font_cache[size] = font
        return font


# ───────────────────────────── 그리기 ─────────────────────────────
def draw_speed_overlay(
    frame: Any,
    speed_kmh: Optional[float],
    valid: bool,
) -> None:
    """영상 우측 상단에 현재 속도를 합성합니다.

    record_drive.py 의 오버레이와 같은 모양이라 두 창이 동일하게 보입니다.
    """
    height, width = frame.shape[:2]

    if valid and speed_kmh is not None:
        main_text = f"{speed_kmh:.1f} km/h"
        sub_text = "OBD-II SPEED"
    else:
        main_text = "--.- km/h"
        sub_text = "OBD-II NO DATA"

    font = cv2.FONT_HERSHEY_SIMPLEX
    main_scale = max(0.75, min(1.15, width / 1100.0))
    sub_scale = max(0.40, min(0.58, width / 2200.0))
    main_thickness = max(2, int(round(main_scale * 2)))

    (main_w, main_h), main_base = cv2.getTextSize(
        main_text, font, main_scale, main_thickness
    )
    (sub_w, sub_h), sub_base = cv2.getTextSize(sub_text, font, sub_scale, 1)

    margin = max(12, int(width * 0.015))
    pad_x = max(12, int(width * 0.012))
    pad_y = max(9, int(height * 0.012))
    gap = max(5, int(height * 0.007))

    box_w = max(main_w, sub_w) + 2 * pad_x
    box_h = pad_y + sub_h + sub_base + gap + main_h + main_base + pad_y

    x1 = max(0, width - margin - box_w)
    y1 = margin
    x2 = min(width - 1, x1 + box_w)
    y2 = min(height - 1, y1 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)

    text_x = x1 + pad_x
    sub_y = y1 + pad_y + sub_h
    main_y = sub_y + sub_base + gap + main_h

    cv2.putText(
        frame, sub_text, (text_x, sub_y), font, sub_scale,
        (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, main_text, (text_x, main_y), font, main_scale,
        (255, 255, 255) if valid else (190, 190, 190),
        main_thickness, cv2.LINE_AA,
    )


def draw_hud(frame: Any, text: str) -> None:
    """좌측 상단 검은 띠에 한 줄 정보를 씁니다."""
    width = frame.shape[1]
    bar_h = max(24, int(width * 0.032))
    cv2.rectangle(frame, (0, 0), (width, bar_h), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (10, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX, max(0.42, min(0.62, width / 1500.0)),
        (240, 240, 240), 1, cv2.LINE_AA,
    )


def draw_text_labels(frame: Any, items: list[tuple[int, int, str, tuple]]) -> Any:
    """items: [(x, y, text, bgr)]. PIL 변환은 프레임당 한 번만 합니다."""
    if not items:
        return frame

    font = load_font(max(14, int(frame.shape[1] / 45)))
    if font is None:
        for x, y, text, color in items:
            ascii_text = text.encode("ascii", "ignore").decode().strip() or "obj"
            cv2.putText(
                frame, ascii_text, (x, max(16, y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for x, y, text, color in items:
        rgb = (color[2], color[1], color[0])
        y = max(2, y - 24)
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle([box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3], fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=rgb)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_detections(
    frame: Any,
    detections: list[dict[str, Any]],
    scale: float,
) -> tuple[Any, dict[str, int]]:
    """폴리곤/박스와 라벨을 그리고, 종류별 개수를 돌려줍니다."""
    counts = {"curb": 0, "drain": 0, "manhole": 0}
    if not detections:
        return frame, counts

    polygons: list[tuple[Any, tuple]] = []
    for detection in detections:
        kind = detection.get("kind", "etc")
        if kind in counts:
            counts[kind] += 1
        polygon = detection.get("polygon")
        if polygon is None or len(polygon) < 3:
            continue
        points = (np.asarray(polygon, dtype=np.float32) * scale).astype(np.int32)
        polygons.append((points, SEG_COLORS.get(kind, SEG_COLORS["etc"])))

    if STREAM_MASK_FILL and polygons:
        layer = frame.copy()
        for points, color in polygons:
            cv2.fillPoly(layer, [points], color)
        cv2.addWeighted(layer, 0.22, frame, 0.78, 0.0, frame)

    labels: list[tuple[int, int, str, tuple]] = []
    for detection in detections:
        kind = detection.get("kind", "etc")
        color = SEG_COLORS.get(kind, SEG_COLORS["etc"])
        polygon = detection.get("polygon")

        if polygon is not None and len(polygon) >= 3:
            points = (np.asarray(polygon, dtype=np.float32) * scale).astype(np.int32)
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
            for point in points[:: max(1, len(points) // 40)]:
                cv2.circle(frame, tuple(int(v) for v in point), 2, color, -1, cv2.LINE_AA)
            label_x, label_y = int(points[:, 0].min()), int(points[:, 1].min())
        else:
            box = detection.get("xyxy")
            if not box:
                continue
            x1, y1, x2, y2 = (int(value * scale) for value in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label_x, label_y = x1, y1

        name = (LABEL_KO if load_font(16) is not None else LABEL_EN).get(kind, kind)
        confidence = detection.get("conf")
        text = name if confidence is None else f"{name} {float(confidence):.2f}"
        labels.append((label_x, label_y, text, color))

    frame = draw_text_labels(frame, labels)
    return frame, counts


def fit_width(frame: Any) -> tuple[Any, float]:
    """송출 해상도로 줄이고, 좌표 변환용 배율을 함께 돌려줍니다."""
    if STREAM_WIDTH <= 0 or frame.shape[1] <= STREAM_WIDTH:
        return frame.copy(), 1.0
    scale = STREAM_WIDTH / float(frame.shape[1])
    resized = cv2.resize(
        frame,
        (STREAM_WIDTH, max(1, int(round(frame.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def placeholder_frame(message: str) -> Any:
    width = STREAM_WIDTH if STREAM_WIDTH > 0 else 800
    frame = np.zeros((int(width * 9 / 16), width, 3), dtype=np.uint8)
    (text_w, text_h), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        frame, message,
        ((width - text_w) // 2, (frame.shape[0] + text_h) // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2.LINE_AA,
    )
    return frame


# ───────────────────────────── 공유 허브 ─────────────────────────────
class StreamHub:
    """서버의 카메라 프레임과 YOLO 결과를 스트림 쪽으로 넘겨주는 통로."""

    def __init__(
        self,
        get_frame: Callable[[], tuple[Any, float]],
        get_speed: Callable[[], tuple[Optional[float], bool]],
    ) -> None:
        self._get_frame = get_frame
        self._get_speed = get_speed
        self._condition = threading.Condition()
        self._frame: Any = None
        self._detections: list[dict[str, Any]] = []
        self._decision: Optional[str] = None
        self._surface_text: Optional[str] = None
        self._sequence = 0
        self._updated_at = 0.0

    def publish_vision(
        self,
        frame: Any,
        detections: list[dict[str, Any]],
        decision: Optional[str] = None,
        surface_text: Optional[str] = None,
    ) -> None:
        """vision_loop 가 추론을 끝낼 때마다 호출합니다."""
        snapshot = None if frame is None else frame.copy()
        with self._condition:
            self._frame = snapshot
            self._detections = list(detections)
            self._decision = decision
            self._surface_text = surface_text
            self._sequence += 1
            self._updated_at = time.time()
            self._condition.notify_all()

    def wait_vision(self, last_sequence: int, timeout: float) -> dict[str, Any]:
        with self._condition:
            if self._sequence == last_sequence:
                self._condition.wait(timeout)
            return {
                "sequence": self._sequence,
                "frame": self._frame,
                "detections": list(self._detections),
                "decision": self._decision,
                "surface_text": self._surface_text,
                "updated_at": self._updated_at,
            }

    def peek_vision(self) -> dict[str, Any]:
        with self._condition:
            return {
                "sequence": self._sequence,
                "frame": self._frame,
                "detections": list(self._detections),
                "decision": self._decision,
                "surface_text": self._surface_text,
                "updated_at": self._updated_at,
            }

    def camera_frame(self) -> tuple[Any, float]:
        try:
            return self._get_frame()
        except Exception:
            return None, 0.0

    def speed(self) -> tuple[Optional[float], bool]:
        try:
            return self._get_speed()
        except Exception:
            return None, False


# ───────────────────────────── MJPEG 송출 ─────────────────────────────
def encode_jpeg(frame: Any) -> Optional[bytes]:
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY]
    )
    return buffer.tobytes() if ok else None


def multipart_chunk(payload: bytes) -> bytes:
    return (
        f"--{BOUNDARY}\r\n"
        f"Content-Type: image/jpeg\r\n"
        f"Content-Length: {len(payload)}\r\n\r\n"
    ).encode("ascii") + payload + b"\r\n"


def raw_stream(hub: StreamHub):
    """카메라 원본 + 속도."""
    period = 1.0 / max(1.0, STREAM_FPS)

    while True:
        started = time.time()
        frame, updated_at = hub.camera_frame()

        if frame is None or updated_at <= 0:
            output = placeholder_frame("NO CAMERA")
        else:
            output, _ = fit_width(frame)

        speed_kmh, speed_valid = hub.speed()
        draw_speed_overlay(output, speed_kmh, speed_valid)

        payload = encode_jpeg(output)
        if payload is not None:
            yield multipart_chunk(payload)

        time.sleep(max(0.0, period - (time.time() - started)))


def yolo_stream(hub: StreamHub):
    """카메라 원본 + YOLO 오버레이 + 속도."""
    period = 1.0 / max(1.0, STREAM_FPS)
    last_sequence = -1

    while True:
        started = time.time()

        if STREAM_YOLO_SMOOTH:
            vision = hub.peek_vision()
            base, _ = hub.camera_frame()
        else:
            vision = hub.wait_vision(last_sequence, timeout=1.0)
            last_sequence = vision["sequence"]
            base = vision["frame"]

        vision_fresh = (
            vision["updated_at"] > 0
            and (time.time() - vision["updated_at"]) <= VISION_IDLE_SEC
        )

        if base is None or not vision_fresh:
            # 판단이 꺼져 있거나 아직 추론 전이면 원본만 보여줍니다.
            live_frame, updated_at = hub.camera_frame()
            if live_frame is None or updated_at <= 0:
                output = placeholder_frame("NO CAMERA")
            else:
                output, _ = fit_width(live_frame)
                draw_hud(output, "vision idle  |  waiting for inference")
            speed_kmh, speed_valid = hub.speed()
            draw_speed_overlay(output, speed_kmh, speed_valid)
            payload = encode_jpeg(output)
            if payload is not None:
                yield multipart_chunk(payload)
            time.sleep(max(0.0, period - (time.time() - started)))
            continue

        output, scale = fit_width(base)
        output, counts = draw_detections(output, vision["detections"], scale)

        hud = f"curb {counts['curb']}  drain {counts['drain']}  manhole {counts['manhole']}"
        if vision["decision"]:
            hud += f"  |  {vision['decision']}"
        if vision["surface_text"]:
            hud += f"  |  {vision['surface_text']}"
        draw_hud(output, hud)

        speed_kmh, speed_valid = hub.speed()
        draw_speed_overlay(output, speed_kmh, speed_valid)

        payload = encode_jpeg(output)
        if payload is not None:
            yield multipart_chunk(payload)

        if STREAM_YOLO_SMOOTH:
            time.sleep(max(0.0, period - (time.time() - started)))


# ───────────────────────────── 뷰어 페이지 ─────────────────────────────
VIEWER_FALLBACK = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DORORI {title}</title>
<style>
  html,body{{margin:0;height:100%;background:#000;overflow:hidden;}}
  img{{width:100%;height:100%;object-fit:contain;display:block;}}
  .tag{{position:fixed;left:10px;bottom:8px;color:#7c8a9c;font:600 11px -apple-system,sans-serif;
        letter-spacing:.06em;text-transform:uppercase;}}
</style></head>
<body>
<img src="{source}" alt="DORORI {title}">
<div class="tag">{title}</div>
</body></html>
"""


def serve_viewer(file_name: str, title: str, source: str):
    """HTML 파일이 있으면 그걸 주고, 없으면 최소 페이지로 대체합니다."""
    if (BASE_DIR / file_name).is_file():
        return send_from_directory(str(BASE_DIR), file_name)
    return VIEWER_FALLBACK.format(title=title, source=source)


def register_stream(app, hub: StreamHub) -> None:
    """Flask 앱에 스트림 라우트를 붙입니다."""

    def _response(generator):
        return Response(
            generator,
            mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Connection": "close",
            },
        )

    @app.route("/video/raw")
    def video_raw():
        return _response(raw_stream(hub))

    @app.route("/video/yolo")
    def video_yolo():
        return _response(yolo_stream(hub))

    @app.route("/raw")
    def view_raw():
        return serve_viewer(RAW_HTML_FILE_NAME, "camera raw", "/video/raw")

    @app.route("/yolo")
    def view_yolo():
        return serve_viewer(YOLO_HTML_FILE_NAME, "camera + yolo", "/video/yolo")

    print(
        f"[STREAM] /raw, /yolo 준비 완료 "
        f"(width={STREAM_WIDTH or 'native'}, fps={STREAM_FPS:.0f}, "
        f"quality={STREAM_JPEG_QUALITY}, smooth={int(STREAM_YOLO_SMOOTH)})"
    )
