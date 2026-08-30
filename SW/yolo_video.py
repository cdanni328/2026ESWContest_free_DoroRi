#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_video.py
폴더에 있는 영상에 YOLO seg + 노면 classify 를 씌워 새 영상으로 저장합니다.

DORORI 서버와 같은 판단 규칙을 씁니다.
    연석 있음                       -> YELLOW (발판 필요)
    연석 없고 배수구 있음            -> RED
    연석 없고 노면이 눈/얼음/불명    -> RED
    연석 없고 배수구 없고 안전 노면  -> GREEN

준비
    pip install ultralytics opencv-python pillow

사용법
    1. 아래 "여기만 고치세요" 칸에 경로를 적는다
    2. python3 yolo_video.py

    끝입니다. 다른 파일은 필요 없습니다.

한글 라벨이 네모로 깨지면
    sudo apt install fonts-nanum
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ═════════════════════ 여기만 고치세요 ═════════════════════

# 욜로를 씌울 영상. 파일 하나를 적어도 되고, 폴더를 적으면 그 안의 영상을 전부 처리합니다.
VIDEO_PATH = "drive.mp4"

# 연석·배수구·맨홀 세그멘테이션 가중치
SEG_WEIGHTS_PATH = "best_selection.pt"

# 노면 상태 분류 가중치. 빈칸("")으로 두면 seg 만 돌립니다.
CLASSIFY_WEIGHTS_PATH = "best노면.pt"

# 저장할 파일 이름. 빈칸으로 두면 원본 옆에 <원본이름>_yolo.mp4 로 저장합니다.
OUTPUT_PATH = ""

SEG_CONF = 0.40        # 검출 임계값. 낮추면 더 많이 잡고 오검출도 늘어납니다
SEG_IMGSZ = 640        # seg 추론 해상도. 416 으로 낮추면 빨라지고 작은 물체를 놓칩니다
CLASSIFY_IMGSZ = 224   # classify 추론 해상도

SHOW_HUD = True        # 좌측 상단에 개수 + 노면 + 판단 표시
SHOW_DECISION = True   # HUD 에 GREEN/YELLOW/RED 판단 표시
FILL_MASK = True       # 마스크 반투명 채우기
PREVIEW = False        # 처리하면서 창으로 보기 (모니터 있을 때만. q 로 중단)

# ═══════════════════════════════════════════════════════════


# BGR. 연석은 연두, 배수구는 주황, 맨홀은 하늘색
COLORS = {
    "curb": (170, 220, 90),
    "drain": (60, 150, 245),
    "manhole": (245, 190, 90),
    "etc": (200, 200, 200),
}
LABELS_KO = {"curb": "연석", "drain": "배수구", "manhole": "맨홀", "etc": "기타"}
LABELS_EN = {"curb": "curb", "drain": "drain", "manhole": "manhole", "etc": "obj"}

DECISION_COLORS = {
    "GREEN": (132, 220, 61),
    "YELLOW": (61, 194, 255),
    "RED": (112, 84, 255),
}

CURB_WORDS = ("curb", "kerb", "연석")
DRAIN_WORDS = ("drain", "drainage", "grate", "gutter", "배수", "배수구")
MANHOLE_WORDS = ("manhole", "맨홀")

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False


# ───────────────────────── 판단 규칙 ─────────────────────────
def surface_group(surface_class):
    """dry/wet/water 는 SAFE, snow/ice 는 DANGER, 나머지는 UNKNOWN."""
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


def decide(curb_detected, drainage_detected, surface_class):
    """연석이 최우선. 맨홀은 표시만 하고 판단에는 쓰지 않습니다."""
    if curb_detected:
        return "YELLOW"
    if drainage_detected:
        return "RED"
    if surface_group(surface_class) != "SAFE":
        return "RED"
    return "GREEN"


# ───────────────────────── 그리기 ─────────────────────────
def load_font(size):
    """한글 폰트를 찾습니다. 없으면 None 이고 라벨이 영문으로 나옵니다."""
    if not HAVE_PIL:
        return None
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return None


def pil_to_bgr(image):
    """np.array(image) 를 피해 numpy 2.x 경고 없이 되돌립니다. 더 빠르기도 합니다."""
    width, height = image.size
    array = np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(height, width, 3)
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def draw_labels(frame, items, font):
    """items: [(x, y, text, bgr)]. PIL 변환은 프레임당 한 번만 합니다."""
    if not items:
        return frame

    if font is None:
        for x, y, text, color in items:
            ascii_text = text.encode("ascii", "ignore").decode().strip() or "obj"
            cv2.putText(frame, ascii_text, (x, max(16, y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for x, y, text, color in items:
        y = max(2, y - 24)
        box = draw.textbbox((x, y), text, font=font)
        draw.rectangle([box[0] - 5, box[1] - 3, box[2] + 5, box[3] + 3], fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return pil_to_bgr(image)


def draw_result(frame, result, kind_map, font):
    """한 프레임에 폴리곤과 라벨을 그리고, 종류별 개수를 돌려줍니다."""
    counts = {"curb": 0, "drain": 0, "manhole": 0}
    if result.boxes is None or len(result.boxes) == 0:
        return frame, counts

    masks_xy = result.masks.xy if getattr(result, "masks", None) is not None else None

    items = []
    for index, box in enumerate(result.boxes):
        kind = kind_map.get(int(box.cls[0]), "etc")
        confidence = float(box.conf[0]) if box.conf is not None else None
        polygon = (
            masks_xy[index]
            if masks_xy is not None and index < len(masks_xy)
            else None
        )
        if kind in counts:
            counts[kind] += 1
        items.append((kind, confidence, polygon, box.xyxy[0].tolist()))

    # 1) 반투명 채우기를 먼저 한 번에
    if FILL_MASK:
        polygons = [
            (np.asarray(poly, np.int32), COLORS[kind])
            for kind, _, poly, _ in items
            if poly is not None and len(poly) >= 3
        ]
        if polygons:
            layer = frame.copy()
            for points, color in polygons:
                cv2.fillPoly(layer, [points], color)
            cv2.addWeighted(layer, 0.22, frame, 0.78, 0.0, frame)

    # 2) 외곽선과 라벨 위치
    labels = []
    for kind, confidence, polygon, xyxy in items:
        color = COLORS[kind]

        if polygon is not None and len(polygon) >= 3:
            points = np.asarray(polygon, np.int32)
            cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
            for point in points[:: max(1, len(points) // 40)]:
                cv2.circle(frame, tuple(int(v) for v in point), 2, color, -1, cv2.LINE_AA)
            label_x, label_y = int(points[:, 0].min()), int(points[:, 1].min())
        else:
            # seg 가 아닌 detect 모델이면 폴리곤이 없으므로 사각형으로 그립니다.
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label_x, label_y = x1, y1

        name = (LABELS_KO if font is not None else LABELS_EN)[kind]
        text = name if confidence is None else f"{name} {confidence:.2f}"
        labels.append((label_x, label_y, text, color))

    return draw_labels(frame, labels, font), counts


def draw_hud(frame, plain_text, decision):
    """좌측 상단 검은 띠. 판단 결과만 색을 넣어 뒤에 붙입니다."""
    width = frame.shape[1]
    bar_height = max(26, int(width * 0.034))
    scale = max(0.42, min(0.62, width / 1500.0))
    baseline_y = int(bar_height * 0.72)

    cv2.rectangle(frame, (0, 0), (width, bar_height), (0, 0, 0), -1)
    cv2.putText(frame, plain_text, (10, baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)

    if decision:
        (text_width, _), _ = cv2.getTextSize(
            plain_text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )
        cv2.putText(frame, decision, (10 + text_width + 6, baseline_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale,
                    DECISION_COLORS.get(decision, (240, 240, 240)), 2, cv2.LINE_AA)


# ───────────────────────── 모델 준비 ─────────────────────────
def map_class_names(names):
    """모델 클래스 이름에서 연석/배수구/맨홀을 찾아냅니다."""
    pairs = (
        [(int(i), str(n)) for i, n in names.items()]
        if isinstance(names, dict)
        else list(enumerate(str(n) for n in names))
    )

    mapping = {}
    for index, name in pairs:
        lowered = name.lower()
        if any(word in lowered for word in CURB_WORDS):
            mapping[index] = "curb"
        elif any(word in lowered for word in DRAIN_WORDS):
            mapping[index] = "drain"
        elif any(word in lowered for word in MANHOLE_WORDS):
            mapping[index] = "manhole"
        else:
            mapping[index] = "etc"

    print(f"[i] seg 클래스: {dict(pairs)}")
    print("[i] 매핑: " + ", ".join(f"{i}={n} -> {mapping[i]}" for i, n in pairs))
    if "curb" not in mapping.values():
        print("[!] 연석 클래스를 이름으로 못 찾았습니다. 전부 '기타' 색으로 그려집니다.")
    return mapping


def read_surface(result):
    """classify 결과에서 top-1 클래스와 확신도를 꺼냅니다."""
    if result is None or getattr(result, "probs", None) is None:
        return None, None
    top1 = int(result.probs.top1)
    return str(result.names[top1]), float(result.probs.top1conf.item())


def format_duration(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}초"
    return f"{seconds // 60}분 {seconds % 60}초"


# ───────────────────────── 본체 ─────────────────────────
def process_video(source, output, seg_model, classify_model, kind_map, font):
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        print(f"[!] 영상을 열 수 없습니다: {source}")
        return False

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"\n[i] 입력: {source.name}  ({width}x{height}, {fps:.1f} fps, {total} 프레임)")
    print(f"[i] 출력: {output}")

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print(f"[!] 출력 파일을 만들 수 없습니다: {output}")
        capture.release()
        return False

    window = f"yolo_video · {source.name}"
    if PREVIEW:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    processed = 0
    totals = {"curb": 0, "drain": 0, "manhole": 0}
    decisions = {"GREEN": 0, "YELLOW": 0, "RED": 0}
    surfaces = {}
    seg_ms_total = 0.0
    classify_ms_total = 0.0
    started = time.time()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            mark = time.time()
            seg_result = seg_model.predict(
                frame, conf=SEG_CONF, imgsz=SEG_IMGSZ, verbose=False
            )[0]
            seg_ms_total += (time.time() - mark) * 1000.0

            classify_result = None
            if classify_model is not None:
                mark = time.time()
                classify_result = classify_model.predict(
                    frame, imgsz=CLASSIFY_IMGSZ, verbose=False
                )[0]
                classify_ms_total += (time.time() - mark) * 1000.0

            surface_class, surface_conf = read_surface(classify_result)
            if surface_class:
                surfaces[surface_class] = surfaces.get(surface_class, 0) + 1

            view, counts = draw_result(frame.copy(), seg_result, kind_map, font)
            for key, value in counts.items():
                totals[key] += value

            decision = decide(
                curb_detected=counts["curb"] > 0,
                drainage_detected=counts["drain"] > 0,
                surface_class=surface_class,
            )
            decisions[decision] += 1

            if SHOW_HUD:
                text = (
                    f"curb {counts['curb']}  drain {counts['drain']}  "
                    f"manhole {counts['manhole']}"
                )
                if surface_class:
                    text += f"  |  {surface_class} {surface_conf:.2f}"
                elif classify_model is not None:
                    text += "  |  surface ?"
                text += "  |  "
                draw_hud(view, text, decision if SHOW_DECISION else None)

            writer.write(view)
            processed += 1

            if PREVIEW:
                cv2.imshow(window, view)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    print("\n[i] 중단했습니다.")
                    break

            if processed % 10 == 0 or processed == total:
                elapsed = time.time() - started
                rate = processed / max(1e-6, elapsed)
                timing = f"seg {seg_ms_total / processed:.0f}ms"
                if classify_model is not None:
                    timing += f" · cls {classify_ms_total / processed:.0f}ms"
                head = (
                    f"\r  [{100.0 * processed / total:5.1f}%] {processed}/{total}"
                    if total else f"\r  {processed}"
                )
                tail = (
                    f" · 남은 시간 {format_duration((total - processed) / max(1e-6, rate))}"
                    if total else ""
                )
                print(f"{head} 프레임 · {rate:.1f} fps · {timing}{tail}   ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n[i] Ctrl+C 로 중단했습니다. 여기까지는 저장됩니다.")
    finally:
        writer.release()
        capture.release()
        if PREVIEW:
            cv2.destroyWindow(window)

    if processed == 0:
        print("\n[!] 처리된 프레임이 없습니다.")
        return False

    elapsed = time.time() - started
    other_ms = (elapsed * 1000.0 - seg_ms_total - classify_ms_total) / processed

    print(f"\n[i] {processed} 프레임 처리 · {format_duration(elapsed)} 소요")
    print(
        f"[i] 프레임당 평균: seg {seg_ms_total / processed:.0f}ms · "
        f"classify {classify_ms_total / processed:.0f}ms · "
        f"그리기+저장 {other_ms:.0f}ms · 합계 {elapsed * 1000.0 / processed:.0f}ms"
    )
    print(
        f"[i] 누적 검출: 연석 {totals['curb']}  배수구 {totals['drain']}  "
        f"맨홀 {totals['manhole']}"
    )
    if surfaces:
        top = sorted(surfaces.items(), key=lambda pair: -pair[1])
        print("[i] 노면: " + ", ".join(f"{name} {count}프레임" for name, count in top))
    print(
        f"[i] 판단: GREEN {decisions['GREEN']}  YELLOW {decisions['YELLOW']}  "
        f"RED {decisions['RED']}"
    )
    return True


def main():
    source_path = Path(VIDEO_PATH).expanduser()

    if not source_path.exists():
        print(f"[!] 영상을 찾을 수 없습니다: {source_path}")
        print("[!] 이 파일 위쪽의 VIDEO_PATH 를 실제 영상 경로로 고치세요.")
        print(f"[!] 참고로 지금 실행 위치는 {Path.cwd()} 입니다.")
        return 1

    if source_path.is_dir():
        sources = sorted(
            path for path in source_path.iterdir()
            if path.suffix.lower() in VIDEO_SUFFIXES and "_yolo" not in path.stem
        )
        if not sources:
            print(f"[!] {source_path} 안에 영상 파일이 없습니다.")
            return 1
        print(f"[i] 폴더에서 영상 {len(sources)}개를 찾았습니다.")
    else:
        sources = [source_path]

    seg_path = Path(SEG_WEIGHTS_PATH).expanduser()
    if not seg_path.is_file():
        print(f"[!] seg 가중치를 찾을 수 없습니다: {seg_path}")
        print("[!] 이 파일 위쪽의 SEG_WEIGHTS_PATH 를 고치세요.")
        return 1

    print(f"[i] seg 모델 로딩: {seg_path}")
    seg_model = YOLO(str(seg_path))
    kind_map = map_class_names(seg_model.names)

    classify_model = None
    if CLASSIFY_WEIGHTS_PATH.strip():
        classify_path = Path(CLASSIFY_WEIGHTS_PATH).expanduser()
        if not classify_path.is_file():
            print(f"[!] classify 가중치를 찾을 수 없습니다: {classify_path}")
            print('[!] CLASSIFY_WEIGHTS_PATH 를 고치거나 빈칸("")으로 두세요.')
            return 1
        print(f"[i] classify 모델 로딩: {classify_path}")
        classify_model = YOLO(str(classify_path))
        print(f"[i] classify 클래스: {classify_model.names}")
    else:
        print("[i] classify 가중치가 비어 있어 seg 만 돌립니다.")

    font = load_font(19)
    if font is None:
        print("[!] 한글 폰트가 없어 라벨이 영문으로 나옵니다.")
        print("[!] sudo apt install fonts-nanum  으로 설치할 수 있습니다.")

    done = []
    for source in sources:
        if OUTPUT_PATH and len(sources) == 1:
            output = Path(OUTPUT_PATH).expanduser()
        else:
            output = source.with_name(f"{source.stem}_yolo.mp4")

        if process_video(source, output, seg_model, classify_model, kind_map, font):
            done.append(output)

    print()
    for output in done:
        print(f"[i] 저장됨: {output}  ({output.stat().st_size / 1e6:.1f} MB)")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
