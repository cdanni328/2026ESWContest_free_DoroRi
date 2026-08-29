# DORORI 프로젝트 — 모터 통합 서버 2종

이 폴더에는 같은 카메라·YOLO 판단·A4988 모터 제어 상태머신을 공유하는 두 실행 버전이 들어 있습니다.

- `dorori_debug_speed.py` + `dorori_debug_speed.html`
  - OBD-II 없이 HTML에서 차량 속도만 수동 입력
  - 연석·배수로·맨홀·노면 판단은 실제 카메라와 두 가중치로 수행
- `dorori_obd.py` + `dorori_obd.html`
  - USB ELM327 계열 OBD-II 어댑터에서 실제 차량 속도를 읽음
  - 속도 데이터가 끊기거나 오래되면 `FAULT` 처리하고, 발판이 움직이는 중이면 조기 수납 요청

가중치 기본 경로는 서버 파일과 같은 폴더의 다음 파일입니다.

- `best_selection.pt`: segmentation (`curb`, `drainage`, `manhole`)
- `best노면.pt`: road-surface classification

> 현재 업로드된 segmentation 가중치에는 별도의 `obstacle` 클래스가 없습니다. 따라서 이 코드의 시각 판단 대상은 연석·배수로·맨홀과 노면입니다.

---

## 1. 기본 동작

운전자가 화면의 판단 스위치를 켜면 다음 순서로 동작합니다.

1. 속도가 `5.0 km/h`를 초과하면 카메라 판단과 전개를 대기합니다.
2. `0.1 < 속도 <= 5.0 km/h`에서는 카메라 판단 결과만 표시하며 모터를 전개하지 않습니다.
3. 속도가 `0.1 km/h` 이하가 되면 **정차 이후 새로 얻은 카메라 결과**를 기다립니다.
4. 정차 후 결과가 `YELLOW`로 1초 유지되면 모터 사이클을 한 번만 시작합니다.
5. 기본 모터 사이클:
   - 전개 방향 `1600` STEP 펄스
   - 전개 상태 `5초` 유지
   - 반대 방향 `1600` STEP 펄스로 수납
6. 속도가 다시 올라가거나, 비전/속도 오류가 생기거나, 시스템을 끄거나, `즉시 수납`을 누르면:
   - 유지 중이면 5초를 기다리지 않고 수납
   - 전개 도중이면 **실제로 전개된 펄스 수만큼만 즉시 역회전**
7. 같은 판단 세션에서는 `deploy_latched`로 반복 전개를 막습니다.

기본 `1600 pulse / 300 Hz`에서는 한 방향 회전에 약 `5.33초`가 걸립니다. 정상 전체 사이클은 약 `5.33 + 5 + 5.33 = 15.67초`입니다.

### 판단 우선순위

1. 연석 감지 → 다른 감지 결과와 관계없이 `YELLOW`, 정차 시 전개
2. 연석 없음 + 배수로 감지 → 노면 불량으로 `RED`, 전개 금지
3. 연석 없음 + 눈·빙판 또는 알 수 없는 노면 → `RED`, 전개 금지
4. 연석·배수로 없음 + dry/wet/water 계열 노면 → `GREEN`

연석과 배수로가 동시에 검출되면 연석을 우선하여 `YELLOW`입니다. 맨홀은 UI에 감지 결과만 표시하며 최종 판단에는 사용하지 않습니다.

현재 `VISION_ROI=(0.0, 0.0, 1.0, 1.0)`으로 카메라 전체 화면을 세그멘테이션과 노면 분류에 사용합니다. 실차 테스트 후 두 서버 파일의 이 값을 정규화 좌표 `(x1, y1, x2, y2)`로 조정하면 두 모델에 같은 ROI가 적용됩니다.

---

## 2. 파일 배치

실행할 때 다음 파일들을 같은 폴더에 둡니다.

```text
dorori_motor_bundle/
├── dorori_debug_speed.py
├── dorori_debug_speed.html
├── dorori_obd.py
├── dorori_obd.html
├── best_selection.pt
├── best노면.pt
├── requirements.txt
└── README_KO.md
```

모델이나 HTML을 다른 위치에 둘 수도 있지만, 그 경우 아래 환경변수로 경로를 지정해야 합니다.

```bash
SEG_MODEL_PATH=/절대경로/best_selection.pt
CLASSIFY_MODEL_PATH=/절대경로/best노면.pt
```

---

## 3. Raspberry Pi 설치

Raspberry Pi OS 계열에서 권장하는 설치 예시입니다.

```bash
sudo apt update
sudo apt install -y python3-venv python3-opencv python3-gpiozero python3-lgpio

cd ~/esw/dorori_motor_bundle
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt`에는 Flask, Ultralytics, python-OBD 패키지가 들어 있습니다. OpenCV와 GPIO 패키지는 Raspberry Pi OS 패키지를 사용하도록 위에서 `apt`로 설치합니다.

---

## 4. A4988 배선

코드는 사용자가 제시한 BCM 핀 맵을 그대로 사용합니다.

| Raspberry Pi BCM | 물리 핀 | A4988 |
|---:|---:|---|
| GPIO17 | 11 | STEP |
| GPIO27 | 13 | DIR |
| GPIO22 | 15 | ENABLE |
| GND | 예: 6 | Logic GND와 공통 접지 |

중요 배선 조건:

- `ENABLE`은 Active-Low입니다. 코드의 `enable_pin.on()`이 실제 GPIO LOW, 즉 드라이버 활성화입니다.
- A4988 `VMOT`에는 모터 전원을 공급하고, Raspberry Pi 5V 핀으로 모터를 구동하지 않습니다.
- Raspberry Pi GND, A4988 logic GND, 모터 전원 GND는 기준 전위를 공유해야 합니다.
- `RESET`과 `SLEEP`은 함께 묶어 VDD에 올려 두는 구성이 일반적입니다.
- VMOT-GND 전해 커패시터는 드라이버 가까이에 배치합니다.
- 6선 스테퍼 모터를 A4988에 연결할 때는 각 코일의 양 끝 4선만 사용하고 센터 탭 2선은 연결하지 않습니다.
- A4988 전류 제한을 모터 정격에 맞게 먼저 설정하십시오.

### 1600펄스가 실제 한 바퀴인지 확인

`MOTOR_STEPS_PER_REV=1600`은 사용자가 제시한 기존 시험값을 그대로 사용한 것입니다. 실제 한 바퀴는 모터 기본 스텝각과 A4988의 MS1/MS2/MS3 마이크로스텝 설정에 따라 달라집니다. 기구를 연결하기 전에 무부하 상태에서 정확히 한 바퀴인지 확인하십시오.

회전 방향이 반대이면 배선을 바꾸지 않고 다음처럼 실행할 수 있습니다.

```bash
MOTOR_DEPLOY_DIRECTION=0 python dorori_debug_speed.py
```

---

## 5. 디버그 버전 실행

### 5-1. GPIO 없이 전체 UI/상태머신 먼저 시험

```bash
cd ~/esw/dorori_motor_bundle
source .venv/bin/activate
MOTOR_DRY_RUN=1 python dorori_debug_speed.py
```

`MOTOR_DRY_RUN=1`에서는 실제 GPIO를 건드리지 않지만, 설정한 펄스 시간과 5초 유지 시간을 실제로 진행하며 화면 상태를 확인할 수 있습니다.

### 5-2. 실제 모터 연결 후 실행

```bash
python dorori_debug_speed.py
```

브라우저에서 다음 주소를 엽니다.

```text
http://라즈베리파이_IP:5000
```

라즈베리파이 IP 확인:

```bash
hostname -I
```

HTML 우측 상단 톱니 버튼에서 속도를 `10`, `3`, `0 km/h` 등으로 바꿀 수 있습니다. 속도만 수동이며 카메라와 두 모델은 실제로 동작합니다.

---

## 6. OBD-II 실사용 버전 실행

### 6-1. 포트 확인

USB 어댑터를 연결한 후 다음을 확인합니다.

```bash
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null
ls -l /dev/serial/by-id/ 2>/dev/null
```

CH340 계열은 보통 `/dev/ttyUSB0`로 나타나지만, `/dev/serial/by-id/...` 경로를 사용하면 재부팅 후에도 더 안정적으로 같은 장치를 지정할 수 있습니다.

현재 사용자를 serial 장치 접근 그룹에 추가합니다.

```bash
sudo usermod -aG dialout "$USER"
```

그 뒤 로그아웃/로그인 또는 재부팅하여 그룹 변경을 적용합니다.

### 6-2. 자동 포트 검색

```bash
cd ~/esw/dorori_motor_bundle
source .venv/bin/activate
python dorori_obd.py
```

### 6-3. 포트 직접 지정

```bash
OBD_PORT=/dev/ttyUSB0 python dorori_obd.py
```

안정적인 by-id 경로 예시:

```bash
OBD_PORT=/dev/serial/by-id/usb-장치이름 python dorori_obd.py
```

복제 ELM327에서 빠른 명령 최적화가 문제를 만들 경우:

```bash
OBD_FAST=0 OBD_TIMEOUT_SEC=1.0 OBD_PORT=/dev/ttyUSB0 python dorori_obd.py
```

서버는 표준 Vehicle Speed 명령을 주기적으로 질의합니다. 연속 응답 실패, 연결 해제, 또는 설정 시간 이상 속도 데이터가 갱신되지 않으면 `FAULT`로 전환합니다.

---

## 7. 주요 환경변수

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `STEP_PIN` | `17` | BCM STEP GPIO |
| `DIR_PIN` | `27` | BCM DIR GPIO |
| `ENABLE_PIN` | `22` | BCM ENABLE GPIO, Active-Low |
| `MOTOR_STEPS_PER_REV` | `1600` | 전개/수납 한 방향 펄스 수 |
| `MOTOR_FREQUENCY_HZ` | `300` | STEP 펄스 주파수 |
| `MOTOR_HOLD_SEC` | `5.0` | 전개 후 유지 시간 |
| `MOTOR_DEPLOY_DIRECTION` | `1` | `1`/`0`으로 전개 방향 반전 |
| `MOTOR_HOLD_TORQUE` | `1` | 유지 중 ENABLE 유지 여부 |
| `MOTOR_DRY_RUN` | `0` | `1`이면 GPIO 없이 시뮬레이션 |
| `CREEP_MAX_KMH` | `5.0` | 판단을 수행하는 최대 저속 |
| `STOP_SPEED_EPSILON_KMH` | `0.1` | 정차로 간주하는 속도 상한 |
| `STOP_DEPLOY_DELAY_SEC` | `1.0` | 정차+YELLOW 후 전개 지연 |
| `VISION_INTERVAL_SEC` | `1.0` | 비전 추론 주기 |
| `VISION_STALE_SEC` | `5.0` | 비전 결과 유효 시간 |
| `CAMERA_SOURCE` | `0` | OpenCV 카메라 번호 또는 스트림 경로 |
| `SEG_IMGSZ` | `640` | segmentation 입력 크기 |
| `CLASSIFY_IMGSZ` | `224` | 노면 분류 입력 크기 |
| `SEG_CONF` | `0.25` | segmentation confidence threshold |
| `OBD_PORT` | 자동 | OBD serial 포트 |
| `OBD_FAST` | `1` | python-OBD fast 모드 |
| `OBD_TIMEOUT_SEC` | `0.2` | OBD 질의 timeout |
| `OBD_POLL_SEC` | `0.2` | 속도 질의 주기 |
| `OBD_STALE_SEC` | `1.5` | 속도 데이터 유효 시간 |
| `PORT` | `5000` | 웹 서버 포트 |

예시:

```bash
MOTOR_STEPS_PER_REV=1600 \
MOTOR_FREQUENCY_HZ=300 \
MOTOR_HOLD_SEC=5 \
CAMERA_SOURCE=0 \
python dorori_obd.py
```

### 유지 토크 설정

기본값 `MOTOR_HOLD_TORQUE=1`은 발판을 5초간 유지하는 동안 A4988을 계속 활성화하여 정지 토크를 유지합니다. 그만큼 모터와 드라이버가 더 뜨거워질 수 있습니다. 기구가 자체 잠금 또는 기계식 지지 구조로 위치를 유지한다면 다음처럼 유지 중 출력을 끌 수 있습니다.

```bash
MOTOR_HOLD_TORQUE=0 python dorori_obd.py
```

하중을 받는 발판에서 모터 정지 토크만으로 사람의 체중을 지지하는 구조는 사용하지 마십시오.

---

## 8. API 요약

공통:

- `POST /api/start`: 판단 활성화
- `POST /api/stop`: 판단 종료 및 가능한 즉시 수납 요청
- `GET /api/vision`: UI용 통합 판단/속도/모터 상태
- `GET /api/status`: 모터 상태
- `GET /api/system`: 전체 내부 상태
- `POST /api/deploy`: 정차·최신 YELLOW·미실행 조건을 모두 통과할 때만 수동 사이클 시작
- `POST /api/retract`: 전개/유지 중 조기 수납
- `GET /api/health`: 카메라·모델·모터 준비 상태

디버그 버전만:

```bash
curl -X POST http://127.0.0.1:5000/api/debug \
  -H 'Content-Type: application/json' \
  -d '{"speed_kmh": 0}'
```

```bash
curl http://127.0.0.1:5000/api/debug
```

---

## 9. 실제 발판 장착 전 필수 안전 조건

현재 핀 맵에는 리미트 스위치나 절대 위치 센서가 없으므로 이 코드는 **open-loop 펄스 수**로만 위치를 추정합니다. 전원이 꺼진 상태에서 발판이 움직였거나 스텝 손실이 발생하면 소프트웨어의 위치 가정과 실제 위치가 달라집니다.

사람이 올라가는 실제 발판으로 시험하기 전 최소한 다음을 추가해야 합니다.

- 전개 끝/수납 끝 리미트 스위치 또는 위치 센서
- 물리적 스토퍼와 체중을 받는 기계식 잠금/지지 구조
- 모터 과전류·끼임 감지 또는 별도 보호 회로
- 운전자가 즉시 차단할 수 있는 비상 정지
- 문 열림, 기어 위치, 주차 브레이크 등 차량 측 하드웨어 인터록
- 전원 인가 시 수납 원점 확인 또는 homing 절차

리미트 스위치가 없는 현재 버전은 반드시 **발판이 완전히 수납된 알려진 시작 위치**에서 켜고, 처음에는 무부하·저속·짧은 이동량으로 검증하십시오.

또한 Flask 서버는 `0.0.0.0:5000`으로 열리며 별도 인증이 없습니다. 차량 내부의 신뢰할 수 있는 로컬 네트워크에서만 사용하십시오. GPIO를 한 프로세스만 소유하도록 서버를 여러 worker로 복제해 실행하지 마십시오.
