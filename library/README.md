# ocrpipe — 문서 OCR 파이프라인 (PaddleOCR-VL)

PDF·HWPX·이미지에서 **본문은 무손실로 추출**하고 **그림 속 텍스트만 OCR** 하는 라이브러리입니다.
RAG 파이프라인 등 다른 코드에서 `import` 해서 쓰는 것을 전제로 합니다.

```python
from ocrpipe import analyze

blocks = analyze("보고서.hwpx")           # 본문(무손실) + 그림 속 텍스트(OCR)
blocks = analyze("문서.pdf", pages="1,3-5")
blocks = analyze("사진.png")
```

## 설계 원칙

| 원칙 | 내용 |
|------|------|
| **엔진 1개** | PaddleOCR-VL 단독 (레이아웃 검출 + VLM 파싱으로 표·수식까지 구조화) |
| **본문 무손실** | PDF 텍스트 레이어 / HWPX `section*.xml` 에서 **원문 그대로** (score 1.0, 인식 오차 0) |
| **OCR 최소화** | 인식이 필요한 **이미지만** 골라서 OCR |
| **변환 없음** | HWPX 는 zip 이라 **PDF 변환 불필요** — 한컴/LibreOffice 의존성 0 |
| **라이브러리** | 웹 UI·REST 서버 없음. `analyze()` 함수와 CLI 만 |

## 설치

설치는 **2단계**입니다. `paddlepaddle`(딥러닝 프레임워크 본체)는 CPU/GPU 빌드가 갈리고
CUDA 버전마다 배포 주소가 달라, pip 의존성으로 고정할 수 없어 **따로** 설치합니다.

```powershell
# 1) 가상환경 + ocrpipe (paddleocr 포함)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt      # 또는: pip install -e .

# 2) paddlepaddle 본체 — 아래 중 환경에 맞는 하나만
#    GPU (CUDA 12.9 빌드 — RTX 50xx/Blackwell 포함)
pip install --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/ paddlepaddle-gpu==3.2.2
#    CPU
pip install paddlepaddle==3.2.2
```

| 파일 | 역할 |
|------|------|
| `pyproject.toml` | 버전 **범위**의 단일 출처. 패키지 정의(`pip install -e .`) |
| `requirements.txt` | 동작을 확인한 **고정 버전** 스냅샷(lock 성격). 팀원이 같은 환경 재현할 때 |

설치 확인:

```powershell
python -c "from ocrpipe.ocr import engine_available; print(engine_available())"
#   → (True, '설치됨')

python -c "from ocrpipe.ocr.device import describe; print(describe())"
#   → nvidia=True, paddle=gpu   (GPU 경로)
```

### 라이브러리 버전

| 구성요소 | 필요 버전 | 검증된 버전 | 비고 |
|----------|-----------|-------------|------|
| Python | `>=3.10` | 3.11.9 | |
| `paddleocr[doc-parser]` | `>=3.6` | — | **v1.6 모델 파이프라인 지원 버전**(3.6.0 = 2026-05-28). 최신은 3.7.0 |
| `paddlepaddle` / `paddlepaddle-gpu` | `>=3.2.1` | 3.2.2 | GPU는 CUDA 12.9 빌드. RTX 5070(sm_120)에서 동작 확인 |
| `pymupdf` | `>=1.24` | 1.26.3 | PDF 렌더링·텍스트 레이어 |
| `opencv-python` | `>=4.10` | 4.13.0 | 전처리 |
| `numpy` | `>=1.26` | 2.4.6 | |
| `pillow` | `>=10` | — | |

- `doc-parser` extra 가 **필수**입니다. 이게 없으면 VLM 문서 파싱(표·수식 구조화)이 동작하지 않습니다.
- **torch 는 필요 없습니다.** 예전 Surya 엔진용이었고, 지금은 `paddle_vl.py` 가 torch 미설치 환경에서도
  import 되도록 최소 stub 을 주입합니다. 새 환경에 설치하지 마세요(수 GB 낭비).

### OCR 모델 다운로드 방식

모델 가중치는 **패키지에 포함되지 않고, 첫 실행 때 자동 다운로드**됩니다(수백 MB~, 인터넷 필요).
따라서 첫 `analyze()` 호출은 수 분이 걸릴 수 있고, 이후에는 캐시를 재사용해 빨라집니다.

| 항목 | 내용 |
|------|------|
| 받는 모델 | VLM `PaddleOCR-VL-1.6-0.9B` + 레이아웃 `PP-DocLayoutV3` |
| 시점 | 엔진 첫 초기화 시 (`PaddleOCRVL()` 생성) — 지연 로드라 import 만으로는 안 받음 |
| 캐시 위치 | `~/.paddlex/official_models/` (Windows: `C:\Users\<사용자>\.paddlex\official_models\`) |
| 다운로드 소스 | 기본 **Hugging Face**. `PADDLE_PDX_MODEL_SOURCE` 로 변경 가능 (`huggingface`/`modelscope`/`aistudio`/`bos`) |

```powershell
# 사내망에서 Hugging Face 가 막혔을 때 (중국 미러/BOS 로 전환)
$env:PADDLE_PDX_MODEL_SOURCE = "bos"
```

**모델 버전 지정** — 기본은 설치된 `paddleocr` 의 기본값(현재 `v1.6`)입니다.
재현성을 위해 고정하려면 환경변수로 못박으세요.

```powershell
$env:OCR_VL_VERSION = "v1.6"     # 선택 가능: v1 / v1.5 / v1.6 (paddleocr 버전에 따름)
```

**오프라인·폐쇄망**: 인터넷 되는 곳에서 한 번 실행해 캐시를 채운 뒤 `~/.paddlex/` 폴더를 그대로
복사하는 방식이 가장 간단합니다. 단 일부 버전은 로컬 캐시가 있어도 먼저 네트워크를 확인하려
드는 사례가 보고돼 있으니, 폐쇄망 배포 전에 **네트워크를 끊고 한 번 실제로 돌려서** 확인하세요.

## 처리 흐름

```
입력 ─┬─ PDF   ─→ 페이지별: 텍스트 레이어 무손실 추출 ─┐
      │                  + 삽입 이미지 크롭 ──────────┤
      ├─ HWPX  ─→ section*.xml 문단 무손실 추출 ──────┤→ 이미지 2단계 OCR → 병합·후처리
      │                  + BinData/ 이미지 원본 ──────┤
      └─ 이미지 ─→ 전체 OCR ───────────────────────────┘
```

**이미지 2단계** (`ocrpipe/stages/`)

1. **1단계 분리** — 문서에서 이미지를 꺼내 `1_images/` 에 저장
   (PDF 는 렌더링된 페이지에서 크롭, HWPX 는 `BinData/` 에서 **원본 그대로** — 재렌더링 손실 없음)
2. **2단계 OCR** — 각 이미지를 전처리(확대·대비) 후 인식, 저신뢰 블록은 재인식.
   결과로 분류까지 겸합니다: 글자가 나오면 `2_text/`, 없으면(로고·장식) `2_dummy/`

> 예전에는 검출(detection)로 먼저 걸러내는 3단계였습니다. PaddleOCR-VL 은 검출 전용 API 가
> 없어 그 방식이면 **이미지당 VLM 을 두 번** 돌리게 되므로, 한 번 돌리고 결과로 나눕니다.

텍스트 레이어가 없는 **스캔 PDF 페이지**는 페이지 전체를 이미지 1장으로 보고 OCR 합니다.
중간 산출물은 기본 보존되며 `OCR_KEEP_STAGE_FILES=0` 으로 끄면 임시 폴더를 쓰고 정리합니다.

## 라이브러리 API

```python
from ocrpipe import analyze, analyze_to_text

blocks = analyze(source, *, pages="", suffix=".png", as_dict=True)
text   = analyze_to_text("사진.png")     # 순수 텍스트만 필요할 때
```

- `source` — 파일 경로 또는 이미지 `bytes`
- `pages` — PDF 페이지 필터(`"1,3-5"`). HWPX·이미지는 무시
- 반환 — 블록 dict 리스트 (`as_dict=False` 면 `OcrItem` 데이터클래스)

### 결과 JSON 형식

```json
[
  {
    "page": 1,
    "type": "heading",
    "text": "Ⅰ. 사업 추진 개요",
    "bbox": [120, 80, 540, 110],
    "order": 0,
    "html": null,
    "markdown": null,
    "score": 1.0,
    "crop": null
  }
]
```

- `type` — `heading` / `paragraph` / `table` / `formula` / `image` / `markdown` 등으로 정규화
- `score` — `1.0` 은 **원문 무손실 추출**, 그 외는 OCR 신뢰도
- `bbox` — PDF 는 페이지 좌표(렌더링 픽셀). **HWPX 는 본문 `null`**(흐름 문서라 페이지 좌표 없음),
  그림에서 인식한 블록은 **그 이미지 내부 좌표**
- `html` / `markdown` — 표·문서 구조 결과(PaddleOCR-VL 제공 시)

## CLI

```powershell
python -m ocrpipe.ocr_cli 보고서.hwpx
python -m ocrpipe.ocr_cli "문서\시스템구성도.png"
python -m ocrpipe.ocr_cli input.pdf --pages 1,3-4 -o result.json
```

미지정 시 결과는 `OCR_result/<원본이름>_paddleocr-vl.json` 에 저장됩니다.

## 지원 형식

| 형식 | 처리 | 비고 |
|------|------|------|
| `.pdf` | 텍스트 레이어 + 이미지 OCR | 스캔 PDF 는 페이지 전체 OCR |
| `.hwpx` | 본문 XML + BinData 이미지 | **변환 도구 불필요** |
| 이미지 | 전체 OCR | png·jpg·jpeg·bmp·tif·tiff·webp |
| ~~`.hwp`~~ | **미지원** | zip 이 아니라 직접 파싱 불가 → 한컴에서 `.hwpx` 로 저장해 사용 |
| ~~`.doc(x)`~~ | **미지원** | 필요하면 PDF 로 저장해 사용 |

## 실제 정확도(CER) 측정

신뢰도(score)는 모델 자기 보고 값이므로, 실제 오류율은 정답셋과 비교해 측정합니다.

```powershell
# 1) 정답 초안 생성 (텍스트 레이어 기준 — 그래프 안 텍스트는 사람이 추가·검수)
python -m ocrpipe.eval_cer --make-drafts 문서.pdf --pages 14,48,50

# 2) 검수 후 초안 상단의 # 주석을 지우고 측정
python -m ocrpipe.eval_cer OCR_result\문서_paddleocr-vl.json
```

표·다단 페이지의 순서 차이에 오염되지 않도록 **줄 정렬 매칭 CER**
(정답 각 줄을 가장 비슷한 결과 블록과 짝지어 편집거리 합산)을 사용합니다.

## 모듈 구조

```
ocrpipe/
├── analyze.py          # 공개 API: analyze() / analyze_to_text()
├── ocr_cli.py          # 명령줄 진입점
├── pipeline.py         # 오케스트레이션 (PDF 페이지 / HWPX / 단일 이미지)
├── hwpx.py             # HWPX 직접 파서 (본문 XML + BinData 이미지)
├── stages/
│   ├── stage1_extract.py   # 1단계: 이미지 분리
│   └── stage2_ocr.py       # 2단계: 이미지 OCR (+ text/dummy 분류)
├── ocr/
│   ├── paddle_vl.py        # PaddleOCR-VL 엔진
│   ├── base.py             # 엔진 인터페이스 + 결과 정규화
│   ├── preprocess.py       # 업스케일·대비·배경 정규화
│   ├── refine.py           # 저신뢰 블록 재인식
│   └── device.py           # GPU/CPU 결정
├── config.py           # 설정 단일 출처 (모든 환경변수는 여기서만 읽음)
├── document_utils.py   # PDF 렌더링·페이지 필터
└── eval_cer.py         # 실제 정확도(CER) 측정
```

### 의존 계층 (위 → 아래로만 의존, 순환 없음)

```
진입   analyze.py · ocr_cli.py
  │
흐름   pipeline.py
  │
단계   stages/ (1_extract → 2_ocr)          기반   hwpx.py
  │                                          │
엔진   ocr/ (paddle_vl + preprocess·refine·device·base)
  │
기반   config.py · document_utils.py
```

- **설정 단일 출처**: 모든 환경변수는 `config.py` 의 `settings` 만 읽습니다
  (장치 "결정"은 `ocr/device.py`, 설정 "값"은 `config.py` — 관심사 분리).
- **정직한 import**: 모든 의존은 파일 상단 import 로 드러납니다. 무거운 라이브러리(paddle)만
  엔진 함수 안에서 지연 로드합니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OCR_DEVICE` | (자동) | 장치 강제: `cuda`/`gpu` 또는 `cpu`. 미지정 시 GPU 있으면 GPU |
| `PADDLE_DEVICE` | (자동) | Paddle 장치(`gpu`/`cpu`). 지정 시 `OCR_DEVICE` 보다 우선 |
| `OCR_VL_VERSION` | (패키지 기본) | PaddleOCR-VL 모델 버전 고정: `v1` / `v1.5` / `v1.6` |
| `OCR_HYBRID` | `1` | 본문 무손실 추출 + 이미지만 OCR on/off (0이면 페이지 전체 OCR) |
| `OCR_KEEP_STAGE_FILES` | `1` | 중간 산출물(`1_images`·`2_text`·`2_dummy`) 폴더 보존 |
| `OCR_MIN_NATIVE_CHARS` | `40` | 이 글자 수 미만이면 스캔 페이지로 간주 |
| `OCR_MIN_REGION_SIDE` | `50` | 짧은 변이 이보다 작은 이미지는 제외(픽셀) |
| `OCR_DPI` | `300` | PDF 페이지 렌더링 DPI |
| `OCR_PAGES` | (없음) | 기본 페이지 필터, 예 `2` 또는 `2,4-5` |
| `OCR_PREPROCESS` | `1` | 이미지 전처리(업스케일·대비·샤프닝) on/off |
| `OCR_PREPROCESS_BG` | `1` | 배경 조명/색 정규화 on/off |
| `OCR_PREPROCESS_SHORT` | `2000` | 업스케일 목표 짧은 변 픽셀 |
| `OCR_REFINE` | `1` | 저신뢰 블록 2차 재인식 on/off |
| `OCR_REFINE_THRESHOLD` | `0.80` | 재인식 대상 신뢰도 상한 |
| `OCR_REFINE_MAX` | `24` | 페이지당 최대 재인식 블록 수 |
| `OCR_NOISE_FILTER` | `1` | 도형 오인식 블록 제거 on/off |
| `OCR_RESULT_DIR` | `OCR_result/` | CLI 결과 JSON 저장 폴더 |

PaddleOCR 자체가 읽는 변수(참고용, `ocrpipe` 가 관여하지 않음):

| 변수 | 설명 |
|------|------|
| `PADDLE_PDX_MODEL_SOURCE` | 모델 다운로드 소스: `huggingface`(기본)/`modelscope`/`aistudio`/`bos` |

## 다른 환경에서 실행

위 설치 절차에 따라 Python 가상환경과 CPU/GPU 런타임을 준비하고, 모델·입력 문서·출력 경로를 해당 환경에 맞게 설정합니다. 실제 입력 문서와 결과 파일은 소스와 별도로 관리합니다.
