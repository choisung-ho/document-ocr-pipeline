# 문서 전처리·OCR 테스트 앱

HWPX, DOCX, PDF와 이미지 파일을 Markdown과 정규화 JSON으로 변환하는 테스트 앱입니다. 원문 구조를 우선 보존하고 OCR이 필요한 페이지·영역·이미지만 `PaddleOCR-VL-1.6`으로 처리합니다. 다른 OCR 모델은 사용하지 않습니다.

## 실행

저장소를 내려받은 뒤 프로젝트 폴더에서 실행합니다:

```bash
git clone https://github.com/choisung-ho/document-ocr-pipeline.git
cd document-ocr-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-ocr.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다.

## 처리 정책

- OCR은 이미지가 있는 페이지·영역에만 수행합니다. 이미지가 없는 텍스트 페이지는 원문 추출만 하고 OCR을 생략합니다(`native` 모드).
- 이미지 유무는 페이지 자원 목록으로 먼저 빠르게 확인하고, 이미지가 있을 때만 정밀 영역 분석을 수행합니다. OCR이 없는 페이지는 고해상도(2배) 렌더링 대신 1배 미리보기만 생성합니다.
- PDF 처리는 2단계 파이프라인으로 동작합니다: 프리페치 스레드가 다음 페이지들의 분석·렌더링·원문 추출을 미리 수행하고(최대 3페이지 선행), 메인 루프는 OCR·병합만 담당합니다. OCR 추론 중에도 분석이 병렬로 진행되어 대기 시간이 줄어듭니다. OCR 추론 자체는 단일 파이프라인 보호를 위해 직렬로 유지됩니다.
- 스캔 PDF: 텍스트 레이어가 없고 이미지가 있는 페이지 전체를 PaddleOCR-VL-1.6으로 처리합니다. 텍스트·이미지가 모두 없는 빈 페이지는 OCR을 생략합니다.
- 텍스트 PDF: 원문 텍스트를 직접 추출하고 이미지 영역만 PaddleOCR-VL-1.6으로 처리합니다. 표 영역 OCR은 기본 비활성화이며 `OCR_INCLUDE_TABLE_REGIONS=1`로 켤 수 있습니다(표 텍스트 자체는 원문 추출로 확보됩니다).
- 혼합 PDF: 페이지마다 원문 추출 또는 전체 OCR 방식을 자동으로 선택합니다.
- DOCX: python-docx로 문단·제목·표를 구조화하고 포함 이미지만 OCR합니다.
- HWPX: XML에서 문단·표를 구조화하고 포함 이미지만 OCR합니다.
- PNG/JPG/BMP/TIFF/WEBP: 이미지 전체를 OCR합니다.
- PDF 이미지·차트: 이미지 블록 OCR과 차트 인식을 활성화하여 구조화 결과에 포함합니다.
- 표: 텍스트와 Markdown 표를 함께 보존합니다.
- 이미지: OCR 결과에 원본 자산 URL을 연결하고, 이미지 내 인식 텍스트를 `description`으로 매핑합니다.
- 차트: 차트 인식 결과에서 X축(항목)·Y축(계열)·데이터 행을 `chart`로 구조화하고 `description`을 생성합니다.
- 수치: `87.5%`, `5,000,000원`처럼 단위가 붙은 수치를 `metrics`(raw/value/unit)로 추출합니다. 전각 숫자(％ 등)는 반각으로 정규화합니다.
- 저해상도 이미지: 긴 변이 기준(기본 1600px)보다 작으면 OCR 전에 확대 렌더링합니다.
- OCR 추론은 기본적으로 별도 프로세스에서 실행됩니다. 네이티브 라이브러리(libpaddle)가 비정상 종료(세그폴트)해도 서버는 유지되고, 워커를 재기동해 1회 재시도한 뒤 실패 시 해당 페이지·이미지만 오류로 표시하고 계속 진행합니다. `OCR_SUBPROCESS=0`으로 끄면 서버 프로세스 안에서 추론합니다.
- DOCX/HWPX 내부 이미지는 이미지 단위로 실패를 허용합니다. 한 이미지의 OCR이 실패해도 원문과 나머지 이미지는 정상 처리됩니다.
- 증빙 문서: `rag_eligible=false`로 표시하여 후속 임베딩 대상에서 제외할 수 있습니다.
- 업로드 원본: 기본적으로 처리 완료 후 삭제합니다. 결과와 이미지 자산은 기본 24시간 보존합니다.

## 결과 형식

```json
{
  "metadata": {
    "doc_id": "...",
    "filename": "보고서.pdf",
    "title": "보고서",
    "document_type": "general",
    "rag_eligible": true,
    "security_level": "internal",
    "access_groups": ["planning"]
  },
  "results": [
    {
      "page": 1,
      "type": "heading",
      "text": "Ⅰ. 사업 추진 개요",
      "markdown": "## Ⅰ. 사업 추진 개요",
      "section_path": ["Ⅰ. 사업 추진 개요"],
      "source_text": "Ⅰ. 사업 추진 개요",
      "bbox": [72.0, 80.0, 240.0, 104.0],
      "asset_url": null,
      "description": null,
      "metrics": null,
      "chart": null
    },
    {
      "page": 2,
      "type": "chart",
      "text": "연도 취업률 2024 68% 2025 72%",
      "markdown": "| 연도 | 취업률 |\n| --- | --- |\n| 2024 | 68% |\n| 2025 | 72% |",
      "description": "차트 데이터 2행 · X축(항목) '연도' · Y축(계열) '취업률'",
      "metrics": [{"raw": "68%", "value": 68.0, "unit": "%"}, {"raw": "72%", "value": 72.0, "unit": "%"}],
      "chart": {"x_axis": "연도", "series": ["취업률"], "rows": [["2024", "68%"], ["2025", "72%"]]}
    }
  ]
}
```

## CPU·GPU 환경

개발용 CPU 또는 Apple Silicon 환경:

```bash
pip install -r requirements-ocr.txt
```

NVIDIA GPU 서버에서는 CPU용 파일과 함께 설치하지 말고 CUDA 버전에 맞는 PaddlePaddle GPU 런타임을 사용합니다.

```bash
pip install -r requirements-ocr-gpu.txt
PADDLE_DEVICE=gpu uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`PaddleOCRVL` 초기화 시 `pipeline_version="v1.6"`과 레이아웃·차트·이미지 OCR·구조화 Markdown 옵션이 고정됩니다.

## 환경 변수

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `PADDLE_DEVICE` | `cpu` | `cpu`, `gpu` 등 Paddle 장치 |
| `OCR_MAX_UPLOAD_MB` | `100` | 최대 업로드 크기 |
| `OCR_RETENTION_HOURS` | `24` | 결과·자산 보존 시간 |
| `OCR_DELETE_SOURCE_AFTER_PROCESS` | `1` | 처리 후 업로드 원본 삭제 여부 |
| `OCR_EXPECTED_PAGE_SECONDS` | `120` | 페이지 내부 OCR 예상 진행률 계산 기준 시간(초) |
| `OCR_MIN_IMAGE_LONG_EDGE` | `1600` | 이미지 OCR 전 확대 기준(긴 변 px). `0`이면 비활성화 |
| `OCR_INCLUDE_TABLE_REGIONS` | `0` | `1`이면 텍스트 PDF의 표 영역도 OCR 대상에 포함 |
| `OCR_SUBPROCESS` | `1` | OCR 추론을 별도 프로세스에서 실행(크래시 격리). `0`이면 서버 내 실행 |
| `OCR_INFERENCE_TIMEOUT_SECONDS` | `900` | 페이지·이미지 1건당 추론 제한 시간(초). `0`이면 무제한 |
| `OCR_MOCK` | 미설정 | `1`이면 스캔 페이지 OCR을 모의 결과로 대체 |

## 개발용 모의 실행

```bash
OCR_MOCK=1 uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## 구조와 검증

- `app/main.py`: 업로드·작업 관리·페이지별 처리·결과 API
- `app/document_parsers.py`: 문서별 파싱·원문 추출·구조화
- `app/paddle_worker.py`: OCR 추론 프로세스
- `app/templates/`, `app/static/`: 작업 진행과 결과를 보여주는 화면

```bash
python -m unittest discover -s tests -v
```

기존 테스트는 합성 문서와 모의 OCR을 사용해 원문 추출·영역 처리·실패 격리 흐름을 확인합니다. 실제 OCR 모델의 정확도·속도를 측정하는 테스트는 아닙니다.

## 공개 범위

원본 업로드 문서, OCR 결과 JSON, 추출 이미지, 개인 설정은 포함하지 않습니다. 실행 시 생기는 `uploads/`, `job_results/`, `assets/`는 Git에서 제외합니다. 원래 프로젝트의 문서 전처리·OCR 기능을 공개용으로 분리한 저장소입니다.

공개 준비 시 실행한 검사와 범위는 [검증 기록](docs/PUBLIC_VALIDATION.md)에 정리했습니다.
