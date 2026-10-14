"""설정 단일 출처(Single Source of Truth).

프로젝트의 모든 ``OCR_*`` 및 엔진별 환경변수를 **여기서만** 읽습니다. 다른 모듈은
``from .config import settings`` (엔진은 ``from ..config import settings``)로 가져다
쓰며, 직접 ``os.getenv`` 를 호출하지 않습니다. 값은 접근할 때마다 환경변수를 다시
읽으므로 런타임에 바꿔 테스트할 수 있습니다.

설정 구성:
- 렌더링 / 파이프라인(하이브리드·2단계) / 전처리 / 후처리 / 저장
- 장치(device) / PaddleOCR

장치 "결정" 로직(GPU 있으면 GPU 등)은 :mod:`ocrpipe.ocr.device` 가 담당하고,
여기서는 사용자가 준 원본 환경변수 값만 노출합니다(설정과 로직의 분리).
"""

from __future__ import annotations

import os
from pathlib import Path


def _flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


class Settings:
    """환경변수 래퍼. 프로퍼티는 접근 시점에 os.environ 을 읽는다."""

    # ============================ 공통 파이프라인 ============================ #

    # --- 렌더링 --------------------------------------------------------- #
    @property
    def dpi(self) -> int:
        return int(os.getenv("OCR_DPI", "300"))

    @property
    def pages(self) -> str:
        return os.getenv("OCR_PAGES", "")

    # --- 하이브리드 / 2단계 파이프라인 ------------------------------------ #
    @property
    def hybrid(self) -> bool:
        """본문 텍스트레이어 무손실 추출 + 이미지만 2단계 OCR."""
        return _flag("OCR_HYBRID")

    @property
    def min_native_chars(self) -> int:
        """이 글자 수 미만이면 텍스트 레이어 없음(스캔 페이지)으로 간주."""
        return int(os.getenv("OCR_MIN_NATIVE_CHARS", "40"))

    @property
    def min_region_side(self) -> int:
        """이보다 작은 이미지(아이콘 등)는 이미지 분리에서 제외(렌더링 픽셀)."""
        return int(os.getenv("OCR_MIN_REGION_SIDE", "50"))

    @property
    def keep_stage_files(self) -> bool:
        """중간 산출물(분리 이미지·text/dummy 분류)을 폴더로 남길지."""
        return _flag("OCR_KEEP_STAGE_FILES")

    # --- 전처리 --------------------------------------------------------- #
    @property
    def preprocess(self) -> bool:
        return _flag("OCR_PREPROCESS")

    @property
    def preprocess_bg(self) -> bool:
        return _flag("OCR_PREPROCESS_BG")

    @property
    def preprocess_short(self) -> int:
        return int(os.getenv("OCR_PREPROCESS_SHORT", "2000"))

    # --- 후처리(재인식·노이즈 필터) --------------------------------------- #
    @property
    def refine(self) -> bool:
        return _flag("OCR_REFINE")

    @property
    def refine_threshold(self) -> float:
        try:
            return float(os.getenv("OCR_REFINE_THRESHOLD", "0.80"))
        except ValueError:
            return 0.80

    @property
    def refine_max(self) -> int:
        try:
            return int(os.getenv("OCR_REFINE_MAX", "24"))
        except ValueError:
            return 24

    @property
    def noise_filter(self) -> bool:
        return _flag("OCR_NOISE_FILTER")

    # --- 저장 위치 ------------------------------------------------------ #
    @property
    def result_dir(self) -> Path:
        root = Path(__file__).resolve().parent.parent
        return Path(os.getenv("OCR_RESULT_DIR", str(root / "OCR_result")))

    # ============================== 엔진 설정 =============================== #
    # 원본 환경변수 값만 노출한다. "GPU/CPU 결정" 은 ocrpipe.ocr.device 가 수행.

    @property
    def device_override(self) -> str:
        """OCR_DEVICE: '' (자동) / 'cuda'|'gpu' / 'cpu'."""
        return os.getenv("OCR_DEVICE", "").strip().lower()

    @property
    def paddle_device_override(self) -> str:
        """PADDLE_DEVICE: '' (자동) / 'gpu' / 'cpu'. 지정 시 OCR_DEVICE 보다 우선."""
        return os.getenv("PADDLE_DEVICE", "").lower()

    @property
    def vl_version(self) -> str:
        """OCR_VL_VERSION: PaddleOCR-VL 파이프라인(=모델) 버전. 예 'v1.6'.

        빈 값(기본)이면 설치된 paddleocr 패키지의 기본 버전을 그대로 씁니다.
        지정하면 ``PaddleOCRVL(pipeline_version=...)`` 로 전달합니다.
        (선택 가능 값은 paddleocr 버전에 따름: 'v1' / 'v1.5' / 'v1.6')
        """
        return os.getenv("OCR_VL_VERSION", "").strip()


settings = Settings()
