"""이미지 OCR 2단계 파이프라인.

문서에서 이미지(사진·도표)만 골라 OCR 하기 위한 명시적 단계입니다. 각 단계는 독립
모듈이며 중간 산출물을 단계별 폴더로 남길 수 있습니다.

- 1단계 :mod:`stage1_extract` — 페이지에서 삽입 이미지를 분리 → ``1_images/``
- 2단계 :mod:`stage2_ocr`     — 분리된 이미지를 OCR (결과로 ``2_text/``·``2_dummy/`` 분류)

본문 텍스트는 이 단계와 별개로 무손실 추출합니다(PDF 는 텍스트 레이어, HWPX 는
``section*.xml``). :mod:`ocrpipe.pipeline` 이 둘을 합칩니다.
"""

from __future__ import annotations

from .stage1_extract import ExtractedImage, extract_images
from .stage2_ocr import ocr_images

__all__ = ["ExtractedImage", "extract_images", "ocr_images"]
