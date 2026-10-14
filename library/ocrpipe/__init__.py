"""ocrpipe — 문서/이미지 OCR 파이프라인 (PaddleOCR-VL 단독).

라이브러리 사용::

    from ocrpipe import analyze
    blocks = analyze("사진.png")            # → 구조화 블록 리스트
    blocks = analyze("보고서.hwpx")          # HWPX 는 변환 없이 본문 + 이미지 처리

명령줄은 ``python -m ocrpipe.ocr_cli`` (또는 설치 후 ``ocrpipe``) 를 참고하세요.
"""

from __future__ import annotations

from .analyze import analyze, analyze_to_text

__version__ = "2.0.0"
__all__ = ["analyze", "analyze_to_text", "__version__"]
