"""OCR 엔진 — PaddleOCR-VL 단독.

여러 엔진을 비교하던 레지스트리는 제거했습니다. 이 프로젝트는 **PaddleOCR-VL** 하나만
사용합니다(레이아웃 검출 + VLM 문서 파싱이 표·수식까지 구조화해 주기 때문).

파이프라인·CLI 는 :func:`get_engine` 으로 엔진 인스턴스를 얻습니다(무거운 모델은
첫 호출 때 한 번만 로드).
"""

from __future__ import annotations

from .base import OcrEngine, OcrItem
from .paddle_vl import PaddleVLEngine

_ENGINE: PaddleVLEngine | None = None


def get_engine() -> PaddleVLEngine:
    """PaddleOCR-VL 엔진 인스턴스(프로세스 내 단일 객체)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PaddleVLEngine()
    return _ENGINE


def engine_available() -> tuple[bool, str]:
    """(설치됨?, 안내 메시지) — paddle/paddleocr 설치 여부."""
    return get_engine().availability()


__all__ = ["OcrEngine", "OcrItem", "PaddleVLEngine", "get_engine", "engine_available"]
