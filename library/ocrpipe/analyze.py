"""라이브러리 공개 API — 이미지/PDF/HWPX 한 개를 OCR 해 구조화 블록으로 반환.

다른 프로젝트에서 다음처럼 씁니다::

    from ocrpipe import analyze

    blocks = analyze("사진.png")            # 이미지 → 구조화 블록 리스트
    blocks = analyze("문서.pdf")            # PDF (본문 무손실 + 이미지 2단계)
    blocks = analyze("보고서.hwpx")          # HWPX (변환 없이 본문 + BinData 이미지)
    blocks = analyze(image_bytes)           # 바이트도 가능 (확장자 지정 권장)

엔진은 **PaddleOCR-VL 단독**입니다(모델 선택 인자 없음).

반환값은 :class:`ocrpipe.ocr.OcrItem` 을 dict 로 편 리스트입니다::

    [{"page": 1, "type": "paragraph", "text": "...", "bbox": [x1,y1,x2,y2],
      "order": 0, "html": None, "markdown": None, "score": 0.98, "crop": None}, ...]
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import hwpx, pipeline
from .document_utils import IMAGE_SUFFIXES, count_pages, parse_page_filter
from .ocr import OcrItem, get_engine


def analyze(
    source: str | Path | bytes,
    *,
    pages: str = "",
    suffix: str = ".png",
    as_dict: bool = True,
) -> list[dict[str, Any]] | list[OcrItem]:
    """이미지/PDF/HWPX 한 개를 OCR 해 구조화 블록 리스트를 반환한다.

    Args:
        source: 파일 경로(str/Path) 또는 이미지 바이트(bytes).
        pages:  PDF 페이지 필터, 예 ``"1,3-5"`` (미지정 시 전체). HWPX 는 무시.
        suffix: ``source`` 가 bytes 일 때 사용할 확장자(기본 ``".png"``).
        as_dict: True 면 dict 리스트, False 면 :class:`OcrItem` 리스트 반환.

    Raises:
        RuntimeError: PaddleOCR-VL 이 설치되어 있지 않을 때.
        FileNotFoundError: 경로가 존재하지 않을 때.
        ValueError: 지원하지 않는 형식(구형 .hwp 등)일 때.
    """
    engine = get_engine()
    available, message = engine.availability()
    if not available:
        raise RuntimeError(
            f"PaddleOCR-VL 사용 불가: {message}\n"
            '설치: pip install "paddleocr[doc-parser]" (+ paddlepaddle 또는 paddlepaddle-gpu)'
        )

    with tempfile.TemporaryDirectory(prefix="ocrpipe_") as tmp:
        work = Path(tmp)
        path = _materialize(source, work, suffix)
        items = _run(engine, path, work, pages)

    return [asdict(item) for item in items] if as_dict else items


def analyze_to_text(source: str | Path | bytes, *, pages: str = "") -> str:
    """:func:`analyze` 결과를 읽기 순서대로 이어붙인 순수 텍스트로 반환(편의 함수)."""
    items: list[OcrItem] = analyze(source, pages=pages, as_dict=False)  # type: ignore[assignment]
    ordered = sorted(items, key=lambda it: (it.page, it.order if it.order is not None else 0))
    return "\n".join(it.text for it in ordered)


# --------------------------------------------------------------------------- #
# 내부 헬퍼
# --------------------------------------------------------------------------- #

def _materialize(source: str | Path | bytes, work: Path, suffix: str) -> Path:
    """bytes 는 임시 파일로 쓰고, 경로는 존재 확인 후 그대로 반환."""
    if isinstance(source, (bytes, bytearray)):
        target = work / f"input{suffix if suffix.startswith('.') else '.' + suffix}"
        target.write_bytes(bytes(source))
        return target
    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    return path


def _run(engine: Any, path: Path, work: Path, pages: str) -> list[OcrItem]:
    suffix = path.suffix.lower()

    # HWPX: PDF 변환 없이 본문(section*.xml) + 삽입 이미지(BinData) 직접 처리
    if suffix in {".hwpx", ".hwp"}:
        reason = hwpx.unsupported_reason(path)
        if reason:
            raise ValueError(reason)
        return pipeline.ocr_hwpx(engine, path, work)

    # 이미지 한 장
    if suffix in IMAGE_SUFFIXES:
        return pipeline.ocr_page(engine, path, 1)

    if suffix != ".pdf":
        raise ValueError(
            f"지원하지 않는 형식: {suffix}\n"
            "지원: .pdf / .hwpx / 이미지(png·jpg·bmp·tif·webp)"
        )

    # PDF: 필요한 페이지만 처리 (텍스트 전용 페이지는 무손실 추출)
    selected = parse_page_filter(pages)
    render_dir = work / "pages"
    items: list[OcrItem] = []
    for page_number in range(1, count_pages(path) + 1):
        if selected and page_number not in selected:
            continue
        page_items, _ = pipeline.ocr_pdf_page(engine, path, page_number, render_dir)
        items.extend(page_items)
    return items
