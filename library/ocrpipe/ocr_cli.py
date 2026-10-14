"""단일 문서 OCR CLI (PaddleOCR-VL).

예시:
    python -m ocrpipe.ocr_cli "문서/시스템구성도.png"
    python -m ocrpipe.ocr_cli 보고서.hwpx
    python -m ocrpipe.ocr_cli input.pdf --pages 1,3-4 -o out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import hwpx, pipeline
from .config import settings
from .document_utils import IMAGE_SUFFIXES, count_pages, parse_page_filter
from .ocr import OcrItem, get_engine


def run(path: Path, page_filter: str, dpi: int) -> list[OcrItem]:
    engine = get_engine()
    available, message = engine.availability()
    if not available:
        raise SystemExit(
            f"PaddleOCR-VL 사용 불가: {message}\n"
            '설치: pip install "paddleocr[doc-parser]" (+ paddlepaddle 또는 paddlepaddle-gpu)'
        )

    suffix = path.suffix.lower()

    # HWPX: 변환 없이 본문 + BinData 이미지
    if suffix in {".hwpx", ".hwp"}:
        reason = hwpx.unsupported_reason(path)
        if reason:
            raise SystemExit(reason)
        print(f"[hwpx] 본문 + 삽입 이미지 처리 중...", file=sys.stderr)
        started = time.perf_counter()
        items = pipeline.ocr_hwpx(engine, path, path.parent / f"{path.stem}_pages")
        print(f"  완료: {time.perf_counter() - started:.2f}s, 블록 {len(items)}개", file=sys.stderr)
        return items

    if suffix in IMAGE_SUFFIXES:
        print("[image] 처리 중...", file=sys.stderr)
        started = time.perf_counter()
        items = pipeline.ocr_page(engine, path, 1)
        print(f"  완료: {time.perf_counter() - started:.2f}s, 블록 {len(items)}개", file=sys.stderr)
        return items

    if suffix != ".pdf":
        raise SystemExit(
            f"지원하지 않는 형식: {suffix}\n"
            "지원: .pdf / .hwpx / 이미지(png·jpg·bmp·tif·webp)"
        )

    selected = parse_page_filter(page_filter)
    total = count_pages(path)
    render_dir = path.parent / f"{path.stem}_pages"
    items: list[OcrItem] = []
    for page_number in range(1, total + 1):
        if selected and page_number not in selected:
            continue
        print(f"[page {page_number}/{total}] 처리 중...", file=sys.stderr)
        started = time.perf_counter()
        # 필요한 페이지만 렌더링·OCR (텍스트 전용 페이지는 무손실 추출만)
        page_items, page_path = pipeline.ocr_pdf_page(engine, path, page_number, render_dir, dpi=dpi)
        items.extend(page_items)
        mode = "무손실 추출(OCR 생략)" if page_path is None else "OCR"
        print(f"  {mode} 완료: {time.perf_counter() - started:.2f}s, 블록 {len(page_items)}개", file=sys.stderr)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 문서 OCR (PDF/HWPX/이미지, PaddleOCR-VL)")
    parser.add_argument("input", help="입력 파일 경로 (.pdf / .hwpx / 이미지)")
    parser.add_argument("--pages", "-p", default="", help="PDF 페이지 필터, 예: 1,3-4")
    parser.add_argument("--dpi", type=int, default=settings.dpi, help="PDF 렌더링 DPI (기본 300)")
    parser.add_argument("--output", "-o", help="결과 JSON 저장 경로 (미지정 시 OCR_result/ 폴더)")
    args = parser.parse_args()

    path = Path(args.input).expanduser()
    if not path.exists():
        raise SystemExit(f"파일을 찾을 수 없습니다: {path}")

    items = run(path, args.pages, args.dpi)
    text = json.dumps([asdict(item) for item in items], ensure_ascii=False, indent=2)

    if args.output:
        output_path = Path(args.output)
    else:
        # 최종 JSON 은 OCR_result/ 폴더에 <원본이름>_paddleocr-vl.json 으로 저장
        output_path = settings.result_dir / f"{path.stem}_paddleocr-vl.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"\n결과 저장: {output_path} (블록 {len(items)}개)", file=sys.stderr)


if __name__ == "__main__":
    main()
