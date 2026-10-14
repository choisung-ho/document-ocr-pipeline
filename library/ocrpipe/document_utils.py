"""PDF/이미지 입력 처리 공용 유틸 (웹앱·CLI 공용).

웹 프레임워크에 의존하지 않으므로 CLI 에서도 그대로 가져다 씁니다.
"""

from __future__ import annotations

from pathlib import Path

from .config import settings

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES


def count_pages(path: Path) -> int:
    """이미지는 1, PDF 는 실제 페이지 수."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    import fitz

    with fitz.open(path) as doc:
        return doc.page_count


def render_pdf_page(pdf_path: Path, page_number: int, output_dir: Path, dpi: int | None = None) -> Path:
    """PDF 의 특정 페이지 한 장만 PNG 로 렌더링한다 (필요한 페이지만 렌더링용)."""
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        dpi = settings.dpi
    zoom = dpi / 72
    page_path = output_dir / f"page_{page_number:04d}.png"
    if page_path.exists():
        return page_path
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pixmap.save(page_path)
    return page_path


def render_pdf_pages(pdf_path: Path, output_dir: Path, dpi: int | None = None) -> list[Path]:
    """PDF 각 페이지를 PNG 로 렌더링하고 경로 리스트를 반환."""
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        dpi = settings.dpi
    zoom = dpi / 72
    page_paths: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            page_path = output_dir / f"page_{index:04d}.png"
            pixmap.save(page_path)
            page_paths.append(page_path)
    return page_paths


def parse_page_filter(raw_value: str) -> set[int]:
    """'2', '2,4-5' 같은 문자열을 페이지 번호 집합으로 변환."""
    pages: set[int] = set()
    for chunk in raw_value.replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            start_raw, end_raw = chunk.split("-", 1)
            if start_raw.isdigit() and end_raw.isdigit():
                start, end = int(start_raw), int(end_raw)
                pages.update(range(min(start, end), max(start, end) + 1))
            continue
        if chunk.isdigit():
            pages.add(int(chunk))
    return pages
