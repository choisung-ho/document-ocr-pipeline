from __future__ import annotations

import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from xml.etree import ElementTree as ET

import fitz
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


@dataclass
class DocumentItem:
    page: int | None
    type: str
    text: str
    markdown: str | None = None
    section_path: list[str] | None = None
    source_text: str | None = None
    bbox: list[float] | None = None
    asset_url: str | None = None
    description: str | None = None
    metrics: list[dict[str, Any]] | None = None
    chart: dict[str, Any] | None = None


@dataclass
class PdfPageAnalysis:
    mode: str
    native_text: str
    text_char_count: int
    text_quality: float
    image_coverage: float
    ocr_regions: list[list[float]]


def count_processing_units(path: Path) -> int:
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as document:
            return max(document.page_count, 1)
    return 1


def extract_docx(path: Path, asset_dir: Path, asset_url_prefix: str) -> list[DocumentItem]:
    document = Document(path)
    items: list[DocumentItem] = []
    section_path: list[str] = []

    for block in iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if not text:
                continue
            level = heading_level(block.style.name if block.style else "")
            item_type = "heading" if level else "paragraph"
            if level:
                section_path = section_path[: level - 1]
                section_path.append(text)
            markdown = f"{'#' * level} {text}" if level else text
            items.append(
                DocumentItem(
                    page=None,
                    type=item_type,
                    text=text,
                    markdown=markdown,
                    section_path=list(section_path),
                    source_text=text,
                    metrics=extract_metrics(text),
                )
            )
        elif isinstance(block, Table):
            rows = [[clean_text(cell.text) for cell in row.cells] for row in block.rows]
            markdown = table_to_markdown(rows)
            if markdown:
                table_text = "\n".join(" | ".join(row) for row in rows)
                items.append(
                    DocumentItem(
                        page=None,
                        type="table",
                        text=table_text,
                        markdown=markdown,
                        section_path=list(section_path),
                        source_text="\n".join("\t".join(row) for row in rows),
                        metrics=extract_metrics(table_text),
                    )
                )

    items.extend(extract_archive_assets(path, "word/media/", asset_dir, asset_url_prefix, section_path))
    return items


def extract_hwpx(path: Path, asset_dir: Path, asset_url_prefix: str) -> list[DocumentItem]:
    items: list[DocumentItem] = []
    with zipfile.ZipFile(path) as archive:
        section_names = sorted(
            (name for name in archive.namelist() if re.fullmatch(r"Contents/section\d+\.xml", name)),
            key=natural_key,
        )
        for section_name in section_names:
            root = ET.fromstring(archive.read(section_name))
            walk_hwpx(root, items)

    items.extend(extract_archive_assets(path, "BinData/", asset_dir, asset_url_prefix, []))
    return deduplicate_items(items)


def render_pdf_page(document: fitz.Document, page_number: int, destination: Path, scale: float = 2.0) -> None:
    page = document[page_number - 1]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    pixmap.save(destination)


def render_pdf_region(page: fitz.Page, bbox: list[float], destination: Path) -> None:
    clip = fitz.Rect(*bbox) & page.rect
    if clip.is_empty or clip.width < 2 or clip.height < 2:
        raise ValueError("OCR 영역의 크기가 너무 작습니다.")
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
    pixmap.save(destination)


def analyze_pdf_page(page: fitz.Page) -> PdfPageAnalysis:
    native_text = clean_text(page.get_text("text"))
    compact = re.sub(r"\s+", "", native_text)
    meaningful = re.findall(r"[0-9A-Za-z가-힣]", compact)
    quality = len(meaningful) / max(len(compact), 1)
    min_chars = 40
    min_quality = 0.6

    # 빠른 사전 확인: 페이지 자원에 래스터 이미지가 없으면 무거운 영역 분석을 생략한다.
    # get_images는 dict 파싱보다 훨씬 저렴하며, dict의 이미지 블록은 항상 이 목록의 부분집합이다.
    if page.get_images(full=True):
        image_regions = extract_pdf_image_regions(page)
    else:
        image_regions = []
    # 표 영역 OCR은 기본 비활성화 — 표 텍스트는 원문 추출로 이미 확보되고,
    # 표가 있는 모든 페이지가 OCR 대상이 되어 전체 처리 시간이 크게 늘어난다.
    ocr_target_regions = list(image_regions)
    if os.getenv("OCR_INCLUDE_TABLE_REGIONS", "0") == "1":
        ocr_target_regions.extend(extract_pdf_table_regions(page))
    regions = merge_overlapping_regions(ocr_target_regions, page.rect)
    page_area = max(page.rect.width * page.rect.height, 1)
    image_coverage = min(
        sum(fitz.Rect(*bbox).get_area() for bbox in image_regions) / page_area,
        1.0,
    )

    has_usable_native_text = len(compact) >= min_chars and quality >= min_quality
    has_images = bool(image_regions)
    if has_usable_native_text:
        mode = "native+region-ocr" if regions else "native"
    else:
        # 이미지가 없는 저텍스트 페이지(표지·간지·빈 페이지)는 전체 OCR을 생략한다.
        mode = "ocr" if has_images else "native"
    return PdfPageAnalysis(
        mode=mode,
        native_text=native_text,
        text_char_count=len(compact),
        text_quality=round(quality, 3),
        image_coverage=round(image_coverage, 3),
        ocr_regions=regions if has_usable_native_text else [],
    )


def extract_pdf_native_page(page: fitz.Page, page_number: int) -> list[DocumentItem]:
    items: list[DocumentItem] = []
    section_path: list[str] = []
    for block in page.get_text("blocks", sort=True):
        if len(block) < 7 or int(block[6]) != 0:
            continue
        text = clean_text(str(block[4]))
        if not text:
            continue
        item_type = "heading" if looks_like_heading(text) else "paragraph"
        if item_type == "heading":
            section_path = [text]
        bbox = [round(float(value) * 2, 2) for value in block[:4]]
        items.append(
            DocumentItem(
                page=page_number,
                type=item_type,
                text=text,
                markdown=f"## {text}" if item_type == "heading" else text,
                section_path=list(section_path),
                source_text=text,
                bbox=bbox,
                metrics=extract_metrics(text),
            )
        )
    return items


def extract_pdf_image_regions(page: fitz.Page) -> list[list[float]]:
    regions: list[list[float]] = []
    try:
        page_dict = page.get_text("dict")
    except (RuntimeError, ValueError):
        return regions
    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if is_valid_bbox(bbox, page.rect):
            regions.append([float(value) for value in bbox])
    return regions


def extract_pdf_table_regions(page: fitz.Page) -> list[list[float]]:
    regions: list[list[float]] = []
    try:
        tables = page.find_tables()
    except (AttributeError, RuntimeError, ValueError):
        return regions
    for table in getattr(tables, "tables", []):
        bbox = getattr(table, "bbox", None)
        if is_valid_bbox(bbox, page.rect):
            regions.append([float(value) for value in bbox])
    return regions


def merge_overlapping_regions(regions: list[list[float]], page_rect: fitz.Rect) -> list[list[float]]:
    minimum_area = max(page_rect.get_area() * 0.002, 400)
    pending = [
        fitz.Rect(*bbox) & page_rect
        for bbox in regions
        if is_valid_bbox(bbox, page_rect) and (fitz.Rect(*bbox) & page_rect).get_area() >= minimum_area
    ]
    merged: list[fitz.Rect] = []
    for region in sorted(pending, key=lambda rect: (rect.y0, rect.x0)):
        for index, current in enumerate(merged):
            intersection = region & current
            if intersection.get_area() > 0:
                merged[index] = region | current
                break
        else:
            merged.append(region)
    return [[round(rect.x0, 2), round(rect.y0, 2), round(rect.x1, 2), round(rect.y1, 2)] for rect in merged]


def is_valid_bbox(value: Any, page_rect: fitz.Rect) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    try:
        rect = fitz.Rect(*(float(item) for item in value[:4])) & page_rect
    except (TypeError, ValueError):
        return False
    return not rect.is_empty and rect.width > 1 and rect.height > 1


def iter_docx_blocks(document: Document) -> Iterator[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        local_name = child.tag.rsplit("}", 1)[-1]
        if local_name == "p":
            yield Paragraph(child, document)
        elif local_name == "tbl":
            yield Table(child, document)


def walk_hwpx(element: ET.Element, items: list[DocumentItem]) -> None:
    local_name = element.tag.rsplit("}", 1)[-1]
    if local_name == "tbl":
        rows: list[list[str]] = []
        for row in (node for node in element.iter() if node.tag.rsplit("}", 1)[-1] == "tr"):
            cells: list[str] = []
            for cell in (node for node in list(row) if node.tag.rsplit("}", 1)[-1] in {"tc", "cell"}):
                text = clean_text("".join(node.text or "" for node in cell.iter() if node.tag.rsplit("}", 1)[-1] == "t"))
                cells.append(text)
            if cells:
                rows.append(cells)
        markdown = table_to_markdown(rows)
        if markdown:
            table_text = "\n".join(" | ".join(row) for row in rows)
            items.append(
                DocumentItem(
                    page=None,
                    type="table",
                    text=table_text,
                    markdown=markdown,
                    section_path=[],
                    source_text="\n".join("\t".join(row) for row in rows),
                    metrics=extract_metrics(table_text),
                )
            )
        return
    if local_name == "p":
        text_parts: list[str] = []
        collect_hwpx_text_without_tables(element, text_parts)
        source_text = "".join(text_parts).strip()
        text = clean_text(source_text)
        if text:
            item_type = "heading" if looks_like_heading(text) else "paragraph"
            items.append(
                DocumentItem(
                    page=None,
                    type=item_type,
                    text=text,
                    markdown=f"## {text}" if item_type == "heading" else source_text,
                    section_path=[text] if item_type == "heading" else [],
                    source_text=source_text,
                    metrics=extract_metrics(text),
                )
            )
        for child in element:
            if any(node.tag.rsplit("}", 1)[-1] == "tbl" for node in child.iter()):
                walk_hwpx(child, items)
        return
    for child in element:
        walk_hwpx(child, items)


def collect_hwpx_text_without_tables(element: ET.Element, output: list[str]) -> None:
    if element.tag.rsplit("}", 1)[-1] == "tbl":
        return
    if element.tag.rsplit("}", 1)[-1] == "t" and element.text:
        output.append(element.text)
    for child in element:
        collect_hwpx_text_without_tables(child, output)


def extract_archive_assets(
    path: Path,
    prefix: str,
    asset_dir: Path,
    asset_url_prefix: str,
    section_path: list[str],
) -> list[DocumentItem]:
    items: list[DocumentItem] = []
    with zipfile.ZipFile(path) as archive:
        for index, name in enumerate(sorted(archive.namelist(), key=natural_key), start=1):
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            original_name = Path(name).name
            extension = safe_extension(Path(original_name).suffix.lstrip(".") or "bin")
            filename = f"asset_{index:03d}_{Path(original_name).stem}.{extension}"
            with archive.open(name) as source, (asset_dir / filename).open("wb") as destination:
                shutil.copyfileobj(source, destination)
            items.append(
                DocumentItem(
                    page=None,
                    type="image",
                    text=original_name,
                    markdown=f"![{original_name}]({asset_url_prefix}/{filename})",
                    section_path=list(section_path),
                    asset_url=f"{asset_url_prefix}/{filename}",
                )
            )
    return items


def heading_level(style_name: str) -> int:
    match = re.search(r"(?:heading|제목)\s*(\d+)", style_name, re.IGNORECASE)
    return min(max(int(match.group(1)), 1), 6) if match else 0


def table_to_markdown(rows: list[list[str]]) -> str:
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    escaped = [[cell.replace("|", "\\|").replace("\n", "<br>") for cell in row] for row in normalized]
    header = escaped[0]
    body = escaped[1:]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def deduplicate_items(items: list[DocumentItem]) -> list[DocumentItem]:
    output: list[DocumentItem] = []
    seen: set[tuple[int | None, str, str, str | None]] = set()
    for item in items:
        key = (item.page, item.type, item.text, item.asset_url)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def looks_like_heading(text: str) -> bool:
    if len(text) > 100:
        return False
    return bool(re.match(r"^(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]|[IVX]+\.|제\s*\d+|\d+(?:\.\d+)*[.)]?|[가-힣][.)])\s*", text))


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", "").split())


# OCR 결과의 전각 숫자·기호를 반각으로 정규화해 수치 추출 누락을 방지한다.
FULLWIDTH_NUMERIC_MAP = str.maketrans("０１２３４５６７８９．，％", "0123456789.,%")
# 긴 단위를 앞에 둔다 — 대안 순서 때문에 "6개월"이 "6개"로 잘려 추출되는 것을 막는다.
METRIC_UNITS = r"퍼센트|개월|시간|%|점|명|원|건|개|회|억|만|천|조|GB|TB|MB|년|주|일|위|배|p"
METRIC_PATTERN = re.compile(rf"(\d{{1,3}}(?:,\d{{3}})+|\d+(?:\.\d+)?)\s*({METRIC_UNITS})?", re.IGNORECASE)


def extract_metrics(text: str) -> list[dict[str, Any]] | None:
    normalized = text.translate(FULLWIDTH_NUMERIC_MAP)
    metrics: list[dict[str, Any]] = []
    for match in METRIC_PATTERN.finditer(normalized):
        number, unit = match.group(1), match.group(2)
        # 단위·천단위 구분·소수점이 없는 순수 정수(연도, 학번 등)는 잡음으로 제외한다.
        if not unit and "," not in number and "." not in number:
            continue
        try:
            value = float(number.replace(",", ""))
        except ValueError:
            continue
        metrics.append({"raw": match.group(0).strip(), "value": value, "unit": unit or ""})
    return metrics or None


def parse_chart_table(raw_text: str) -> dict[str, Any] | None:
    rows: list[list[str]] = []
    for line in raw_text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", cell or "---") for cell in cells):
            continue
        if any(cells):
            rows.append(cells)
    if len(rows) < 2 or not rows[0]:
        return None
    return {
        "x_axis": rows[0][0],
        "series": rows[0][1:],
        "rows": rows[1:],
    }


def describe_chart(chart: dict[str, Any] | None, fallback_text: str) -> str:
    if not chart:
        return f"차트 인식 내용: {fallback_text[:200]}"
    series = ", ".join(name for name in chart["series"] if name) or "값"
    return f"차트 데이터 {len(chart['rows'])}행 · X축(항목) '{chart['x_axis']}' · Y축(계열) '{series}'"


def natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def safe_extension(extension: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", extension.lower())
    return cleaned or "bin"
