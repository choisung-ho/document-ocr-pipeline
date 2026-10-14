"""OCR 파이프라인 오케스트레이션 (라이브러리·CLI 공용).

입력을 다음 원칙으로 처리합니다:

1. **본문 = 무손실 추출** (score 1.0, 인식 오차 없음)
   - PDF : 텍스트 레이어
   - HWPX: ``Contents/section*.xml`` 문단 (:mod:`ocrpipe.hwpx`)
2. **이미지 = 2단계 파이프라인** (:mod:`ocrpipe.stages`)
   - 1단계: 문서에서 이미지 분리 → ``1_images/``
     (PDF 는 페이지에서 크롭, HWPX 는 ``BinData/`` 에서 원본 그대로)
   - 2단계: 분리된 이미지 OCR → 결과로 ``2_text/`` / ``2_dummy/`` 분류
3. **후처리**: 노이즈 필터·중복 제거.

텍스트 레이어가 없는 스캔 PDF 페이지는 페이지 전체를 이미지 1장으로 보고 OCR 합니다.
단일 이미지 입력은 곧바로 전체 OCR 합니다.

세부 설정은 :mod:`ocrpipe.config` 의 ``settings`` 를 따릅니다.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from . import hwpx
from .config import settings
from .document_utils import IMAGE_SUFFIXES, render_pdf_page
from .ocr import OcrItem
from .ocr.base import dedupe_key, normalize_blocks
from .ocr.preprocess import imread_unicode, preprocess_page_image, rescale_bboxes
from .ocr.refine import refine_items
from .stages import ExtractedImage, extract_images, ocr_images


# --------------------------------------------------------------------------- #
# 공개 API
# --------------------------------------------------------------------------- #

def ocr_pdf_page(
    engine: Any,
    pdf_path: Path,
    page_number: int,
    work_dir: Path,
    dpi: int | None = None,
) -> tuple[list[OcrItem], Path | None]:
    """PDF 한 페이지를 처리한다 (본문 텍스트레이어 + 이미지 2단계).

    반환: (결과 블록, 렌더링된 페이지 이미지 경로 또는 None).
    텍스트 전용 페이지는 렌더링이 없으므로 이미지 경로가 None 이다.
    """
    dpi = dpi or settings.dpi
    zoom = dpi / 72

    if not settings.hybrid:
        page_path = render_pdf_page(pdf_path, page_number, work_dir, dpi=dpi)
        return _ocr_full_image(engine, page_path, page_number), page_path

    import fitz

    with fitz.open(pdf_path) as doc:
        if page_number < 1 or page_number > doc.page_count:
            return [], None
        page = doc[page_number - 1]
        raw = page.get_text("dict", sort=True)  # 읽기 순서(위→아래) 정렬
        native_blocks = _native_line_blocks(raw, zoom, zoom)
        total_chars = sum(len(block["text"]) for block in native_blocks)
        has_images = bool(page.get_images(full=True))

    is_scan = total_chars < settings.min_native_chars

    # 텍스트 레이어 있고 이미지 없음 → 렌더링·OCR 생략 (본문 무손실 추출만)
    if not is_scan and not has_images:
        return _drop_noise(normalize_blocks(native_blocks, page_number)), None

    # 이미지가 있거나 스캔 페이지 → 이 페이지만 렌더링해 2단계 파이프라인
    page_path = render_pdf_page(pdf_path, page_number, work_dir, dpi=dpi)
    stage_items = _run_stages(engine, pdf_path, page_number, page_path, work_dir)

    if is_scan:
        # 스캔 페이지: 본문 자체가 이미지 OCR 결과
        return _drop_noise(stage_items), page_path

    items = normalize_blocks(native_blocks, page_number)
    for offset, item in enumerate(stage_items):
        item.order = len(items) + offset
    items.extend(stage_items)
    return _drop_noise(_dedupe(items)), page_path


def ocr_page(engine: Any, image_path: Path, page_number: int = 1) -> list[OcrItem]:
    """이미지 파일(PNG/JPG 등) 한 장을 OCR 한다."""
    return _ocr_full_image(engine, image_path, page_number)


def ocr_hwpx(engine: Any, path: Path, work_dir: Path) -> list[OcrItem]:
    """HWPX 를 PDF 변환 없이 처리한다 (본문 문단 무손실 + BinData 이미지 OCR).

    본문은 ``section*.xml`` 에서 그대로 읽고, 삽입 이미지는 ``BinData/`` 에서 원본을
    꺼내 2단계 OCR 을 태운다. 흐름 문서라 페이지 좌표가 없어 본문 bbox 는 None 이고,
    이미지에서 인식한 블록의 bbox 는 **그 이미지 내부 좌표**다.
    """
    items = hwpx.text_items(path)

    def run(base: Path) -> list[OcrItem]:
        image_paths = hwpx.extract_images(path, base / "1_images")
        if not image_paths:
            return []
        return ocr_images(
            engine,
            [_as_extracted(p, index) for index, p in enumerate(image_paths)],
            page_number=1,
            order_base=len(items),
            text_dir=base / "2_text",
            dummy_dir=base / "2_dummy",
        )

    if settings.keep_stage_files:
        image_items = run(work_dir / "stages")
    else:
        with tempfile.TemporaryDirectory(prefix="ocr_stages_") as tmp:
            image_items = run(Path(tmp))

    items.extend(image_items)
    return _drop_noise(_dedupe(items))


def _as_extracted(path: Path, index: int) -> ExtractedImage:
    """HWPX 에서 꺼낸 이미지 파일을 2단계 입력 형식으로 감싼다(오프셋 없음)."""
    image = imread_unicode(path)
    height, width = image.shape[:2] if image is not None else (0, 0)
    return ExtractedImage(path=path, page=1, index=index, bbox=(0, 0, width, height))


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


# --------------------------------------------------------------------------- #
# 2단계 오케스트레이션 (PDF 페이지)
# --------------------------------------------------------------------------- #

def _run_stages(
    engine: Any,
    pdf_path: Path,
    page_number: int,
    page_path: Path,
    work_dir: Path,
) -> list[OcrItem]:
    """이미지 2단계(분리→OCR)를 실행하고 결과 블록을 반환.

    ``settings.keep_stage_files`` 가 켜져 있으면 단계별 폴더를 work_dir 아래에
    영구 보존하고, 아니면 임시 폴더를 써서 종료 후 정리한다.
    """
    if settings.keep_stage_files:
        base = work_dir / "stages"
        return _stages_in(engine, pdf_path, page_number, page_path, base)
    with tempfile.TemporaryDirectory(prefix="ocr_stages_") as tmp:
        return _stages_in(engine, pdf_path, page_number, page_path, Path(tmp))


def _stages_in(
    engine: Any,
    pdf_path: Path,
    page_number: int,
    page_path: Path,
    base: Path,
) -> list[OcrItem]:
    images, _ = extract_images(pdf_path, page_number, page_path, base / "1_images")
    if not images:
        return []
    return ocr_images(
        engine,
        images,
        page_number,
        text_dir=base / "2_text",
        dummy_dir=base / "2_dummy",
    )


def _ocr_full_image(engine: Any, image_path: Path, page_number: int) -> list[OcrItem]:
    """전체 이미지 OCR: 전처리 → 인식 → 저신뢰 재인식 → 좌표 원복 → 노이즈 필터."""
    processed_path, scale = preprocess_page_image(image_path)
    items = engine.run_page(processed_path, page_number)
    refine_items(engine, processed_path, page_number, items)
    rescale_bboxes(items, scale)
    return _drop_noise(items)


# --------------------------------------------------------------------------- #
# 텍스트 레이어 추출
# --------------------------------------------------------------------------- #

def _native_line_blocks(raw: dict[str, Any], zoom_x: float, zoom_y: float) -> list[dict[str, Any]]:
    """PDF 텍스트 레이어를 줄 단위 블록 dict 리스트로 변환 (렌더링 좌표계)."""
    blocks: list[dict[str, Any]] = []
    order = 0
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            if not text.strip():
                continue
            x1, y1, x2, y2 = line.get("bbox", (0, 0, 0, 0))
            blocks.append(
                {
                    "text": text,
                    "type": None,
                    "bbox": [x1 * zoom_x, y1 * zoom_y, x2 * zoom_x, y2 * zoom_y],
                    "order": order,
                    "score": 1.0,  # 텍스트 레이어 원문 → 최고 신뢰도
                }
            )
            order += 1
    return blocks


# --------------------------------------------------------------------------- #
# 후처리
# --------------------------------------------------------------------------- #

def _dedupe(items: list[OcrItem]) -> list[OcrItem]:
    """같은 (타입, 텍스트, 위치) 블록 중복 제거 (먼저 나온 쪽 유지)."""
    seen: set[tuple] = set()
    unique: list[OcrItem] = []
    for item in items:
        key = dedupe_key(item.type, item.text.strip(), item.bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


#: 한국어 문서에서 의미 있는 문자: 한글·라틴·숫자·한자
_MEANINGFUL_RE = re.compile(r"[0-9A-Za-z가-힣ㄱ-ㆎ一-鿿]")


def _drop_noise(items: list[OcrItem]) -> list[OcrItem]:
    """그래프 도형·기호를 글자로 오인한 쓰레기 블록 제거."""
    if not settings.noise_filter:
        return items
    kept: list[OcrItem] = []
    for item in items:
        if _MEANINGFUL_RE.search(item.text.strip()):
            kept.append(item)
        elif item.score is not None and item.score >= 0.95:
            kept.append(item)  # 텍스트 레이어 원문(1.0) 등 확실한 것만 유지
    return kept
