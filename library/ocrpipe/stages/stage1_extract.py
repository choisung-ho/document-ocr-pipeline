"""1단계: 문서 스캔 → 이미지 분리.

PDF 페이지에 삽입된 래스터 이미지(사진·삽화)를 찾아, 렌더링된 페이지에서
해당 영역을 잘라 ``1_images/`` 폴더에 저장합니다. 각 이미지의 페이지 내
위치(bbox, 렌더링 픽셀 좌표)를 함께 반환해 이후 단계가 원래 위치로 되돌릴 수
있게 합니다.

텍스트 레이어가 없는 스캔 페이지는 "페이지 전체가 하나의 이미지"로 취급합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import settings
from ..ocr.preprocess import imread_unicode, imwrite_unicode


@dataclass
class ExtractedImage:
    """1단계에서 분리된 이미지 한 장."""

    path: Path                      # 저장된 크롭 이미지 경로
    page: int                       # 원본 페이지 번호(1-base)
    index: int                      # 페이지 내 이미지 순번(0-base)
    bbox: tuple[int, int, int, int]  # 페이지 내 위치(렌더링 픽셀 좌표)
    full_page: bool = False         # 페이지 전체(스캔 페이지)인지 여부


def extract_images(
    pdf_path: Path,
    page_number: int,
    rendered_page: Path,
    out_dir: Path,
) -> tuple[list[ExtractedImage], bool]:
    """페이지에서 이미지를 분리해 저장한다.

    반환: (분리된 이미지 목록, 스캔페이지 여부).
    스캔 페이지(텍스트 레이어 없음)면 페이지 전체를 이미지 1장으로 반환한다.
    """
    import fitz

    rendered = imread_unicode(rendered_page)
    if rendered is None:
        return [], False
    img_height, img_width = rendered.shape[:2]

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        zoom_x = img_width / max(page.rect.width, 1e-6)
        zoom_y = img_height / max(page.rect.height, 1e-6)
        native_chars = len(page.get_text("text").strip())
        rects = _image_rects(page, zoom_x, zoom_y, img_width, img_height, native_chars)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 스캔 페이지: 텍스트 레이어가 없으면 페이지 전체를 이미지로 취급
    if native_chars < settings.min_native_chars:
        target = out_dir / f"page_{page_number:04d}_full.png"
        imwrite_unicode(target, rendered)
        return (
            [ExtractedImage(target, page_number, 0, (0, 0, img_width, img_height), full_page=True)],
            True,
        )

    extracted: list[ExtractedImage] = []
    for index, (left, top, right, bottom) in enumerate(rects):
        crop = rendered[top:bottom, left:right]
        if crop.size == 0:
            continue
        target = out_dir / f"page_{page_number:04d}_img_{index:02d}.png"
        if not imwrite_unicode(target, crop):
            continue
        extracted.append(
            ExtractedImage(target, page_number, index, (left, top, right, bottom))
        )
    return extracted, False


def _image_rects(
    page,
    zoom_x: float,
    zoom_y: float,
    img_width: int,
    img_height: int,
    native_chars: int,
) -> list[tuple[int, int, int, int]]:
    """페이지 내 래스터 이미지 영역(렌더링 픽셀 좌표)을 수집하고 겹치면 병합."""
    min_side = settings.min_region_side
    rects: list[tuple[int, int, int, int]] = []
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            placements = page.get_image_rects(xref)
        except Exception:
            continue
        for rect in placements:
            left = max(0, int(rect.x0 * zoom_x))
            top = max(0, int(rect.y0 * zoom_y))
            right = min(img_width, int(rect.x1 * zoom_x))
            bottom = min(img_height, int(rect.y1 * zoom_y))
            if right - left < min_side or bottom - top < min_side:
                continue
            # 텍스트가 충분한 페이지의 전면 배경 이미지는 제외
            if native_chars >= 200 and (right - left) * (bottom - top) > img_width * img_height * 0.7:
                continue
            rects.append((left, top, right, bottom))
    return _merge_rects(rects)


def _merge_rects(rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    merged = list(rects)
    changed = True
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        for rect in merged:
            for index, other in enumerate(result):
                if not (rect[2] <= other[0] or other[2] <= rect[0] or rect[3] <= other[1] or other[3] <= rect[1]):
                    result[index] = (
                        min(rect[0], other[0]), min(rect[1], other[1]),
                        max(rect[2], other[2]), max(rect[3], other[3]),
                    )
                    changed = True
                    break
            else:
                result.append(rect)
        merged = result
    return merged
