"""2단계: 분리된 이미지 OCR (+ 결과로 text/dummy 분류).

1단계에서 나온 이미지를 각각 전처리(확대·대비·샤프닝) 후 PaddleOCR-VL 로 인식하고,
저신뢰 블록은 재인식(refine)합니다. 결과 bbox 는 이미지의 페이지 내 위치를 더해
**원본 페이지 좌표**로 되돌립니다.

분류는 **OCR 결과로 판정**합니다: 텍스트가 나오면 ``2_text/``, 안 나오면(로고·장식)
``2_dummy/`` 로 사본을 남깁니다. 예전에는 검출(detection)로 미리 걸렀지만, PaddleOCR-VL
단독 구성에서는 검출 전용 API 가 없어 그 방식이 **VLM 을 이미지당 두 번** 돌리게 됩니다.
한 번만 돌리고 결과로 나누는 편이 같은 정보를 절반 비용으로 얻습니다.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..ocr import OcrItem
from ..ocr.preprocess import imread_unicode, imwrite_unicode, preprocess_image
from ..ocr.refine import refine_items
from .stage1_extract import ExtractedImage


def ocr_images(
    engine: Any,
    images: list[ExtractedImage],
    page_number: int,
    order_base: int = 0,
    text_dir: Path | None = None,
    dummy_dir: Path | None = None,
) -> list[OcrItem]:
    """이미지들을 OCR 해 OcrItem 목록을 반환(페이지 좌표로 정규화).

    ``text_dir`` / ``dummy_dir`` 를 주면 텍스트 유무에 따라 사본을 남겨 눈으로
    확인할 수 있게 한다(중간 산출물 보존용, 인식 결과에는 영향 없음).
    """
    results: list[OcrItem] = []
    order = order_base
    for image in images:
        items = _ocr_one(engine, image, page_number)
        _sort_copy(image.path, text_dir if items else dummy_dir)
        for item in items:
            item.order = order
            order += 1
        results.extend(items)
    return results


def _sort_copy(source: Path, dest_dir: Path | None) -> None:
    """분류 결과 폴더로 사본 저장(실패해도 인식 결과에는 영향 없음)."""
    if dest_dir is None:
        return
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_dir / source.name)
    except Exception:
        pass


def _ocr_one(engine: Any, image: ExtractedImage, page_number: int) -> list[OcrItem]:
    left, top, _, _ = image.bbox

    src = imread_unicode(image.path)
    if src is None:
        return []
    # 작은 글씨 대비 최소 2배 확대 후 전처리 (스캔 전체 페이지는 그대로)
    min_scale = 1.0 if image.full_page else 2.0
    processed, scale = preprocess_image(src, min_scale=min_scale)

    tmp = image.path.with_name(f"{image.path.stem}.pre.png")
    if not imwrite_unicode(tmp, processed):
        return []
    try:
        items = engine.run_page(tmp, page_number)
    except Exception:
        tmp.unlink(missing_ok=True)
        return []
    refine_items(engine, tmp, page_number, items)
    tmp.unlink(missing_ok=True)

    # bbox 를 원본 페이지 좌표로 환산 (확대 배율 되돌리고 이미지 위치 오프셋 더하기)
    for item in items:
        if item.bbox:
            x1, y1, x2, y2 = item.bbox
            if image.full_page:
                item.bbox = [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]
            else:
                item.bbox = [
                    int(x1 / scale) + left, int(y1 / scale) + top,
                    int(x2 / scale) + left, int(y2 / scale) + top,
                ]
    return items
