"""저신뢰 블록 2차 재인식(refine) 패스 (모든 엔진 공용).

1차 OCR 결과 중 신뢰도(score)가 낮은 블록의 bbox 영역을 잘라 확대한 뒤
같은 엔진으로 다시 인식하고, 신뢰도가 더 높게 나오면 텍스트만 교체합니다.
그래프 축 라벨·범례·표 안의 작은 숫자처럼 원본 해상도에서 뭉개지는
텍스트의 정확도를 끌어올리는 것이 목적입니다.

JSON 결과 형식(OcrItem 필드)은 변경하지 않습니다 — text/score 값만 갱신됩니다.

환경변수:
- ``OCR_REFINE``           : on/off (기본 1)
- ``OCR_REFINE_THRESHOLD`` : 재인식 대상 신뢰도 상한 (기본 0.80)
- ``OCR_REFINE_MAX``       : 페이지당 최대 재인식 블록 수 (기본 24)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import cv2

from ..config import settings
from .base import OcrItem
from .preprocess import imread_unicode, imwrite_unicode

#: 재인식 시 블록 높이를 이 픽셀 크기까지 확대
_TARGET_HEIGHT = 110
_MAX_SCALE = 4.0
_MIN_SCALE = 1.5


def _candidates(items: list[OcrItem], image_width: int, image_height: int) -> list[OcrItem]:
    threshold = settings.refine_threshold
    picked: list[OcrItem] = []
    page_area = image_width * image_height
    for item in items:
        if item.score is None or item.score >= threshold:
            continue
        if not item.bbox or len(item.bbox) != 4:
            continue
        x1, y1, x2, y2 = item.bbox
        width, height = abs(x2 - x1), abs(y2 - y1)
        if width < 4 or height < 4:
            continue
        # 표·본문 덩어리 전체 재인식은 비효율 → 작은 블록만 대상
        if height > 220 or (width * height) > page_area * 0.3:
            continue
        picked.append(item)
    picked.sort(key=lambda it: it.score or 0.0)
    return picked[: settings.refine_max]


def _reocr_text(engine: Any, crop_path: Path, page_number: int) -> tuple[str, float] | None:
    """크롭 이미지를 재인식해 (텍스트, 평균 신뢰도)를 반환. 실패 시 None."""
    try:
        results = engine.run_page(crop_path, page_number)
    except Exception:
        return None
    texts: list[tuple[list[int] | None, str, float | None]] = []
    for result in results:
        if result.text and result.text.strip():
            texts.append((result.bbox, result.text.strip(), result.score))
    if not texts:
        return None
    # 읽기 순서(위→아래, 왼→오른쪽)로 정렬 후 이어붙임
    texts.sort(key=lambda entry: (entry[0][1], entry[0][0]) if entry[0] else (0, 0))
    joined = " ".join(entry[1] for entry in texts)
    scores = [entry[2] for entry in texts if entry[2] is not None]
    if not scores:
        return None
    return joined, sum(scores) / len(scores)


def _crop_variants(crop: "Any"):
    """재인식용 이미지 변형을 생성한다.

    - plain    : 확대 원본
    - binary   : OTSU 이진화 — 연한 색 배경 위 텍스트(그래프 축 라벨 등)에 유리
    - inverted : 반전 — 어두운 배경 위 밝은 텍스트(색 막대 안 수치 등)에 유리
    """
    yield crop
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if float(gray.mean()) < 128:
        yield cv2.cvtColor(255 - gray, cv2.COLOR_GRAY2BGR)


def refine_items(engine: Any, image_path: Path, page_number: int, items: list[OcrItem]) -> int:
    """저신뢰 블록을 재인식해 제자리 갱신. 교체된 블록 수를 반환."""
    if not settings.refine or not items:
        return 0
    image = imread_unicode(image_path)
    if image is None:
        return 0
    height, width = image.shape[:2]
    targets = _candidates(items, width, height)
    if not targets:
        return 0

    replaced = 0
    with tempfile.TemporaryDirectory(prefix="ocr_refine_") as tmp:
        tmp_dir = Path(tmp)
        for idx, item in enumerate(targets):
            x1, y1, x2, y2 = item.bbox  # type: ignore[misc]
            left, right = sorted((int(x1), int(x2)))
            top, bottom = sorted((int(y1), int(y2)))
            pad = max(6, int((bottom - top) * 0.3))
            left = max(0, left - pad)
            top = max(0, top - pad)
            right = min(width, right + pad)
            bottom = min(height, bottom + pad)
            if right - left < 8 or bottom - top < 8:
                continue

            crop = image[top:bottom, left:right]
            block_height = bottom - top
            scale = max(_MIN_SCALE, min(_MAX_SCALE, _TARGET_HEIGHT / max(block_height, 1)))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

            # 여러 변형(원본·이진화·반전)으로 재인식해 가장 신뢰도 높은 결과 채택
            best: tuple[str, float] | None = None
            for variant_index, variant in enumerate(_crop_variants(crop)):
                crop_path = tmp_dir / f"refine_{idx:03d}_{variant_index}.png"
                if not imwrite_unicode(crop_path, variant):
                    continue
                result = _reocr_text(engine, crop_path, page_number)
                if result is None:
                    continue
                if best is None or result[1] > best[1]:
                    best = result
                if best[1] >= 0.95:
                    break  # 충분히 확실하면 추가 변형 생략
            if best is None:
                continue
            new_text, new_score = best
            old_score = item.score or 0.0
            # 신뢰도가 의미 있게 오르고 길이가 크게 어긋나지 않을 때만 교체
            if new_score <= old_score + 0.02:
                continue
            old_len = max(len(item.text), 1)
            if not (0.3 <= len(new_text) / old_len <= 3.0):
                continue
            item.text = new_text
            item.score = round(new_score, 4)
            replaced += 1
    return replaced
