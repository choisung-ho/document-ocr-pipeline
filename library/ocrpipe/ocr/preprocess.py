"""OCR 입력 이미지 전처리 (모든 엔진 공용).

그래프 축 라벨·범례처럼 작고 대비가 낮은 텍스트의 인식률을 높이기 위해
엔진에 이미지를 넘기기 전에 다음을 적용합니다.

- 저해상도 이미지 업스케일 (짧은 변 기준)
- 배경 조명/색 배경 정규화 (L 채널 나눗셈 보정)
- CLAHE 대비 향상 (L 채널)
- 언샤프 마스크 (경계 선명화)

원본 파일은 건드리지 않고 ``<이름>.pre.png`` 를 새로 만들어 반환합니다.
업스케일된 경우 bbox 를 원본 좌표로 되돌릴 수 있도록 배율도 함께 반환합니다.

환경변수:
- ``OCR_PREPROCESS``      : 전체 on/off (기본 1)
- ``OCR_PREPROCESS_BG``   : 배경 정규화 on/off (기본 1)
- ``OCR_PREPROCESS_SHORT``: 업스케일 목표 짧은 변 픽셀 (기본 2000)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import settings


def imread_unicode(path: Path) -> np.ndarray | None:
    """한글 경로에서도 동작하는 이미지 로드 (cv2.imread 는 비ASCII 경로 실패)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def compute_upscale(width: int, height: int) -> float:
    """짧은 변이 목표보다 작으면 확대 배율을 계산 (최대 3배)."""
    target_short = settings.preprocess_short
    short = min(width, height)
    if short <= 0 or short >= target_short:
        return 1.0
    return min(3.0, round(target_short / short, 2))


def normalize_background(l_channel: np.ndarray) -> np.ndarray:
    """넓게 블러한 배경으로 나눠 조명 불균형·연한 색 배경을 평탄화."""
    small = cv2.resize(l_channel, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    background = cv2.GaussianBlur(small, (0, 0), sigmaX=15)
    background = cv2.resize(background, (l_channel.shape[1], l_channel.shape[0]), interpolation=cv2.INTER_LINEAR)
    background = np.maximum(background, 1)
    normalized = l_channel.astype(np.float32) / background.astype(np.float32)
    normalized = normalized * float(np.median(background))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def unsharp_mask(image: np.ndarray, amount: float = 0.6, sigma: float = 1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened


def preprocess_image(image: np.ndarray, min_scale: float = 1.0) -> tuple[np.ndarray, float]:
    """이미지 배열에 전처리를 적용하고 (결과, 업스케일 배율)을 반환.

    ``min_scale`` 로 최소 확대 배율을 보장할 수 있다 (그래프 영역 크롭 등
    작은 글씨가 많은 입력용). 과대 확대로 인한 속도 저하를 막기 위해
    긴 변이 4500px 을 넘지 않도록 배율을 자동 제한한다.
    """
    height, width = image.shape[:2]
    scale = max(compute_upscale(width, height), min_scale)
    long_side = max(width, height)
    if long_side * scale > 4500:
        scale = max(1.0, round(4500 / long_side, 2))
    if scale > 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    if settings.preprocess_bg:
        l_channel = normalize_background(l_channel)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    lab = cv2.merge((l_channel, a_channel, b_channel))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = unsharp_mask(image)
    return image, scale


def preprocess_page_image(image_path: Path, output_path: Path | None = None) -> tuple[Path, float]:
    """페이지 이미지 파일을 전처리해 (새 경로, 업스케일 배율)을 반환.

    비활성화(OCR_PREPROCESS=0)이거나 실패하면 원본 경로와 배율 1.0 을 반환하므로
    호출부는 실패를 신경 쓰지 않아도 됩니다.
    """
    if not settings.preprocess:
        return image_path, 1.0
    image = imread_unicode(image_path)
    if image is None:
        return image_path, 1.0
    try:
        processed, scale = preprocess_image(image)
        target = output_path or image_path.with_suffix(".pre.png")
        if not imwrite_unicode(target, processed):
            return image_path, 1.0
        return target, scale
    except Exception:
        return image_path, 1.0


def rescale_bboxes(items: list, scale: float) -> None:
    """업스케일된 좌표를 원본 이미지 좌표로 되돌린다 (제자리 수정)."""
    if scale == 1.0:
        return
    for item in items:
        if item.bbox:
            item.bbox = [int(round(coord / scale)) for coord in item.bbox]
