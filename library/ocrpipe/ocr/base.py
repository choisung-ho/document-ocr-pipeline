"""OCR 엔진 인터페이스와 결과 정규화 헬퍼.

엔진(:class:`OcrEngine`)은 이미지 한 장을 받아 :class:`OcrItem` 리스트를 반환합니다.
무거운 의존성(paddle 등)은 메서드 안에서 lazy import 합니다(미설치여도 import 는 되도록).
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OcrItem:
    """정규화된 OCR 결과 한 블록."""

    page: int
    type: str
    text: str
    bbox: list[int] | None = None
    order: int | None = None
    html: str | None = None
    markdown: str | None = None
    score: float | None = None
    crop: str | None = None  # 잘라낸 블록 이미지 경로(URL)


class OcrEngine:
    """OCR 엔진 베이스 클래스. 하위 클래스는 :meth:`run_page` 를 구현합니다."""

    id: str = "base"
    name: str = "Base"
    description: str = ""
    #: 설치 여부 판단에 쓰이는 파이썬 모듈. 안내 메시지에 사용.
    required_modules: tuple[str, ...] = ()

    def availability(self) -> tuple[bool, str]:
        """(설치됨?, 안내 메시지)."""
        missing = [m for m in self.required_modules if importlib.util.find_spec(m) is None]
        if missing:
            return False, f"미설치 모듈: {', '.join(missing)}"
        return True, "설치됨"

    def run_page(self, image_path: Path, page_number: int) -> list[OcrItem]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 결과 정규화 헬퍼
# --------------------------------------------------------------------------- #

def clean_text(text: str) -> str:
    # NUL·BOM(zero-width) 등 눈에 안 보이는 문자를 제거
    for invisible in ("\x00", "﻿", "​", "‌", "‍"):
        text = text.replace(invisible, "")
    return " ".join(text.split())


def looks_like_heading(text: str) -> bool:
    compact = text.strip()
    if len(compact) > 80:
        return False
    prefixes = ("Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "제", "1.", "2.", "3.", "4.", "5.")
    return compact.startswith(prefixes)


def normalize_type(raw_type: str, text: str) -> str:
    label = (raw_type or "").lower().strip()
    if label in {"markdown", "html"}:
        return label
    if label in {"doc_title", "title", "paragraph_title", "heading", "header", "section_header", "sectionheader", "section-header"}:
        return "heading"
    if label in {"table", "formula", "image", "figure", "caption", "footer", "footnote", "seal", "chart"}:
        return label
    if label in {"picture", "figuregroup", "picturegroup"}:
        return "image"
    if label in {"text", "paragraph", "reference", "plain text", "line", "listitem", "list-item", "list_item", "textinlinemath"}:
        return "paragraph"
    if looks_like_heading(text):
        return "heading"
    return "paragraph"


def normalize_bbox(value: Any) -> list[int] | None:
    """[x1, y1, x2, y2] 또는 4점 폴리곤을 [x1, y1, x2, y2] 로 정규화."""
    if value is None:
        return None
    # 폴리곤 [[x,y], [x,y], ...] 형태 처리
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        xs = [float(p[0]) for p in value if len(p) >= 2]
        ys = [float(p[1]) for p in value if len(p) >= 2]
        if not xs or not ys:
            return None
        return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [int(round(float(item))) for item in value[:4]]
    except (TypeError, ValueError):
        return None


def normalize_order(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def dedupe_key(item_type: str, text: str, bbox: list[int] | None) -> tuple:
    """중복 판정 키: 같은 (타입, 텍스트)라도 위치가 다르면 다른 블록으로 취급.

    표의 반복 값처럼 같은 문구가 페이지의 서로 다른 위치에 실제로 여러 번
    나오는 경우를 보존하기 위해 bbox 중심을 24px 그리드로 양자화해 포함한다.
    (HWP faux-bold 의 이중 그리기 offset 은 수 px 이내라 24px 로도 잡히고,
    표의 인접 행 반복 텍스트는 행 간격이 그보다 커서 보존된다.)
    bbox 가 없는 블록(markdown 내보내기 등)은 기존처럼 텍스트만으로 판정.
    """
    if not bbox or len(bbox) < 4:
        return (item_type, text, None)
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return (item_type, text, (int(center_x // 24), int(center_y // 24)))


def normalize_blocks(blocks: list[Any], page_number: int) -> list[OcrItem]:
    """엔진별 dict 블록 리스트를 중복 제거하며 OcrItem 리스트로 변환."""
    normalized: list[OcrItem] = []
    seen: set[tuple] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = clean_text(str(block.get("text", "")))
        if not text:
            continue
        item_type = normalize_type(str(block.get("type", "")), text)
        bbox = normalize_bbox(block.get("bbox"))
        key = dedupe_key(item_type, text, bbox)
        if key in seen:
            continue
        seen.add(key)
        score = block.get("score")
        normalized.append(
            OcrItem(
                page=page_number,
                type=item_type,
                text=text,
                bbox=bbox,
                order=normalize_order(block.get("order")),
                html=block.get("html") if isinstance(block.get("html"), str) else None,
                markdown=block.get("markdown") if isinstance(block.get("markdown"), str) else None,
                score=float(score) if isinstance(score, (int, float)) else None,
            )
        )
    return normalized


def first_text(value: dict[str, Any]) -> str:
    for key in ("text", "content", "block_content", "markdown", "html"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def collect_text_blocks(value: Any, blocks: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        text = first_text(value)
        if text:
            blocks.append(
                {
                    "text": text,
                    "type": value.get("type")
                    or value.get("label")
                    or value.get("block_type")
                    or value.get("category")
                    or value.get("cls_label"),
                    "bbox": value.get("bbox") or value.get("block_bbox"),
                    "order": value.get("order") or value.get("block_order"),
                }
            )
        for child in value.values():
            collect_text_blocks(child, blocks)
    elif isinstance(value, list):
        for item in value:
            collect_text_blocks(item, blocks)
    elif hasattr(value, "__dict__"):
        collect_text_blocks(vars(value), blocks)


def result_to_plain_data(result: Any) -> Any:
    if isinstance(result, dict):
        return result
    if hasattr(result, "json"):
        try:
            json_value = result.json
            if callable(json_value):
                json_value = json_value()
            return json_value if isinstance(json_value, dict) else json.loads(json_value)
        except Exception:
            pass
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump()
        except Exception:
            pass
    if hasattr(result, "__dict__"):
        return vars(result)
    return {"text": str(result)}
