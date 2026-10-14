"""PaddleOCR-VL 추론 워커.

추론은 별도 프로세스에서 실행된다. libpaddle의 세그폴트가 발생해도
서버 프로세스는 살아남고, 해당 페이지만 오류로 처리한 뒤 계속 진행한다.
이 모듈은 자식 프로세스가 단독으로 임포트하므로 서버(app.main)에
의존하면 안 된다. paddle 임포트는 함수 내부에서 지연 수행한다.
"""

from __future__ import annotations

import json
import os
from typing import Any


def create_paddle_pipeline() -> Any:
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR-VL-1.6 의존성이 설치되어 있지 않습니다. "
            "`pip install -r requirements-ocr.txt` 후 다시 실행하거나 OCR_MOCK=1로 테스트하세요."
        ) from exc

    return PaddleOCRVL(
        pipeline_version="v1.6",
        device=os.getenv("PADDLE_DEVICE", "cpu"),
        use_layout_detection=True,
        use_chart_recognition=True,
        use_ocr_for_image_block=True,
        format_block_content=True,
    )


def predict_blocks(pipeline: Any, page_path: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for result in pipeline.predict(input=page_path):
        blocks.extend(extract_paddle_blocks(result))
    return blocks


def run_worker(request_queue: Any, response_queue: Any) -> None:
    """워커 프로세스 본체. 파이프라인을 1회 로딩해 요청마다 재사용한다."""
    pipeline: Any | None = None
    while True:
        request = request_queue.get()
        if request is None:
            return
        request_id, page_path = request
        try:
            if pipeline is None:
                pipeline = create_paddle_pipeline()
            response_queue.put((request_id, "ok", predict_blocks(pipeline, page_path)))
        except Exception as exc:  # noqa: BLE001 — 워커에서는 모든 예외를 부모로 전달한다.
            response_queue.put((request_id, "error", f"{type(exc).__name__}: {exc}"))


def extract_paddle_blocks(result: Any) -> list[dict[str, Any]]:
    data = result_to_plain_data(result)
    candidates = []
    for key in ("pruned_res", "prunedResult", "layout_parsing_result", "layoutParsingResult", "res"):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            candidates.append(value)
    candidates.append(data)

    blocks: list[dict[str, Any]] = []
    for candidate in candidates:
        collect_paddle_parsing_blocks(candidate, blocks)
        collect_text_blocks(candidate, blocks)
    return blocks


def result_to_plain_data(result: Any) -> Any:
    # PaddleOCR result objects inherit from dict, but values such as
    # parsing_res_list still contain PaddleOCRVLBlock instances. Prefer the
    # public JSON representation so those blocks become plain dictionaries.
    if hasattr(result, "json"):
        try:
            json_value = result.json
            if callable(json_value):
                json_value = json_value()
            return json_value if isinstance(json_value, dict) else json.loads(json_value)
        except Exception:
            pass
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        try:
            return result.model_dump()
        except Exception:
            pass
    if hasattr(result, "__dict__"):
        return vars(result)
    return {"text": str(result)}


def collect_paddle_parsing_blocks(value: Any, blocks: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        parsing_res_list = value.get("parsing_res_list")
        if isinstance(parsing_res_list, list):
            for item in parsing_res_list:
                if not isinstance(item, dict):
                    continue
                text = item.get("block_content")
                if isinstance(text, str) and text.strip():
                    blocks.append(
                        {
                            "text": text.strip(),
                            "type": item.get("block_label"),
                            "bbox": item.get("block_bbox") or item.get("block_polygon_points"),
                        }
                    )
        for child in value.values():
            collect_paddle_parsing_blocks(child, blocks)
    elif isinstance(value, list):
        for item in value:
            collect_paddle_parsing_blocks(item, blocks)


def collect_text_blocks(value: Any, blocks: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        text = first_text(value)
        if text:
            blocks.append(
                {
                    "text": text,
                    "type": value.get("type")
                    or value.get("label")
                    or value.get("block_label")
                    or value.get("block_type")
                    or value.get("category")
                    or value.get("cls_label"),
                    "bbox": value.get("bbox")
                    or value.get("block_bbox")
                    or value.get("block_polygon_points"),
                }
            )
        for child in value.values():
            collect_text_blocks(child, blocks)
    elif isinstance(value, list):
        for item in value:
            collect_text_blocks(item, blocks)


def first_text(value: dict[str, Any]) -> str:
    for key in ("text", "content", "block_content", "markdown", "html"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""
