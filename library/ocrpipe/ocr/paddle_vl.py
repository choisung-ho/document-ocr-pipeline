"""PaddleOCR-VL 엔진.

레이아웃 분석 + VLM 기반 문서 파싱. parsing_res_list / markdown / table HTML 을 추출합니다.
"""

from __future__ import annotations

import importlib.machinery
import sys
import threading
import types
from pathlib import Path
from typing import Any

from .base import (
    OcrEngine,
    OcrItem,
    collect_text_blocks,
    normalize_blocks,
    result_to_plain_data,
)

_PIPELINE: Any | None = None
_PIPELINE_LOCK = threading.Lock()


class PaddleVLEngine(OcrEngine):
    id = "paddleocr-vl"
    name = "PaddleOCR-VL"
    description = "레이아웃 검출 + VLM 문서 파싱. 표/수식/마크다운 구조화 결과 제공."
    required_modules = ("paddle", "paddleocr")

    def run_page(self, image_path: Path, page_number: int) -> list[OcrItem]:
        pipeline = _get_pipeline()
        output = pipeline.predict(
            input=str(image_path),
            use_layout_detection=True,
            use_ocr_for_image_block=True,
            format_block_content=True,
            merge_layout_blocks=True,
        )
        raw_items: list[Any] = []
        for result in output:
            raw_items.extend(_extract_blocks(result, page_number))
        return normalize_blocks(raw_items, page_number)


def _get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            import paddle

            _install_fake_torch()
            from paddleocr import PaddleOCRVL

            from ..config import settings

            device = _resolve_device(paddle)
            options: dict[str, Any] = {
                "device": device,
                "use_layout_detection": True,
                "use_ocr_for_image_block": True,
                "format_block_content": True,
                "merge_layout_blocks": True,
            }
            # 모델(파이프라인) 버전을 고정하고 싶을 때만 전달.
            # 미지정 시 설치된 paddleocr 의 기본 버전을 그대로 쓴다(현재 기본 v1.6).
            if settings.vl_version:
                options["pipeline_version"] = settings.vl_version
            _PIPELINE = PaddleOCRVL(**options)
    return _PIPELINE


def _resolve_device(paddle: Any) -> str:
    from ..config import settings
    from .device import resolve_paddle_device

    requested = settings.paddle_device_override
    if not requested:
        # 미지정 시 자동: GPU 있으면 gpu, 없으면 cpu
        return resolve_paddle_device()
    wants_gpu = requested.startswith("gpu")
    cuda_available = bool(paddle.device.is_compiled_with_cuda())
    if wants_gpu and not cuda_available:
        raise RuntimeError(
            "GPU 모드를 요청했지만 PaddlePaddle 이 CUDA 로 빌드되지 않았습니다. "
            "paddlepaddle-gpu 를 설치하거나 PADDLE_DEVICE=cpu 로 실행하세요."
        )
    return requested


def _install_fake_torch() -> None:
    """PaddleOCR-VL import 시 torch 가 없어도 동작하도록 최소 stub 주입."""
    if "torch" in sys.modules:
        return
    fake_torch = types.ModuleType("torch")
    fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    fake_torch.__version__ = "0.0.0"

    multiprocessing = types.ModuleType("torch.multiprocessing")
    multiprocessing.get_start_method = lambda allow_none=True: "spawn"
    multiprocessing.set_start_method = lambda *args, **kwargs: None

    distributed = types.ModuleType("torch.distributed")
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    distributed.get_rank = lambda group=None: 0

    fake_torch.multiprocessing = multiprocessing
    fake_torch.distributed = distributed
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        set_device=lambda *args, **kwargs: None,
        manual_seed_all=lambda *args, **kwargs: None,
    )
    fake_torch.nn = types.SimpleNamespace(Module=object, Linear=object)
    fake_torch.Tensor = object
    fake_torch.LongTensor = object

    sys.modules["torch"] = fake_torch
    sys.modules["torch.multiprocessing"] = multiprocessing
    sys.modules["torch.distributed"] = distributed


# --------------------------------------------------------------------------- #
# 결과 추출
# --------------------------------------------------------------------------- #

def _extract_blocks(result: Any, page_number: int | None = None) -> list[dict[str, Any]]:
    data = result_to_plain_data(result)
    candidates = []
    for key in ("pruned_res", "prunedResult", "layout_parsing_result", "layoutParsingResult", "res"):
        value = data.get(key) if isinstance(data, dict) else None
        if value:
            candidates.append(value)
    candidates.append(data)

    blocks: list[dict[str, Any]] = []
    for candidate in candidates:
        _collect_parsing_blocks(candidate, blocks)
        collect_text_blocks(candidate, blocks)
    _append_export_blocks(result, blocks)
    return blocks


def _collect_parsing_blocks(value: Any, blocks: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        parsing_res_list = value.get("parsing_res_list")
        if isinstance(parsing_res_list, list):
            for item in parsing_res_list:
                if not isinstance(item, dict) and hasattr(item, "__dict__"):
                    item = vars(item)
                if not isinstance(item, dict):
                    continue
                text = item.get("block_content") or item.get("content")
                if isinstance(text, str) and text.strip():
                    block_type = item.get("block_label") or item.get("label")
                    html = (
                        text.strip()
                        if str(block_type).lower() == "table" and "<table" in text.lower()
                        else None
                    )
                    blocks.append(
                        {
                            "text": text.strip(),
                            "type": block_type,
                            "bbox": item.get("block_bbox") or item.get("bbox"),
                            "order": item.get("block_order"),
                            "html": html,
                        }
                    )
        for child in value.values():
            _collect_parsing_blocks(child, blocks)
    elif isinstance(value, list):
        for item in value:
            _collect_parsing_blocks(item, blocks)
    elif hasattr(value, "__dict__"):
        _collect_parsing_blocks(vars(value), blocks)


def _append_export_blocks(result: Any, blocks: list[dict[str, Any]]) -> None:
    markdown = _get_export(result, "markdown")
    if isinstance(markdown, dict):
        markdown_text = markdown.get("markdown_text") or markdown.get("text")
        if isinstance(markdown_text, str) and markdown_text.strip():
            blocks.append(
                {
                    "text": markdown_text.strip(),
                    "type": "markdown",
                    "markdown": markdown_text.strip(),
                    "order": 10_000,
                }
            )

    html = _get_export(result, "html")
    if isinstance(html, dict):
        for table_html in html.values():
            if isinstance(table_html, str) and table_html.strip():
                blocks.append(
                    {
                        "text": table_html.strip(),
                        "type": "table",
                        "html": table_html.strip(),
                        "order": 20_000,
                    }
                )


def _get_export(result: Any, name: str) -> Any:
    try:
        value = getattr(result, name)
        return value() if callable(value) else value
    except Exception:
        return None
