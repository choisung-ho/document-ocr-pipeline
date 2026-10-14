"""장치(GPU/CPU) 결정 로직.

정책: **GPU 가 있으면 GPU, 없으면 CPU** 로 자동 선택합니다.
사용자가 ``OCR_DEVICE``(또는 ``PADDLE_DEVICE``)로 강제하면 그 값을 따릅니다.

- 설정(원본 환경변수) : :mod:`ocrpipe.config`
- 결정(GPU/CPU 판단)  : 이 모듈  ← 관심사 분리
"""

from __future__ import annotations

import functools
import shutil
import subprocess

from ..config import settings


def _forced_device() -> str | None:
    """OCR_DEVICE 강제값을 정규화: 'cuda' / 'cpu' / None(자동)."""
    value = settings.device_override
    if value in {"cuda", "gpu"}:
        return "cuda"
    if value == "cpu":
        return "cpu"
    return None


@functools.lru_cache(maxsize=1)
def nvidia_gpu_present() -> bool:
    """nvidia-smi 로 물리 GPU 존재 여부 확인 (프레임워크 무관)."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False
    try:
        result = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0 and "GPU" in (result.stdout or "")
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def paddle_cuda_available() -> bool:
    try:
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            return False
        return paddle.device.cuda.device_count() > 0
    except Exception:
        return False


def resolve_paddle_device() -> str:
    """PaddleOCR 용 장치를 결정. paddle 표기는 'gpu'/'cpu'."""
    forced = _forced_device()
    if forced:
        return "gpu" if forced == "cuda" else "cpu"
    return "gpu" if paddle_cuda_available() else "cpu"


def describe() -> str:
    """로그용 장치 요약 문자열."""
    return (
        f"nvidia={nvidia_gpu_present()}, "
        f"paddle={'gpu' if paddle_cuda_available() else 'cpu'}"
    )
