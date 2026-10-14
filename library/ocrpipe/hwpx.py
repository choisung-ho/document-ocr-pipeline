"""HWPX(한글 표준 포맷) 직접 파서 — PDF 변환 없이 본문 텍스트 + 삽입 이미지 추출.

``.hwpx`` 는 **zip + XML**(OWPML) 이므로 표준 라이브러리만으로 내용을 열 수 있습니다.
따라서 한컴오피스·LibreOffice 같은 외부 변환 도구가 **전혀 필요 없습니다**.

- **본문 텍스트** : ``Contents/section*.xml`` 의 문단(``<p>``) → 무손실 추출(score 1.0).
- **삽입 이미지** : ``BinData/*`` 에 원본 그대로 저장돼 있어 **그대로 꺼내서** OCR 합니다.
  (PDF 로 변환·재렌더링하지 않으므로 화질 손실이 없습니다.)

미리보기 이미지(``Preview/PrvImage.png``)는 문서 내용이 아니라 썸네일이므로 제외합니다.

지원 범위: ``.hwpx`` 전용입니다. 구형 바이너리 ``.hwp`` 는 zip 이 아니라 이 경로를 쓸 수
없으므로, 한컴오피스에서 ``.hwpx`` 로 저장한 뒤 넣어 주세요.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .config import settings
from .ocr import OcrItem
from .ocr.base import clean_text, looks_like_heading

#: BinData 안에서 OCR 대상으로 삼는 래스터 이미지 확장자.
#: (emf/wmf 는 벡터 도형이라 OCR 대상이 아니므로 제외)
_RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def is_hwpx(path: Path) -> bool:
    """이 모듈이 직접 처리할 수 있는 형식인지(.hwpx 전용)."""
    return path.suffix.lower() == ".hwpx"


def unsupported_reason(path: Path) -> str | None:
    """처리 불가 형식이면 안내 문구, 가능하면 None."""
    suffix = path.suffix.lower()
    if suffix == ".hwpx":
        return None
    if suffix == ".hwp":
        return (
            "구형 바이너리 .hwp 는 지원하지 않습니다(zip 구조가 아니라 직접 파싱 불가).\n"
            "한컴오피스에서 '다른 이름으로 저장 → HWPX' 로 변환한 뒤 다시 시도하세요."
        )
    return f"HWPX 파서가 지원하지 않는 형식: {suffix}"


# --------------------------------------------------------------------------- #
# 본문 텍스트
# --------------------------------------------------------------------------- #

def _local(tag: str) -> str:
    """네임스페이스를 제거한 로컬 태그명."""
    return tag.rsplit("}", 1)[-1]


def _section_names(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name for name in archive.namelist()
        if name.startswith("Contents/section") and name.endswith(".xml")
    )


def extract_paragraphs(path: Path) -> list[str]:
    """``Contents/section*.xml`` 의 문단 텍스트를 순서대로 반환."""
    paragraphs: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in _section_names(archive):
            root = ET.fromstring(archive.read(name))
            for para in root.iter():
                if _local(para.tag) != "p":
                    continue
                # 문단(<p>) 안의 모든 텍스트런(<t>)을 이어붙여 한 줄로
                runs = [
                    node.text for node in para.iter()
                    if _local(node.tag) == "t" and node.text
                ]
                line = "".join(runs).strip()
                if line:
                    paragraphs.append(line)
    return paragraphs


def text_items(path: Path, order_base: int = 0) -> list[OcrItem]:
    """본문 문단을 OcrItem 리스트로 반환(score 1.0 = 원문 무손실)."""
    items: list[OcrItem] = []
    order = order_base
    for text in extract_paragraphs(path):
        cleaned = clean_text(text)
        if not cleaned:
            continue
        items.append(
            OcrItem(
                page=1,
                type="heading" if looks_like_heading(cleaned) else "paragraph",
                text=cleaned,
                bbox=None,  # 흐름 문서라 페이지 좌표가 없음
                order=order,
                score=1.0,  # 원문 텍스트 직접 추출 → 인식 오차 없음
            )
        )
        order += 1
    return items


# --------------------------------------------------------------------------- #
# 삽입 이미지 (BinData)
# --------------------------------------------------------------------------- #

def extract_images(path: Path, out_dir: Path) -> list[Path]:
    """``BinData/`` 의 래스터 이미지를 ``out_dir`` 에 원본 그대로 풀어 경로 목록 반환.

    ``settings.min_region_side`` 보다 짧은 변을 가진 이미지(아이콘·장식)는 제외합니다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name for name in archive.namelist()
            if name.startswith("BinData/")
            and Path(name).suffix.lower() in _RASTER_SUFFIXES
        )
        for index, name in enumerate(names):
            target = out_dir / f"img_{index:02d}{Path(name).suffix.lower()}"
            target.write_bytes(archive.read(name))
            if _too_small(target):
                target.unlink(missing_ok=True)
                continue
            saved.append(target)
    return saved


def _too_small(path: Path) -> bool:
    """아이콘·장식처럼 너무 작은 이미지인지(짧은 변 기준)."""
    from .ocr.preprocess import imread_unicode

    image = imread_unicode(path)
    if image is None:
        return True
    height, width = image.shape[:2]
    return min(width, height) < settings.min_region_side
