"""OCR 결과 CER(문자 오류율) 측정 도구.

OCR 결과의 "신뢰도(score)"는 모델의 자기 보고 값이라 실제 정답률이 아닙니다.
이 도구는 사람이 검수한 정답 텍스트와 비교해 **실제 오류율**을 측정합니다.

사용 절차:

1) 정답 초안 생성 — PDF 텍스트 레이어에서 페이지별 텍스트를 뽑아 초안 파일 생성
       python -m ocrpipe.eval_cer --make-drafts 문서.pdf --pages 14,48,50

   → ``eval/ground_truth/page_0014.txt`` 등이 생성됨.
   ※ 초안은 텍스트 레이어 기준이므로 **그래프·그림 안의 텍스트는 사람이 직접
     추가하고 전체를 검수해야** 진짜 정답이 됩니다.

2) 측정 — 결과 JSON 의 페이지별 텍스트를 정답과 비교해 CER 계산
       python -m ocrpipe.eval_cer OCR_result\\문서_paddleocr-vl.json

   CER = 편집거리(삽입+삭제+교체) / 정답 글자 수.
   기본은 공백을 제거하고 비교합니다(--keep-space 로 공백 포함 비교).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_GT_DIR = Path(__file__).resolve().parent.parent / "eval" / "ground_truth"


# --------------------------------------------------------------------------- #
# 텍스트 정규화 / 편집거리
# --------------------------------------------------------------------------- #

def normalize(text: str, keep_space: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[﻿​‌‍]", "", text)  # BOM·zero-width 제거
    if keep_space:
        return re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+", "", text)


def levenshtein(a: str, b: str) -> int:
    """두 문자열의 편집거리 (2행 DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


# --------------------------------------------------------------------------- #
# 결과 JSON → 페이지 텍스트
# --------------------------------------------------------------------------- #

def page_text(items: list[dict], page: int) -> str:
    """결과 블록을 읽기 순서(위→아래, 왼→오른쪽)로 이어붙인다."""
    blocks = [it for it in items if it.get("page") == page and it.get("text")]
    with_bbox = [it for it in blocks if it.get("bbox")]
    if len(with_bbox) >= len(blocks) * 0.8 and with_bbox:
        heights = sorted(b["bbox"][3] - b["bbox"][1] for b in with_bbox)
        row_height = max(heights[len(heights) // 2], 8)

        def sort_key(it):
            if it.get("bbox"):
                x1, y1, x2, y2 = it["bbox"]
                return (int(((y1 + y2) / 2) // (row_height * 0.8)), (x1 + x2) / 2)
            return (10**9, it.get("order") or 0)

        blocks.sort(key=sort_key)
    else:
        blocks.sort(key=lambda it: it.get("order") or 0)
    return "\n".join(it["text"] for it in blocks)


# --------------------------------------------------------------------------- #
# 정답 초안 생성
# --------------------------------------------------------------------------- #

def make_drafts(pdf_path: Path, pages: list[int], out_dir: Path) -> None:
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    with fitz.open(pdf_path) as doc:
        for page_number in pages:
            if page_number < 1 or page_number > doc.page_count:
                print(f"  page {page_number}: 범위 밖 (총 {doc.page_count}p)", file=sys.stderr)
                continue
            lines = _draft_lines(doc[page_number - 1])
            text = "\n".join(lines)
            draft_path = out_dir / f"page_{page_number:04d}.txt"
            header = (
                "# 정답 초안 — PDF 텍스트 레이어 기준.\n"
                "# 그래프·그림 안의 텍스트는 직접 추가하고 전체를 검수한 뒤 이 주석을 지우세요.\n"
            )
            draft_path.write_text(header + text, encoding="utf-8")
            print(f"  초안 생성: {draft_path} ({len(text)}자)", file=sys.stderr)


def _draft_lines(page) -> list[str]:
    """텍스트 레이어를 줄 단위로 추출하되, HWP 변환기의 faux-bold
    (굵은 글씨를 같은 위치에 두 번 그리는) 중복 런을 제거한다."""
    raw = page.get_text("dict", sort=True)
    lines: list[str] = []
    seen: set[tuple] = set()
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", "")).strip()
            if not text:
                continue
            x1, y1, x2, y2 = line.get("bbox", (0, 0, 0, 0))
            # 같은 텍스트가 거의 같은 위치(2pt 그리드)에 다시 나오면 faux-bold 중복
            key = (text, round((x1 + x2) / 4), round((y1 + y2) / 4))
            if key in seen:
                continue
            seen.add(key)
            lines.append(text)
    return lines


# --------------------------------------------------------------------------- #
# 측정
# --------------------------------------------------------------------------- #

def page_cer(gt_lines: list[str], ocr_texts: list[str]) -> tuple[int, int, int]:
    """줄 정렬 매칭 CER: (편집거리 합, 정답 글자 수, GT 외 추가 글자 수).

    표·다단 페이지에서는 읽기 순서가 도구마다 달라 연결 문자열 비교(CER)가
    순서 차이만으로 크게 오염된다. 대신 정답의 각 줄을 가장 비슷한 OCR
    블록과 짝지어(길이 가중 탐욕 매칭) 줄 단위 편집거리를 합산한다.
    짝을 찾지 못한 정답 줄은 전체 삭제로 계산하고, 남은 OCR 블록 글자 수는
    'GT 외 추가'(그래프 텍스트 등 정답에 없는 추출)로 따로 보고한다.
    """
    pool = [t for t in ocr_texts if t]
    used = [False] * len(pool)
    total_distance = 0
    total_chars = 0
    for gt_line in sorted(gt_lines, key=len, reverse=True):
        if not gt_line:
            continue
        total_chars += len(gt_line)
        best_index, best_distance = -1, None
        for index, candidate in enumerate(pool):
            if used[index]:
                continue
            if abs(len(candidate) - len(gt_line)) >= len(gt_line) and best_distance is not None:
                continue
            distance = levenshtein(gt_line, candidate)
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
                if distance == 0:
                    break
        # 절반 이상 일치하는 블록만 짝으로 인정
        if best_distance is not None and best_distance < len(gt_line) * 0.5:
            used[best_index] = True
            total_distance += best_distance
        else:
            total_distance += len(gt_line)  # 누락(전체 삭제)
    extra_chars = sum(len(pool[i]) for i in range(len(pool)) if not used[i])
    return total_distance, total_chars, extra_chars


def evaluate(result_path: Path, gt_dir: Path, keep_space: bool) -> None:
    items = json.loads(result_path.read_text(encoding="utf-8"))
    gt_files = sorted(gt_dir.glob("page_*.txt"))
    if not gt_files:
        raise SystemExit(f"정답 파일이 없습니다: {gt_dir} (--make-drafts 로 초안을 먼저 만드세요)")

    print(f"결과: {result_path.name}")
    print(f"정답: {gt_dir} ({len(gt_files)}페이지) — 줄 정렬 매칭 CER")
    print(f"{'페이지':>6} {'정답글자':>8} {'편집거리':>8} {'CER':>8} {'GT외 추가':>9}")

    grand_distance = 0
    grand_chars = 0
    for gt_file in gt_files:
        page_number = int(gt_file.stem.split("_")[1])
        gt_raw = gt_file.read_text(encoding="utf-8")
        if gt_raw.lstrip().startswith("#"):
            print(f"{page_number:>6}  (미검수 초안 — 주석이 남아 있어 건너뜀)")
            continue
        gt_lines = [normalize(line, keep_space) for line in gt_raw.splitlines() if line.strip()]
        ocr_texts = [
            normalize(it["text"], keep_space)
            for it in items
            if it.get("page") == page_number and it.get("text")
        ]
        if not gt_lines:
            continue
        distance, chars, extra = page_cer(gt_lines, ocr_texts)
        grand_distance += distance
        grand_chars += chars
        print(f"{page_number:>6} {chars:>8} {distance:>8} {distance / chars:>7.2%} {extra:>9}")

    if grand_chars:
        print(f"{'전체':>6} {grand_chars:>8} {grand_distance:>8} {grand_distance / grand_chars:>7.2%}")
    else:
        print("측정된 페이지가 없습니다. 초안 검수 후 파일 상단 주석(#)을 지우세요.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR 결과 CER 측정")
    parser.add_argument("target", nargs="?", help="결과 JSON 경로 (측정 모드)")
    parser.add_argument("--make-drafts", metavar="PDF", help="정답 초안을 만들 PDF 경로")
    parser.add_argument("--pages", default="", help="초안 페이지, 예: 14,48,50")
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR), help="정답 폴더")
    parser.add_argument("--keep-space", action="store_true", help="공백 포함 비교")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    if args.make_drafts:
        pages = [int(p) for p in args.pages.replace(" ", "").split(",") if p.isdigit()]
        if not pages:
            parser.error("--pages 로 초안을 만들 페이지를 지정하세요. 예: --pages 14,48")
        make_drafts(Path(args.make_drafts), pages, gt_dir)
        return
    if not args.target:
        parser.error("결과 JSON 경로를 지정하거나 --make-drafts 를 사용하세요.")
    evaluate(Path(args.target), gt_dir, args.keep_space)


if __name__ == "__main__":
    main()
