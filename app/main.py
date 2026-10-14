from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Literal

import fitz
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import paddle_worker
from .document_parsers import (
    DocumentItem,
    analyze_pdf_page,
    count_processing_units,
    describe_chart,
    extract_docx,
    extract_hwpx,
    extract_metrics,
    extract_pdf_native_page,
    parse_chart_table,
    render_pdf_page,
    render_pdf_region,
)
from .paddle_worker import extract_paddle_blocks, result_to_plain_data

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
UPLOAD_DIR = ROOT_DIR / "uploads"
PAGE_DIR = ROOT_DIR / "page_images"
ASSET_DIR = ROOT_DIR / "assets"
RESULT_DIR = ROOT_DIR / "job_results"
MODEL_NAME = "paddleocr-vl-1.6"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".hwpx", *IMAGE_EXTENSIONS}
ALLOWED_DOCUMENT_TYPES = {"general", "evidence"}
ALLOWED_SECURITY_LEVELS = {"public", "internal", "restricted"}

app = FastAPI(title="Document Preprocessing and OCR Test App")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/assets", StaticFiles(directory=ASSET_DIR, check_dir=False), name="assets")


@dataclass
class PageStatus:
    page: int
    status: Literal["waiting", "processing", "done", "error"]
    elapsed: float | None = None
    error: str | None = None
    mode: str | None = None
    item_count: int = 0


@dataclass
class JobState:
    id: str
    model: str
    filename: str
    title: str
    document_type: str
    rag_eligible: bool
    security_level: str
    access_groups: list[str]
    status: Literal["queued", "running", "done", "error"]
    total_pages: int
    pages: list[PageStatus]
    results: list[DocumentItem]
    progress: float = 0.0
    progress_message: str = "대기 중"
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


JOBS: dict[str, JobState] = {}
PADDLE_PIPELINE: Any | None = None
PADDLE_PIPELINE_LOCK = threading.Lock()
PADDLE_INFERENCE_LOCK = threading.Lock()
JOB_PERSIST_LOCK = threading.Lock()


def log_job(job_id: str, message: str) -> None:
    print(f"[OCR][{job_id}] {message}", flush=True)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    document: UploadFile = File(...),
    title: str = Form(""),
    document_type: str = Form("general"),
    security_level: str = Form("internal"),
    access_groups: str = Form(""),
) -> JSONResponse:
    if not document.filename:
        raise HTTPException(status_code=400, detail="문서 파일을 업로드하세요.")
    suffix = Path(document.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="PDF, DOCX, HWPX 또는 PNG/JPG/BMP/TIFF/WEBP 이미지만 업로드할 수 있습니다.",
        )
    if document_type not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail="지원하지 않는 문서 유형입니다.")
    if security_level not in ALLOWED_SECURITY_LEVELS:
        raise HTTPException(status_code=400, detail="지원하지 않는 보안 등급입니다.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_expired_artifacts()

    job_id = uuid.uuid4().hex
    safe_name = Path(document.filename).name
    document_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
    await save_upload(document, document_path)

    try:
        total_pages = count_processing_units(document_path)
    except Exception as exc:
        document_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"문서 확인 실패: {exc}") from exc

    now = time.time()
    groups = [group.strip() for group in access_groups.split(",") if group.strip()]
    JOBS[job_id] = JobState(
        id=job_id,
        model=MODEL_NAME,
        filename=safe_name,
        title=title.strip() or Path(safe_name).stem,
        document_type=document_type,
        rag_eligible=document_type != "evidence",
        security_level=security_level,
        access_groups=groups,
        status="queued",
        total_pages=total_pages,
        pages=[PageStatus(page=i, status="waiting") for i in range(1, total_pages + 1)],
        results=[],
        created_at=now,
        updated_at=now,
    )
    persist_job(JOBS[job_id])
    background_tasks.add_task(process_job, job_id, document_path)
    return JSONResponse(
        {
            "job_id": job_id,
            "total_pages": total_pages,
            "rag_eligible": document_type != "evidence",
        }
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> JSONResponse:
    job = get_existing_job(job_id)
    return JSONResponse(serialize_job(job))


@app.get("/api/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    get_existing_job(job_id)

    async def stream():
        last_payload = ""
        while True:
            if await request.is_disconnected():
                break
            job = get_existing_job(job_id)
            payload = json.dumps(serialize_job(job), ensure_ascii=False)
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
            if job.status in {"done", "error"}:
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(stream(), media_type="text/event-stream")


def get_existing_job(job_id: str) -> JobState:
    job = JOBS.get(job_id)
    if not job:
        job = load_persisted_job(job_id)
        if job:
            JOBS[job_id] = job
    if not job:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return job


def serialize_job(job: JobState) -> dict[str, Any]:
    return {
        **asdict(job),
        "results": [asdict(item) for item in job.results],
        "pages": [asdict(page) for page in job.pages],
        "metadata": {
            "doc_id": job.id,
            "filename": job.filename,
            "title": job.title,
            "document_type": job.document_type,
            "rag_eligible": job.rag_eligible,
            "security_level": job.security_level,
            "access_groups": job.access_groups,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        },
    }


async def save_upload(upload: UploadFile, destination: Path) -> None:
    max_bytes = max(int(os.getenv("OCR_MAX_UPLOAD_MB", "100")), 1) * 1024 * 1024
    written = 0
    try:
        with destination.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail=f"파일 크기는 {max_bytes // (1024 * 1024)}MB 이하여야 합니다.")
                handle.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def persist_job(job: JobState) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULT_DIR / f"{job.id}.json"
    temporary = destination.with_suffix(".json.tmp")
    with JOB_PERSIST_LOCK:
        temporary.write_text(json.dumps(serialize_job(job), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)


def load_persisted_job(job_id: str) -> JobState | None:
    path = RESULT_DIR / f"{job_id}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return JobState(
            id=data["id"],
            model=data["model"],
            filename=data["filename"],
            title=data["title"],
            document_type=data["document_type"],
            rag_eligible=bool(data["rag_eligible"]),
            security_level=data["security_level"],
            access_groups=list(data.get("access_groups", [])),
            status=data["status"],
            total_pages=int(data["total_pages"]),
            pages=[PageStatus(**page) for page in data.get("pages", [])],
            results=[DocumentItem(**item) for item in data.get("results", [])],
            progress=float(data.get("progress", 0)),
            progress_message=data.get("progress_message", "대기 중"),
            error=data.get("error"),
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def cleanup_expired_artifacts() -> None:
    retention_seconds = max(int(os.getenv("OCR_RETENTION_HOURS", "24")), 1) * 3600
    cutoff = time.time() - retention_seconds
    for directory in (UPLOAD_DIR, PAGE_DIR, ASSET_DIR, RESULT_DIR):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError:
                continue


async def process_job(job_id: str, document_path: Path) -> None:
    job = JOBS[job_id]
    job.status = "running"
    job.updated_at = time.time()
    update_job_progress(job, 1, "문서 처리 준비 중")
    try:
        log_job(job_id, f"started: model={MODEL_NAME}, file={document_path.name}, units={job.total_pages}")
        runner = OcrRunner()
        await asyncio.to_thread(process_document, job, document_path, runner)
        job.status = "done" if not any(page.status == "error" for page in job.pages) else "error"
        if job.status == "error":
            job.error = "일부 문서 처리에 실패했습니다."
            update_job_progress(job, job.progress, "처리 오류")
        else:
            update_job_progress(job, 100, "처리 완료")
        log_job(job_id, f"finished: status={job.status}, result_items={len(job.results)}")
    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        update_job_progress(job, job.progress, "처리 오류")
        log_job(job_id, f"failed: {exc}")
    finally:
        job.updated_at = time.time()
        persist_job(job)
        if os.getenv("OCR_DELETE_SOURCE_AFTER_PROCESS", "1").lower() in {"1", "true", "yes"}:
            document_path.unlink(missing_ok=True)


def process_document(job: JobState, document_path: Path, runner: "OcrRunner") -> None:
    suffix = document_path.suffix.lower()
    asset_dir = ASSET_DIR / job.id
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_url_prefix = f"/assets/{job.id}"

    if suffix == ".pdf":
        process_pdf(job, document_path, runner, asset_dir, asset_url_prefix)
        return
    if suffix in IMAGE_EXTENSIONS:
        process_image(job, document_path, runner, asset_dir, asset_url_prefix)
        return

    mark_page(job, 1, status="processing", mode="native")
    update_job_progress(job, 10, "문서 구조 분석 중")
    started = time.perf_counter()
    try:
        if suffix == ".docx":
            items = extract_docx(document_path, asset_dir, asset_url_prefix)
        elif suffix == ".hwpx":
            items = extract_hwpx(document_path, asset_dir, asset_url_prefix)
        else:
            raise RuntimeError(f"지원하지 않는 문서 형식입니다: {suffix}")
        if not items:
            raise RuntimeError("문서에서 추출된 텍스트·표·이미지가 없습니다.")
        processing_mode = "native+image-ocr" if any(item.type == "image" for item in items) else "native"
        update_job_progress(job, 35, "문서 내부 이미지 확인 중")
        items = enrich_embedded_images_with_ocr(
            job,
            items,
            runner,
            asset_dir,
            progress_start=35,
            progress_end=92,
        )
        job.results.extend(items)
        update_job_progress(job, 95, "구조화 결과 정리 중")
        mark_page(
            job,
            1,
            status="done",
            elapsed=time.perf_counter() - started,
            mode=processing_mode,
            item_count=len(items),
        )
    except Exception as exc:
        mark_page(
            job,
            1,
            status="error",
            elapsed=time.perf_counter() - started,
            error=str(exc),
            mode="native",
        )


@dataclass
class PreparedPdfPage:
    """프리페치 스레드가 미리 수행한 페이지 분석·렌더링 결과."""

    page_number: int
    mode: str = "native"
    text_char_count: int = 0
    text_quality: float = 0.0
    image_coverage: float = 0.0
    page_path: Path | None = None
    preview_name: str = ""
    native_items: list[DocumentItem] = field(default_factory=list)
    region_paths: list[tuple[list[float], Path]] = field(default_factory=list)
    error: str | None = None


def prepare_pdf_page(
    document: fitz.Document,
    page_number: int,
    page_dir: Path,
    asset_dir: Path,
) -> PreparedPdfPage:
    page = document[page_number - 1]
    analysis = analyze_pdf_page(page)
    prepared = PreparedPdfPage(
        page_number=page_number,
        mode=analysis.mode,
        text_char_count=analysis.text_char_count,
        text_quality=analysis.text_quality,
        image_coverage=analysis.image_coverage,
        preview_name=f"page_{page_number:04d}_source.png",
    )
    preview_path = asset_dir / prepared.preview_name
    if analysis.mode == "ocr":
        # 2배 고해상도 렌더링은 OCR 입력용 — OCR 페이지에서만 수행한다.
        prepared.page_path = page_dir / f"page_{page_number:04d}.png"
        render_pdf_page(document, page_number, prepared.page_path)
        shutil.copy2(prepared.page_path, preview_path)
    else:
        render_pdf_page(document, page_number, preview_path, scale=1.0)
        prepared.native_items = extract_pdf_native_page(page, page_number)
        for index, region in enumerate(analysis.ocr_regions, start=1):
            region_path = page_dir / f"page_{page_number:04d}_region_{index:03d}.png"
            render_pdf_region(page, region, region_path)
            prepared.region_paths.append((region, region_path))
    return prepared


def prefetch_pdf_pages(
    document_path: Path,
    page_dir: Path,
    asset_dir: Path,
    queue: "Queue[PreparedPdfPage | None]",
    stop_event: threading.Event,
) -> None:
    """OCR이 도는 동안 다음 페이지의 분석·렌더링을 미리 수행해 큐에 쌓는다.

    소비자와 fitz 문서를 공유하지 않도록 스레드 전용 문서 핸들을 연다.
    """
    try:
        with fitz.open(document_path) as document:
            for page_number in range(1, document.page_count + 1):
                if stop_event.is_set():
                    return
                try:
                    prepared = prepare_pdf_page(document, page_number, page_dir, asset_dir)
                except Exception as exc:
                    prepared = PreparedPdfPage(page_number=page_number, error=str(exc))
                queue.put(prepared)
    except Exception as exc:
        queue.put(PreparedPdfPage(page_number=0, error=str(exc)))
    finally:
        queue.put(None)


def process_pdf(
    job: JobState,
    document_path: Path,
    runner: "OcrRunner",
    asset_dir: Path,
    asset_url_prefix: str,
) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    page_dir = PAGE_DIR / job.id
    page_dir.mkdir(parents=True, exist_ok=True)

    # 분석·렌더링(CPU)은 프리페치 스레드에서 비동기로 앞서 진행하고,
    # 이 루프는 OCR·병합만 담당한다. 큐 크기로 과도한 선행 작업을 막는다.
    queue: "Queue[PreparedPdfPage | None]" = Queue(maxsize=3)
    stop_event = threading.Event()
    prefetcher = threading.Thread(
        target=prefetch_pdf_pages,
        args=(document_path, page_dir, asset_dir, queue, stop_event),
        daemon=True,
    )
    prefetcher.start()

    try:
        while True:
            prepared = queue.get()
            if prepared is None:
                break
            if prepared.page_number == 0:
                raise RuntimeError(prepared.error or "PDF 페이지 준비에 실패했습니다.")
            process_prepared_pdf_page(job, prepared, runner, asset_url_prefix)
    finally:
        stop_event.set()
        while not queue.empty():
            queue.get_nowait()
        prefetcher.join(timeout=10)


def process_prepared_pdf_page(
    job: JobState,
    prepared: PreparedPdfPage,
    runner: "OcrRunner",
    asset_url_prefix: str,
) -> None:
    page_number = prepared.page_number
    mode = prepared.mode
    started = time.perf_counter()
    mark_page(job, page_number, status="processing", mode=mode)
    page_base = (page_number - 1) / job.total_pages
    page_weight = 1 / job.total_pages

    def report_page_progress(fraction: float, message: str) -> None:
        overall = (page_base + max(0.0, min(fraction, 1.0)) * page_weight) * 100
        update_job_progress(job, overall, f"{page_number}/{job.total_pages} 페이지 · {message}")

    log_job(
        job.id,
        (
            f"page {page_number}/{job.total_pages}: mode={mode}, "
            f"native_chars={prepared.text_char_count}, "
            f"text_quality={prepared.text_quality}, "
            f"image_coverage={prepared.image_coverage}, "
            f"ocr_regions={len(prepared.region_paths)}"
        ),
    )
    try:
        if prepared.error:
            raise RuntimeError(prepared.error)
        if mode == "ocr":
            report_page_progress(0.12, "스캔 페이지 전체 OCR 준비 중")
            items = runner.run_page(prepared.page_path, page_number, report_page_progress)
            if not items:
                raise RuntimeError("PaddleOCR-VL-1.6 전체 페이지 결과가 비어 있습니다.")
        else:
            report_page_progress(0.12, "원문 텍스트 정리 중")
            items = merge_pdf_region_ocr(
                job,
                page_number,
                prepared.native_items,
                prepared.region_paths,
                runner,
                report_page_progress,
            )
            # 빈 페이지(원문·이미지 모두 없음)는 결과가 없어도 오류가 아니다.
            if not items and (prepared.region_paths or prepared.text_char_count):
                raise RuntimeError("PDF 원문과 OCR 영역에서 추출된 결과가 없습니다.")

        items.append(
            DocumentItem(
                page=page_number,
                type="page_image",
                text=prepared.preview_name,
                markdown=f"![원본 페이지]({asset_url_prefix}/{prepared.preview_name})",
                section_path=[],
                asset_url=f"{asset_url_prefix}/{prepared.preview_name}",
            )
        )
        job.results.extend(items)
        report_page_progress(0.98, "결과 정리 중")
        elapsed = time.perf_counter() - started
        mark_page(job, page_number, status="done", elapsed=elapsed, mode=mode, item_count=len(items))
        update_job_progress(job, page_number / job.total_pages * 100, f"{page_number}/{job.total_pages} 페이지 완료")
        log_job(job.id, f"page {page_number}/{job.total_pages}: {mode} done in {elapsed:.2f}s, items={len(items)}")
    except Exception as exc:
        elapsed = time.perf_counter() - started
        mark_page(job, page_number, status="error", elapsed=elapsed, error=str(exc), mode=mode)
        log_job(job.id, f"page {page_number}/{job.total_pages}: failed after {elapsed:.2f}s: {exc}")


def merge_pdf_region_ocr(
    job: JobState,
    page_number: int,
    native_items: list[DocumentItem],
    region_paths: list[tuple[list[float], Path]],
    runner: "OcrRunner",
    progress_callback: Callable[[float, str], None],
) -> list[DocumentItem]:
    if not region_paths:
        progress_callback(0.9, "원문 텍스트 사용")
        return native_items

    ocr_items: list[DocumentItem] = []
    successful_regions: list[list[float]] = []
    region_count = len(region_paths)
    for index, (region, region_path) in enumerate(region_paths, start=1):

        def report_region_progress(fraction: float, message: str) -> None:
            completed = index - 1
            mapped = 0.16 + ((completed + max(0.0, min(fraction, 1.0))) / region_count) * 0.76
            progress_callback(mapped, f"이미지 영역 {index}/{region_count} · {message}")

        region_items = runner.run_page(region_path, page_number, report_region_progress)
        if not region_items:
            log_job(job.id, f"page {page_number}: OCR region {index}/{region_count} returned no items")
            continue
        translate_ocr_bboxes(region_items, region[0] * 2, region[1] * 2)
        ocr_items.extend(region_items)
        successful_regions.append(region)

    if not successful_regions:
        return native_items

    preserved_native = [
        item for item in native_items if not any(item_overlaps_pdf_region(item, region) for region in successful_regions)
    ]
    return sort_page_items([*preserved_native, *ocr_items])


def item_overlaps_pdf_region(item: DocumentItem, region: list[float]) -> bool:
    if not item.bbox or len(item.bbox) < 4:
        return False
    item_rect = fitz.Rect(bbox_to_rect(item.bbox))
    region_rect = fitz.Rect(*(value * 2 for value in region))
    if item_rect.is_empty:
        return False
    intersection = item_rect & region_rect
    center = fitz.Point((item_rect.x0 + item_rect.x1) / 2, (item_rect.y0 + item_rect.y1) / 2)
    return region_rect.contains(center) or intersection.get_area() / max(item_rect.get_area(), 1) >= 0.5


def bbox_to_rect(bbox: list[float]) -> list[float]:
    if len(bbox) == 4:
        return bbox
    xs = bbox[0::2]
    ys = bbox[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def translate_ocr_bboxes(items: list[DocumentItem], offset_x: float, offset_y: float) -> None:
    for item in items:
        if not item.bbox:
            continue
        item.bbox = [
            round(value + (offset_x if index % 2 == 0 else offset_y), 2)
            for index, value in enumerate(item.bbox)
        ]


def sort_page_items(items: list[DocumentItem]) -> list[DocumentItem]:
    return sorted(
        items,
        key=lambda item: (
            bbox_to_rect(item.bbox)[1] if item.bbox and len(item.bbox) >= 4 else float("inf"),
            bbox_to_rect(item.bbox)[0] if item.bbox and len(item.bbox) >= 4 else float("inf"),
        ),
    )


def enrich_embedded_images_with_ocr(
    job: JobState,
    items: list[DocumentItem],
    runner: "OcrRunner",
    asset_dir: Path,
    progress_start: float,
    progress_end: float,
) -> list[DocumentItem]:
    image_items = [item for item in items if item.type == "image" and item.asset_url]
    if not image_items:
        update_job_progress(job, progress_end, "내부 이미지 없음 · 원문 구조 사용")
        return items

    output = list(items)
    for index, image_item in enumerate(image_items, start=1):
        source = asset_dir / Path(image_item.asset_url or "").name
        ocr_source = prepare_image_for_ocr(source, asset_dir, index)
        if ocr_source is None:
            log_job(job.id, f"embedded image skipped: {source.name}")
            continue

        def report_image_progress(fraction: float, message: str) -> None:
            completed = index - 1
            overall = progress_start + (
                (completed + max(0.0, min(fraction, 1.0))) / len(image_items)
            ) * (progress_end - progress_start)
            update_job_progress(job, overall, f"내부 이미지 {index}/{len(image_items)} · {message}")

        try:
            recognized = runner.run_page(ocr_source, 1, report_image_progress)
        except Exception as exc:
            # 이미지 1장의 OCR 실패(추론 프로세스 중단 포함)가 문서 전체를
            # 실패시키지 않도록 해당 이미지만 건너뛰고 계속 진행한다.
            log_job(job.id, f"embedded image {index}/{len(image_items)} OCR failed: {exc}")
            image_item.description = f"OCR 실패: {exc}"
            continue
        recognized_texts = [item.text for item in recognized if item.text]
        if recognized_texts and not image_item.description:
            image_item.description = ("이미지 내 텍스트: " + " / ".join(recognized_texts))[:300]
        for recognized_item in recognized:
            recognized_item.asset_url = image_item.asset_url
            if not recognized_item.section_path:
                recognized_item.section_path = list(image_item.section_path or [])
        output.extend(recognized)
    return output


def prepare_upscaled_image(source: Path, work_dir: Path) -> Path:
    """긴 변이 기준치보다 작은 이미지를 확대 렌더링해 작은 글씨의 인식률을 높인다."""
    min_edge = int(os.getenv("OCR_MIN_IMAGE_LONG_EDGE", "1600"))
    if min_edge <= 0:
        return source
    try:
        pixmap = fitz.Pixmap(str(source))
        if max(pixmap.width, pixmap.height) >= min_edge:
            return source
        with fitz.open(source) as image_document:
            page = image_document[0]
            zoom = min_edge / max(page.rect.width, page.rect.height)
            upscaled = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        destination = work_dir / "ocr_upscaled.png"
        upscaled.save(destination)
        return destination
    except (RuntimeError, ValueError, OSError):
        return source


def prepare_image_for_ocr(source: Path, asset_dir: Path, index: int) -> Path | None:
    if not source.exists():
        return None
    if source.suffix.lower() in IMAGE_EXTENSIONS:
        return source
    try:
        converted = asset_dir / f"ocr_image_{index:03d}.png"
        with fitz.open(source) as image_document:
            pixmap = image_document[0].get_pixmap(alpha=False)
            pixmap.save(converted)
        return converted
    except (RuntimeError, ValueError, OSError):
        return None


def process_image(
    job: JobState,
    document_path: Path,
    runner: "OcrRunner",
    asset_dir: Path,
    asset_url_prefix: str,
) -> None:
    started = time.perf_counter()
    mark_page(job, 1, status="processing", mode="ocr")
    update_job_progress(job, 8, "이미지 OCR 준비 중")
    try:
        source_name = f"source{document_path.suffix.lower()}"
        source_asset = asset_dir / source_name
        shutil.copy2(document_path, source_asset)

        def report_progress(fraction: float, message: str) -> None:
            update_job_progress(job, 8 + max(0.0, min(fraction, 1.0)) * 86, message)

        ocr_source = prepare_upscaled_image(source_asset, asset_dir)
        items = runner.run_page(ocr_source, 1, report_progress)
        if not items:
            raise RuntimeError("PaddleOCR-VL-1.6 이미지 결과가 비어 있습니다.")
        asset_url = f"{asset_url_prefix}/{source_name}"
        for item in items:
            item.asset_url = asset_url
        items.append(
            DocumentItem(
                page=1,
                type="page_image",
                text=source_name,
                markdown=f"![원본 이미지]({asset_url})",
                section_path=[],
                asset_url=asset_url,
            )
        )
        job.results.extend(items)
        update_job_progress(job, 98, "이미지 OCR 결과 정리 중")
        mark_page(
            job,
            1,
            status="done",
            elapsed=time.perf_counter() - started,
            mode="ocr",
            item_count=len(items),
        )
    except Exception as exc:
        mark_page(
            job,
            1,
            status="error",
            elapsed=time.perf_counter() - started,
            error=str(exc),
            mode="ocr",
        )


def mark_page(
    job: JobState,
    page_number: int,
    status: Literal["waiting", "processing", "done", "error"],
    elapsed: float | None = None,
    error: str | None = None,
    mode: str | None = None,
    item_count: int = 0,
) -> None:
    page = job.pages[page_number - 1]
    page.status = status
    page.elapsed = round(elapsed, 2) if elapsed is not None else page.elapsed
    page.error = error
    page.mode = mode or page.mode
    page.item_count = item_count
    job.updated_at = time.time()
    persist_job(job)


def update_job_progress(job: JobState, progress: float, message: str) -> None:
    job.progress = round(max(job.progress, min(max(progress, 0.0), 100.0)), 1)
    job.progress_message = message
    job.updated_at = time.time()
    persist_job(job)


class OcrRunner:
    def __init__(self) -> None:
        self.mock = os.getenv("OCR_MOCK", "").lower() in {"1", "true", "yes"}

    def run_page(
        self,
        page_path: Path,
        page_number: int,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[DocumentItem]:
        if self.mock:
            if progress_callback:
                progress_callback(0.3, "모의 OCR 추론 중")
            time.sleep(0.8)
            if progress_callback:
                progress_callback(0.9, "모의 OCR 결과 생성 중")
            return [
                DocumentItem(
                    page=page_number,
                    type="heading",
                    text=f"{page_number}. OCR 테스트 제목",
                    markdown=f"## {page_number}. OCR 테스트 제목",
                    section_path=[f"{page_number}. OCR 테스트 제목"],
                    source_text=f"{page_number}. OCR 테스트 제목",
                ),
                DocumentItem(
                    page=page_number,
                    type="paragraph",
                    text="모델 설치 전 UI 확인용 모의 결과입니다.",
                    markdown="모델 설치 전 UI 확인용 모의 결과입니다.",
                    section_path=[f"{page_number}. OCR 테스트 제목"],
                    source_text="모델 설치 전 UI 확인용 모의 결과입니다.",
                ),
            ]

        return self._run_paddleocr_vl_1_6(page_path, page_number, progress_callback)

    def _run_paddleocr_vl_1_6(
        self,
        page_path: Path,
        page_number: int,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> list[DocumentItem]:
        use_subprocess = os.getenv("OCR_SUBPROCESS", "1").lower() in {"1", "true", "yes"}
        raw_items: list[Any] = []
        with PADDLE_INFERENCE_LOCK:
            if progress_callback:
                progress_callback(0.18, "모델 로딩 중")
            with inference_progress_pulse(progress_callback):
                if progress_callback:
                    progress_callback(0.3, "OCR 추론 중 (예상 진행률)")
                if use_subprocess:
                    raw_items = run_paddle_in_worker(page_path)
                else:
                    pipeline = get_paddle_pipeline()
                    output = pipeline.predict(input=str(page_path))
                    for result in output:
                        raw_items.extend(extract_paddle_blocks(result))
            if progress_callback:
                progress_callback(0.92, "OCR 결과 변환 중")
        return normalize_blocks(raw_items, page_number)


class PaddleWorkerCrash(RuntimeError):
    pass


class PaddleWorkerClient:
    """Paddle 추론을 자식 프로세스에서 실행하는 클라이언트.

    libpaddle 세그폴트가 나도 서버는 살아남는다. 워커는 모델을 1회
    로딩해 재사용하며, 죽으면 다음 요청 때 자동으로 재기동된다.
    호출자는 PADDLE_INFERENCE_LOCK으로 직렬화되어 있다.
    """

    def __init__(self, worker_target: Callable[..., None] = paddle_worker.run_worker) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._process: Any | None = None
        self._request_queue: Any | None = None
        self._response_queue: Any | None = None
        self._request_id = 0

    def run(self, page_path: str, timeout_seconds: float) -> list[dict[str, Any]]:
        self._ensure_worker()
        self._request_id += 1
        request_id = self._request_id
        self._request_queue.put((request_id, page_path))
        deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
        while True:
            try:
                response_id, status, payload = self._response_queue.get(timeout=1.0)
            except Empty:
                if self._process is None or not self._process.is_alive():
                    exit_code = getattr(self._process, "exitcode", None)
                    self._reset()
                    raise PaddleWorkerCrash(f"OCR 추론 프로세스가 비정상 종료되었습니다 (exit code {exit_code}).")
                if deadline is not None and time.monotonic() > deadline:
                    self.terminate()
                    raise RuntimeError(f"OCR 추론이 제한 시간({timeout_seconds:.0f}초)을 초과해 중단했습니다.")
                continue
            if response_id != request_id:
                continue
            if status == "ok":
                return payload
            raise RuntimeError(payload)

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._reset()
        self._request_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self._process = self._context.Process(
            target=self._worker_target,
            args=(self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()

    def terminate(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._reset()

    def _reset(self) -> None:
        self._process = None
        self._request_queue = None
        self._response_queue = None


PADDLE_WORKER_CLIENT = PaddleWorkerClient()


def run_paddle_in_worker(page_path: Path) -> list[dict[str, Any]]:
    timeout_seconds = float(os.getenv("OCR_INFERENCE_TIMEOUT_SECONDS", "900"))
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            return PADDLE_WORKER_CLIENT.run(str(page_path), timeout_seconds)
        except PaddleWorkerCrash as exc:
            log_job("worker", f"paddle worker crashed (attempt {attempt}/{attempts}): {exc}")
            if attempt == attempts:
                raise RuntimeError(
                    f"OCR 추론 프로세스가 반복적으로 중단되었습니다. 해당 페이지를 건너뜁니다. ({exc})"
                ) from exc
    raise RuntimeError("OCR 추론에 실패했습니다.")


@contextmanager
def inference_progress_pulse(
    progress_callback: Callable[[float, str], None] | None,
):
    if progress_callback is None:
        yield
        return

    stop_event = threading.Event()
    expected_seconds = max(float(os.getenv("OCR_EXPECTED_PAGE_SECONDS", "120")), 10.0)
    started = time.perf_counter()

    def pulse() -> None:
        while not stop_event.wait(1.0):
            elapsed = time.perf_counter() - started
            fraction = min(0.18 + (elapsed / expected_seconds) * 0.7, 0.88)
            progress_callback(fraction, f"OCR 추론 중 · {elapsed:.0f}초 경과 (예상 진행률)")

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=2)

def get_paddle_pipeline(job_id: str | None = None) -> Any:
    """OCR_SUBPROCESS=0일 때 사용하는 서버 내 파이프라인 (개발·디버깅용)."""
    global PADDLE_PIPELINE
    if PADDLE_PIPELINE is None:
        with PADDLE_PIPELINE_LOCK:
            if PADDLE_PIPELINE is None:
                if job_id:
                    log_job(job_id, "creating PaddleOCRVL object")
                PADDLE_PIPELINE = paddle_worker.create_paddle_pipeline()
                if job_id:
                    log_job(job_id, "PaddleOCRVL object created")
    elif job_id:
        log_job(job_id, "using cached PaddleOCRVL object")
    return PADDLE_PIPELINE


def infer_page_number(result: Any, fallback: int) -> int:
    data = result_to_plain_data(result)
    if isinstance(data, dict):
        for key in ("page", "page_id", "page_index", "page_no", "page_num"):
            value = data.get(key)
            if isinstance(value, int):
                return value + 1 if key == "page_index" else value
            if isinstance(value, str) and value.isdigit():
                number = int(value)
                return number + 1 if key == "page_index" else number
    return fallback


def normalize_blocks(blocks: list[Any], page_number: int) -> list[DocumentItem]:
    normalized: list[DocumentItem] = []
    seen: set[tuple[str, str, tuple[float, ...] | None]] = set()
    section_path: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = clean_text(str(block.get("text", "")))
        if not text:
            continue
        item_type = normalize_type(str(block.get("type", "")), text)
        bbox = normalize_bbox(block.get("bbox"))
        # bbox까지 키에 포함해, 다이어그램에서 같은 라벨이 다른 위치에
        # 반복되는 경우(예: 동일 항목명 박스 2개)를 유실하지 않는다.
        key = (item_type, text, tuple(bbox) if bbox else None)
        if key in seen:
            continue
        seen.add(key)
        if item_type == "heading":
            section_path = [text]
        raw_text = str(block.get("text", "")).strip()
        item = DocumentItem(
            page=page_number,
            type=item_type,
            text=text,
            markdown=f"## {raw_text}" if item_type == "heading" else raw_text,
            section_path=list(section_path),
            source_text=raw_text,
            bbox=bbox,
            metrics=extract_metrics(text),
        )
        if item_type == "chart":
            item.chart = parse_chart_table(raw_text)
            item.description = describe_chart(item.chart, text)
        elif item_type in {"image", "figure", "seal"}:
            item.description = f"이미지 인식 내용: {text[:300]}"
        normalized.append(item)
    return normalized


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    flattened: list[float] = []
    for item in value:
        if isinstance(item, (list, tuple)):
            flattened.extend(float(number) for number in item if isinstance(number, (int, float)))
        elif isinstance(item, (int, float)):
            flattened.append(float(item))
    return [round(number, 2) for number in flattened] or None


def normalize_type(raw_type: str, text: str) -> str:
    label = raw_type.lower().strip()
    if label in {"doc_title", "title", "paragraph_title", "heading", "header", "section_header"}:
        return "heading"
    if label in {"page_number", "page_num"}:
        return "footer"
    if label in {"table", "formula", "image", "figure", "caption", "footer", "footnote", "seal", "chart"}:
        return label
    if looks_like_heading(text):
        return "heading"
    if label in {"text", "paragraph", "reference"}:
        return "paragraph"
    return "paragraph"


def looks_like_heading(text: str) -> bool:
    compact = text.strip()
    if len(compact) > 80:
        return False
    # "1.5%" 같은 소수점 수치를 제목으로 오분류하지 않도록 "1." 뒤에 숫자가 오면 제외한다.
    if re.match(r"^\d+\.(?!\d)", compact):
        return True
    if re.match(r"^제\s*\d+", compact):
        return True
    return compact.startswith(("○", "□", "■", "●", "Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ"))


def clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", "").split())
