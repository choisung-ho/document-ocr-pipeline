const form = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#document, #pdf");
const fileLabel = document.querySelector("#fileLabel");
const submitBtn = document.querySelector("#submitBtn");
const jobStatus = document.querySelector("#jobStatus");
const pagesEl = document.querySelector("#pages");
const pageCount = document.querySelector("#pageCount");
const resultJson = document.querySelector("#resultJson");
const copyBtn = document.querySelector("#copyBtn");
const progressMessage = document.querySelector("#progressMessage");
const progressPercent = document.querySelector("#progressPercent");
const progressFill = document.querySelector("#progressFill");
const progressTrack = document.querySelector(".progress-track");

let events = null;

if (!form || !fileInput || !submitBtn) {
  throw new Error("문서 업로드 UI 초기화에 실패했습니다. 브라우저를 새로고침하세요.");
}

fileInput.addEventListener("change", () => {
  fileLabel.textContent = fileInput.files[0]?.name || "HWPX·DOCX·PDF·이미지 파일 선택";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) return;

  closeEvents();
  setBusy(true);
  setStatus("running", "업로드 중");
  setProgress(0, "문서 업로드 중");
  resultJson.textContent = "[]";
  copyBtn.disabled = true;
  pagesEl.className = "pages empty";
  pagesEl.textContent = "업로드 중입니다.";

  const body = new FormData(form);
  try {
    const response = await fetch("/api/jobs", { method: "POST", body });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "업로드 실패");
    watchJob(payload.job_id);
  } catch (error) {
    setBusy(false);
    setStatus("error", "오류");
    pagesEl.textContent = error.message;
  }
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(resultJson.textContent);
  copyBtn.textContent = "복사됨";
  setTimeout(() => {
    copyBtn.textContent = "복사";
  }, 1200);
});

function watchJob(jobId) {
  events = new EventSource(`/api/jobs/${jobId}/events`);
  events.onmessage = (message) => {
    const job = JSON.parse(message.data);
    renderJob(job);
    if (job.status === "done" || job.status === "error") {
      closeEvents();
      setBusy(false);
    }
  };
  events.onerror = () => {
    closeEvents();
    setBusy(false);
    setStatus("error", "연결 오류");
  };
}

function renderJob(job) {
  const statusText = {
    queued: "대기",
    running: "처리 중",
    done: "완료",
    error: "오류",
  }[job.status];
  setStatus(job.status, statusText);
  setProgress(job.progress ?? 0, job.progress_message || statusText);
  pageCount.textContent = `${job.total_pages}개 처리 단위`;
  renderPages(job.pages);
  resultJson.textContent = JSON.stringify({ metadata: job.metadata, results: job.results }, null, 2);
  copyBtn.disabled = job.results.length === 0;
}

function renderPages(pages) {
  pagesEl.className = "pages";
  pagesEl.innerHTML = "";
  pages.forEach((page) => {
    const row = document.createElement("div");
    row.className = `page-row ${page.status}`;

    const left = document.createElement("div");
    left.className = "page-left";
    const mode = page.mode ? ` · ${page.mode}` : "";
    left.innerHTML = `<strong>처리 단위 ${page.page}</strong><span>${page.status}${mode} · ${page.item_count}개 결과</span>`;

    const right = document.createElement("div");
    right.className = "page-time";
    if (page.status === "processing") {
      right.textContent = "진행 중";
    } else if (page.elapsed !== null) {
      right.textContent = `${page.elapsed.toFixed(2)}초`;
    } else {
      right.textContent = "-";
    }

    row.append(left, right);
    if (page.error) {
      const error = document.createElement("div");
      error.className = "page-error";
      error.textContent = page.error;
      row.append(error);
    }
    pagesEl.append(row);
  });
}

function setStatus(kind, text) {
  jobStatus.className = `status ${kind}`;
  jobStatus.textContent = text;
}

function setBusy(isBusy) {
  submitBtn.disabled = isBusy;
  submitBtn.textContent = isBusy ? "처리 중" : "처리 시작";
}

function setProgress(value, message) {
  const normalized = Math.max(0, Math.min(Number(value) || 0, 100));
  const rounded = normalized.toFixed(1).replace(".0", "");
  progressPercent.textContent = `${rounded}%`;
  progressMessage.textContent = message;
  progressFill.style.width = `${normalized}%`;
  progressTrack.setAttribute("aria-valuenow", String(normalized));
}

function closeEvents() {
  if (events) {
    events.close();
    events = null;
  }
}
