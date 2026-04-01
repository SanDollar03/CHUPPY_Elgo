const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const selectedFile = document.getElementById("selectedFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const progressBar = document.getElementById("progressBar");
const progressMessage = document.getElementById("progressMessage");
const resultSection = document.getElementById("resultSection");
const summaryText = document.getElementById("summaryText");
const downloadVideo = document.getElementById("downloadVideo");
const downloadSummaryText = document.getElementById("downloadSummaryText");
const downloadSummaryJson = document.getElementById("downloadSummaryJson");
const downloadVtt = document.getElementById("downloadVtt");

let selected = null;
let pollingTimer = null;

function setProgress(value, message = "") {
    const safe = Math.max(0, Math.min(100, Number(value) || 0));
    progressBar.style.width = `${safe}%`;
    progressBar.textContent = `${safe}%`;
    progressMessage.textContent = message || "";
}

function enableAnalyze() {
    analyzeBtn.disabled = !selected;
}

function setSelectedFile(file) {
    selected = file;
    selectedFile.textContent = file ? `選択中: ${file.name}` : "未選択";
    enableAnalyze();
}

dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        setSelectedFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
        setSelectedFile(e.target.files[0]);
    }
});

analyzeBtn.addEventListener("click", async () => {
    if (!selected) return;

    resultSection.classList.add("hidden");
    summaryText.textContent = "";
    setProgress(0, "アップロード中...");

    const formData = new FormData();
    formData.append("file", selected);

    try {
        const res = await fetch("/api/upload", {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (!res.ok || !data.ok) {
            throw new Error(data.error || "アップロードに失敗しました");
        }

        pollStatus(data.job_id);
    } catch (err) {
        setProgress(0, `エラー: ${err.message}`);
    }
});

function pollStatus(jobId) {
    if (pollingTimer) {
        clearInterval(pollingTimer);
    }

    pollingTimer = setInterval(async () => {
        try {
            const res = await fetch(`/api/jobs/${jobId}/status`);
            const data = await res.json();

            if (!res.ok || !data.ok) {
                throw new Error(data.error || "状態取得に失敗しました");
            }

            setProgress(data.progress, data.message);

            if (data.status === "completed") {
                clearInterval(pollingTimer);
                renderResult(data.result);
            } else if (data.status === "failed") {
                clearInterval(pollingTimer);
                setProgress(100, `失敗: ${data.error || data.message}`);
            }
        } catch (err) {
            clearInterval(pollingTimer);
            setProgress(100, `失敗: ${err.message}`);
        }
    }, 1000);
}

function renderResult(result) {
    if (!result) return;

    resultSection.classList.remove("hidden");
    summaryText.textContent = result.summary_text || "";

    downloadVideo.href = result.download_video_url;
    downloadSummaryText.href = result.report_txt_url;
    downloadSummaryJson.href = result.summary_json_url;
    downloadVtt.href = result.download_vtt_url;
}