const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const selectedFile = document.getElementById("selectedFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const progressBar = document.getElementById("progressBar");
const progressMessage = document.getElementById("progressMessage");
const uploadSection = document.getElementById("uploadSection");
const progressSection = document.getElementById("progressSection");
const resultSection = document.getElementById("resultSection");
const resultVideo = document.getElementById("resultVideo");

const summaryStats = document.getElementById("summaryStats");
const reportCards = document.getElementById("reportCards");
const graphsWrap = document.getElementById("graphsWrap");

const downloadVideo = document.getElementById("downloadVideo");
const downloadSummaryText = document.getElementById("downloadSummaryText");
const downloadSummaryJson = document.getElementById("downloadSummaryJson");
const downloadVtt = document.getElementById("downloadVtt");

let selected = null;
let pollingTimer = null;

const LABEL_META = {
    bending: { ja: "前屈姿勢", statClass: "stat-bending", cardClass: "risk-bending", color: "#f97316" },
    overhead_work: { ja: "頭上作業", statClass: "stat-overhead_work", cardClass: "risk-overhead_work", color: "#8b5cf6" },
    squatting: { ja: "しゃがみ", statClass: "stat-squatting", cardClass: "risk-squatting", color: "#06b6d4" },
    one_arm_overextension: { ja: "片腕過伸展", statClass: "stat-one_arm_overextension", cardClass: "risk-one_arm_overextension", color: "#ef4444" },
    same_posture_continuation: { ja: "同一姿勢継続", statClass: "stat-same_posture_continuation", cardClass: "risk-same_posture_continuation", color: "#eab308" },
};

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

function showResultLayout() {
    uploadSection.classList.add("hidden");
    progressSection.classList.add("hidden");
    resultSection.classList.remove("hidden");
}

function resetResultVideo() {
    resultVideo.pause();
    resultVideo.removeAttribute("src");
    resultVideo.load();
}

async function playResultVideo() {
    try {
        await resultVideo.play();
    } catch (err) {
        console.warn("自動再生に失敗しました:", err);
    }
}

async function jumpToVideo(sec) {
    try {
        resultVideo.currentTime = Number(sec || 0);
        await resultVideo.play();
    } catch (err) {
        console.warn("動画ジャンプに失敗しました:", err);
    }
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function formatSec(sec) {
    return `${Number(sec || 0).toFixed(2)}秒`;
}

function formatShortSec(sec) {
    return `${Number(sec || 0).toFixed(1)}s`;
}

function clearResultViews() {
    summaryStats.innerHTML = "";
    reportCards.innerHTML = "";
    graphsWrap.innerHTML = "";
}

function totalDurationFromSummary(summary) {
    const durations = [];
    for (const seg of (summary.event_segments || [])) {
        durations.push(Number(seg.end_sec || 0));
    }
    for (const seg of (summary.same_posture_segments || [])) {
        durations.push(Number(seg.end_sec || 0));
    }
    return Math.max(1, ...durations, 1);
}

function renderSummaryStats(summary) {
    const stats = summary.aggregated_stats || {};
    const order = [
        { key: "前屈姿勢", label: "bending" },
        { key: "頭上作業", label: "overhead_work" },
        { key: "しゃがみ", label: "squatting" },
        { key: "片腕過伸展", label: "one_arm_overextension" },
        { key: "同一姿勢継続", label: "same_posture_continuation" },
    ];

    summaryStats.innerHTML = order.map(({ key, label }) => {
        const item = stats[key] || { count: 0, total_duration_sec: 0 };
        const meta = LABEL_META[label];
        return `
            <div class="summary-stat ${meta.statClass}">
                <div class="summary-stat-label">${escapeHtml(key)}</div>
                <div class="summary-stat-value">${escapeHtml(item.count)}回</div>
                <div class="summary-stat-sub">合計 ${escapeHtml(Number(item.total_duration_sec || 0).toFixed(2))}秒</div>
            </div>
        `;
    }).join("");
}

function buildCompactCards(summary) {
    const cards = [];

    for (const seg of (summary.event_segments || [])) {
        cards.push({
            label: seg.label,
            labelJa: seg.label_ja,
            startSec: Number(seg.start_sec || 0),
            durationSec: Number(seg.duration_sec || 0),
            desc: seg.label === "bending" && seg.max_trunk_angle != null
                ? `最大前傾角 ${Number(seg.max_trunk_angle).toFixed(1)}度`
                : `継続 ${Number(seg.duration_sec || 0).toFixed(2)}秒`
        });
    }

    for (const seg of (summary.same_posture_segments || [])) {
        cards.push({
            label: "same_posture_continuation",
            labelJa: seg.label_ja || "同一姿勢継続",
            startSec: Number(seg.start_sec || 0),
            durationSec: Number(seg.duration_sec || 0),
            desc: `継続 ${Number(seg.duration_sec || 0).toFixed(2)}秒`
        });
    }

    cards.sort((a, b) => a.startSec - b.startSec);
    return cards;
}

function renderReportCards(cards) {
    reportCards.innerHTML = "";

    if (!cards.length) {
        reportCards.innerHTML = `
            <button type="button" class="report-card">
                <div class="report-card-row">
                    <div class="report-card-title">検出結果なし</div>
                    <div class="report-card-desc">エルゴ負荷は検出されませんでした</div>
                    <div class="report-card-action">-</div>
                </div>
            </button>
        `;
        return;
    }

    for (const card of cards) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `report-card ${LABEL_META[card.label]?.cardClass || ""}`;
        btn.innerHTML = `
            <div class="report-card-row">
                <div class="report-card-title">${escapeHtml(card.labelJa)}</div>
                <div class="report-card-desc">${escapeHtml(card.desc)}</div>
                <div class="report-card-action">ジャンプ</div>
            </div>
        `;
        btn.addEventListener("click", () => jumpToVideo(card.startSec));
        reportCards.appendChild(btn);
    }
}

function svgWrap(title, note, inner) {
    return `
        <div class="graph-card">
            <div class="graph-card-header">
                <div class="graph-card-title">${escapeHtml(title)}</div>
                <div class="graph-card-note">${escapeHtml(note)}</div>
            </div>
            ${inner}
        </div>
    `;
}

function createXAxis(duration, width, height, padding) {
    const y = height - padding.bottom;
    const ticks = 5;
    let out = `
        <line class="axis-line" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>
    `;
    for (let i = 0; i <= ticks; i++) {
        const ratio = i / ticks;
        const x = padding.left + ratio * (width - padding.left - padding.right);
        const sec = duration * ratio;
        out += `
            <line class="grid-line" x1="${x}" y1="${padding.top}" x2="${x}" y2="${y}"></line>
            <text class="axis-label" x="${x}" y="${y + 14}" text-anchor="middle">${formatShortSec(sec)}</text>
        `;
    }
    return out;
}

function renderLineGraphSvg(title, color, points, duration, yMax, yLabel) {
    const width = 760;
    const height = 150;
    const padding = { top: 16, right: 16, bottom: 26, left: 42 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const xOf = (sec) => padding.left + (Number(sec || 0) / duration) * chartW;
    const yOf = (val) => padding.top + chartH - (Number(val || 0) / yMax) * chartH;

    let grid = "";
    const yTicks = 4;
    for (let i = 0; i <= yTicks; i++) {
        const ratio = i / yTicks;
        const y = padding.top + ratio * chartH;
        const label = (yMax - ratio * yMax).toFixed(0);
        grid += `
            <line class="grid-line" x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}"></line>
            <text class="axis-label" x="${padding.left - 6}" y="${y + 3}" text-anchor="end">${label}</text>
        `;
    }

    const path = points.map((p, idx) => `${idx === 0 ? "M" : "L"} ${xOf(p.sec)} ${yOf(p.value)}`).join(" ");

    const dots = points.map((p) => `
        <circle class="point-dot graph-jump"
            cx="${xOf(p.sec)}"
            cy="${yOf(p.value)}"
            r="4.5"
            fill="${color}"
            data-jump-sec="${p.jumpSec}">
            <title>${escapeHtml(formatSec(p.jumpSec))} / ${escapeHtml(yLabel)} ${escapeHtml(Number(p.value).toFixed(1))}</title>
        </circle>
    `).join("");

    const svg = `
        <svg class="graph-svg" viewBox="0 0 ${width} ${height}">
            ${grid}
            ${createXAxis(duration, width, height, padding)}
            <text class="axis-label" x="${padding.left}" y="12">${escapeHtml(yLabel)}</text>
            <path class="line-path" d="${path}" stroke="${color}"></path>
            ${dots}
        </svg>
    `;

    return svgWrap(title, "ポイントクリックで動画ジャンプ", svg);
}

function renderBandGraphSvg(title, color, segments, duration) {
    const width = 760;
    const height = 110;
    const padding = { top: 18, right: 16, bottom: 26, left: 18 };
    const chartW = width - padding.left - padding.right;
    const bandY = 40;
    const bandH = 28;

    const xOf = (sec) => padding.left + (Number(sec || 0) / duration) * chartW;

    const rects = segments.map((seg) => {
        const x = xOf(seg.start_sec);
        const w = Math.max(6, xOf(seg.end_sec) - x);
        return `
            <rect class="band-rect graph-jump"
                x="${x}"
                y="${bandY}"
                width="${w}"
                height="${bandH}"
                rx="6"
                fill="${color}"
                fill-opacity="0.75"
                data-jump-sec="${seg.start_sec}">
                <title>${escapeHtml(formatSec(seg.start_sec))} - ${escapeHtml(formatSec(seg.end_sec))}</title>
            </rect>
        `;
    }).join("");

    const svg = `
        <svg class="graph-svg" viewBox="0 0 ${width} ${height}">
            <line class="axis-line" x1="${padding.left}" y1="${bandY + bandH / 2}" x2="${width - padding.right}" y2="${bandY + bandH / 2}"></line>
            ${createXAxis(duration, width, height, padding)}
            ${rects}
        </svg>
    `;

    return svgWrap(title, "帯クリックで動画ジャンプ", svg);
}

function attachGraphJumpEvents() {
    graphsWrap.querySelectorAll(".graph-jump").forEach((el) => {
        el.addEventListener("click", () => {
            const sec = Number(el.getAttribute("data-jump-sec") || 0);
            jumpToVideo(sec);
        });
    });
}

function renderGraphs(summary) {
    const duration = totalDurationFromSummary(summary);

    const bendingSegments = (summary.event_segments || []).filter((s) => s.label === "bending");
    const overheadSegments = (summary.event_segments || []).filter((s) => s.label === "overhead_work");
    const squattingSegments = (summary.event_segments || []).filter((s) => s.label === "squatting");
    const armSegments = (summary.event_segments || []).filter((s) => s.label === "one_arm_overextension");
    const samePostureSegments = summary.same_posture_segments || [];

    const htmlParts = [];

    if (bendingSegments.length) {
        const points = [];
        for (const seg of bendingSegments) {
            const angle = Number(seg.max_trunk_angle || 0);
            points.push({ sec: Number(seg.start_sec || 0), value: angle, jumpSec: Number(seg.start_sec || 0) });
            points.push({ sec: Number(seg.end_sec || 0), value: angle, jumpSec: Number(seg.start_sec || 0) });
        }
        points.sort((a, b) => a.sec - b.sec);
        const yMax = Math.max(50, ...points.map((p) => p.value), 50);
        htmlParts.push(
            renderLineGraphSvg("前屈姿勢（前傾角）", LABEL_META.bending.color, points, duration, yMax, "角度")
        );
    }

    if (overheadSegments.length) {
        htmlParts.push(
            renderBandGraphSvg("頭上作業（発生帯）", LABEL_META.overhead_work.color, overheadSegments, duration)
        );
    }

    if (squattingSegments.length) {
        htmlParts.push(
            renderBandGraphSvg("しゃがみ（発生帯）", LABEL_META.squatting.color, squattingSegments, duration)
        );
    }

    if (armSegments.length) {
        htmlParts.push(
            renderBandGraphSvg("片腕過伸展（発生帯）", LABEL_META.one_arm_overextension.color, armSegments, duration)
        );
    }

    if (samePostureSegments.length) {
        htmlParts.push(
            renderBandGraphSvg("同一姿勢継続（発生帯）", LABEL_META.same_posture_continuation.color, samePostureSegments, duration)
        );
    }

    if (!htmlParts.length) {
        htmlParts.push(svgWrap("分析グラフ", "データなし", `<div class="graph-card-note">表示できるイベントがありません。</div>`));
    }

    graphsWrap.innerHTML = htmlParts.join("");
    attachGraphJumpEvents();
}

async function fetchSummaryJson(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
        throw new Error("summary json の取得に失敗しました");
    }
    return await res.json();
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

    if (pollingTimer) {
        clearInterval(pollingTimer);
    }

    resultSection.classList.add("hidden");
    uploadSection.classList.remove("hidden");
    progressSection.classList.remove("hidden");

    clearResultViews();
    resetResultVideo();
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
                pollingTimer = null;
                await renderResult(data.result);
            } else if (data.status === "failed") {
                clearInterval(pollingTimer);
                pollingTimer = null;
                setProgress(100, `失敗: ${data.error || data.message}`);
            }
        } catch (err) {
            clearInterval(pollingTimer);
            pollingTimer = null;
            setProgress(100, `失敗: ${err.message}`);
        }
    }, 1000);
}

async function renderResult(result) {
    if (!result) return;

    downloadVideo.href = result.download_video_url || "#";
    downloadSummaryText.href = result.report_txt_url || "#";
    downloadSummaryJson.href = result.summary_json_url || "#";
    downloadVtt.href = result.download_vtt_url || "#";

    if (result.overlay_video_url) {
        resultVideo.src = result.overlay_video_url;
        resultVideo.load();
    }

    let summary = null;
    try {
        summary = await fetchSummaryJson(result.summary_json_url);
    } catch (err) {
        console.error(err);
    }

    if (summary) {
        renderSummaryStats(summary);
        renderReportCards(buildCompactCards(summary));
        renderGraphs(summary);
    }

    showResultLayout();
    playResultVideo();
}