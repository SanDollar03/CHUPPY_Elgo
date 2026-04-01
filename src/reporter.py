from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


LABEL_TO_JA = {
    "bending": "前屈姿勢",
    "overhead_work": "頭上作業",
    "squatting": "しゃがみ",
    "one_arm_overextension": "片腕過伸展",
    "same_posture_continuation": "同一姿勢継続",
}


def save_frame_results_csv(rows: list[dict], output_csv_path: str | Path) -> Path:
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return output_csv_path


def summarize_segments(rows: list[dict], min_duration_sec: float = 1.0) -> list[dict]:
    if not rows:
        return []

    target_labels = {"bending", "overhead_work", "squatting", "one_arm_overextension"}
    segments: list[dict] = []
    current_label = None
    start_idx = None

    for i, row in enumerate(rows):
        label = row["label"]

        if label not in target_labels:
            if current_label is not None:
                segment = _make_segment(rows, start_idx, i - 1, current_label)
                if segment["duration_sec"] >= min_duration_sec:
                    segments.append(segment)
                current_label = None
                start_idx = None
            continue

        if current_label is None:
            current_label = label
            start_idx = i
            continue

        if label != current_label:
            segment = _make_segment(rows, start_idx, i - 1, current_label)
            if segment["duration_sec"] >= min_duration_sec:
                segments.append(segment)
            current_label = label
            start_idx = i

    if current_label is not None and start_idx is not None:
        segment = _make_segment(rows, start_idx, len(rows) - 1, current_label)
        if segment["duration_sec"] >= min_duration_sec:
            segments.append(segment)

    return segments


def summarize_same_posture_segments(rows: list[dict], min_duration_sec: float = 3.0) -> list[dict]:
    if not rows:
        return []

    segments: list[dict] = []
    start_idx = None
    current_signature = None

    for i, row in enumerate(rows):
        if not row.get("visible", False):
            if current_signature is not None and start_idx is not None:
                segment = _make_same_posture_segment(rows, start_idx, i - 1, current_signature)
                if segment["duration_sec"] >= min_duration_sec:
                    segments.append(segment)
                current_signature = None
                start_idx = None
            continue

        sig = row.get("posture_signature", "unknown")

        if current_signature is None:
            current_signature = sig
            start_idx = i
            continue

        if sig != current_signature:
            segment = _make_same_posture_segment(rows, start_idx, i - 1, current_signature)
            if segment["duration_sec"] >= min_duration_sec:
                segments.append(segment)
            current_signature = sig
            start_idx = i

    if current_signature is not None and start_idx is not None:
        segment = _make_same_posture_segment(rows, start_idx, len(rows) - 1, current_signature)
        if segment["duration_sec"] >= min_duration_sec:
            segments.append(segment)

    return segments


def _make_segment(rows: list[dict], start_idx: int, end_idx: int, label: str) -> dict:
    start_row = rows[start_idx]
    end_row = rows[end_idx]

    trunk_values = [
        r.get("trunk_angle")
        for r in rows[start_idx:end_idx + 1]
        if r.get("trunk_angle") is not None
    ]
    max_trunk = max(trunk_values) if trunk_values else None

    risk_values = [r.get("risk_score", 0) for r in rows[start_idx:end_idx + 1]]
    max_risk = max(risk_values) if risk_values else 0

    return {
        "label": label,
        "label_ja": LABEL_TO_JA.get(label, label),
        "start_frame": int(start_row["frame"]),
        "end_frame": int(end_row["frame"]),
        "start_sec": float(start_row["time_sec"]),
        "end_sec": float(end_row["time_sec"]),
        "duration_sec": max(0.0, float(end_row["time_sec"]) - float(start_row["time_sec"])),
        "max_risk_score": int(max_risk),
        "max_trunk_angle": None if max_trunk is None else float(max_trunk),
    }


def _make_same_posture_segment(rows: list[dict], start_idx: int, end_idx: int, posture_signature: str) -> dict:
    start_row = rows[start_idx]
    end_row = rows[end_idx]

    return {
        "label": "same_posture_continuation",
        "label_ja": LABEL_TO_JA["same_posture_continuation"],
        "posture_signature": posture_signature,
        "start_frame": int(start_row["frame"]),
        "end_frame": int(end_row["frame"]),
        "start_sec": float(start_row["time_sec"]),
        "end_sec": float(end_row["time_sec"]),
        "duration_sec": max(0.0, float(end_row["time_sec"]) - float(start_row["time_sec"])),
        "max_risk_score": int(max(r.get("risk_score", 0) for r in rows[start_idx:end_idx + 1])),
    }


def build_aggregated_stats(event_segments: list[dict], same_posture_segments: list[dict]) -> dict:
    stats = {
        "前屈姿勢": {"count": 0, "total_duration_sec": 0.0},
        "頭上作業": {"count": 0, "total_duration_sec": 0.0},
        "しゃがみ": {"count": 0, "total_duration_sec": 0.0},
        "片腕過伸展": {"count": 0, "total_duration_sec": 0.0},
        "同一姿勢継続": {"count": 0, "total_duration_sec": 0.0},
    }

    for seg in event_segments:
        label_ja = seg["label_ja"]
        if label_ja in stats:
            stats[label_ja]["count"] += 1
            stats[label_ja]["total_duration_sec"] += float(seg["duration_sec"])

    for seg in same_posture_segments:
        stats["同一姿勢継続"]["count"] += 1
        stats["同一姿勢継続"]["total_duration_sec"] += float(seg["duration_sec"])

    return stats


def save_summary_json(summary: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_path


def build_natural_language_summary(
    input_video_name: str,
    aggregated_stats: dict,
    event_segments: list[dict],
    same_posture_segments: list[dict],
) -> str:
    lines: list[str] = []

    lines.append("KinKotsuMan 解析レポート")
    lines.append(f"対象動画: {input_video_name}")
    lines.append("")

    total_events = len(event_segments)
    total_same_posture = len(same_posture_segments)

    lines.append("■ 総評")
    if total_events == 0 and total_same_posture == 0:
        lines.append("エルゴ作業に該当する明確な区間は検出されませんでした。")
        return "\n".join(lines) + "\n"

    lines.append(
        f"エルゴ負荷が疑われる動作は合計 {total_events} 件、"
        f"同一姿勢の継続は {total_same_posture} 件 検出されました。"
    )
    lines.append("重要なのは、どの時間帯で発生しているかです。以下の発生箇所を確認してください。")
    lines.append("")

    lines.append("■ 発生回数と合計時間")
    for label in ["前屈姿勢", "頭上作業", "しゃがみ", "片腕過伸展", "同一姿勢継続"]:
        item = aggregated_stats[label]
        lines.append(f"- {label}: {item['count']}回 / 合計 {item['total_duration_sec']:.2f}秒")
    lines.append("")

    lines.append("■ 発生箇所一覧（時間帯）")
    if event_segments:
        for idx, seg in enumerate(event_segments, start=1):
            extra = ""
            if seg["label_ja"] == "前屈姿勢" and seg.get("max_trunk_angle") is not None:
                extra = f"、最大前傾角 {seg['max_trunk_angle']:.1f}度"

            lines.append(
                f"{idx}. {seg['label_ja']} が "
                f"{seg['start_sec']:.2f}秒〜{seg['end_sec']:.2f}秒 に発生 "
                f"(継続 {seg['duration_sec']:.2f}秒{extra})"
            )
    else:
        lines.append("エルゴ作業に該当する区間は検出されませんでした。")
    lines.append("")

    lines.append("■ 同一姿勢継続の発生箇所")
    if same_posture_segments:
        for idx, seg in enumerate(same_posture_segments, start=1):
            lines.append(
                f"{idx}. 同一姿勢継続 が "
                f"{seg['start_sec']:.2f}秒〜{seg['end_sec']:.2f}秒 に発生 "
                f"(継続 {seg['duration_sec']:.2f}秒)"
            )
    else:
        lines.append("長時間の同一姿勢継続は検出されませんでした。")
    lines.append("")

    lines.append("※ 本結果は骨格推定ベースの簡易判定です。現場判断では映像確認と併用してください。")

    return "\n".join(lines) + "\n"


def save_text_report(text: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(text)
    return output_path


def format_vtt_time(sec: float) -> str:
    total_ms = int(round(sec * 1000))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    seconds = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def _event_segment_to_vtt_text(seg: dict) -> str:
    label = seg["label_ja"]
    start_sec = seg["start_sec"]
    end_sec = seg["end_sec"]
    duration_sec = seg["duration_sec"]

    if label == "前屈姿勢" and seg.get("max_trunk_angle") is not None:
        return (
            f"{label}\n"
            f"{start_sec:.2f}s - {end_sec:.2f}s\n"
            f"継続 {duration_sec:.2f}s / 最大前傾角 {seg['max_trunk_angle']:.1f}度"
        )

    return (
        f"{label}\n"
        f"{start_sec:.2f}s - {end_sec:.2f}s\n"
        f"継続 {duration_sec:.2f}s"
    )


def _same_posture_segment_to_vtt_text(seg: dict) -> str:
    return (
        f"同一姿勢継続\n"
        f"{seg['start_sec']:.2f}s - {seg['end_sec']:.2f}s\n"
        f"継続 {seg['duration_sec']:.2f}s"
    )


def build_vtt_from_segments(event_segments: list[dict], same_posture_segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]

    idx = 1
    for seg in event_segments:
        lines.append(str(idx))
        lines.append(f"{format_vtt_time(seg['start_sec'])} --> {format_vtt_time(seg['end_sec'])}")
        lines.append(_event_segment_to_vtt_text(seg))
        lines.append("")
        idx += 1

    for seg in same_posture_segments:
        lines.append(str(idx))
        lines.append(f"{format_vtt_time(seg['start_sec'])} --> {format_vtt_time(seg['end_sec'])}")
        lines.append(_same_posture_segment_to_vtt_text(seg))
        lines.append("")
        idx += 1

    return "\n".join(lines).rstrip() + "\n"


def save_vtt_file(text: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(text)
    return output_path