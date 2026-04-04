from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

import cv2

from .ergo_rules import ErgoRuleEngine
from .pose_estimator import POSE_CONNECTIONS, PoseEstimator
from .reporter import (
    build_aggregated_stats,
    build_natural_language_summary,
    build_vtt_from_segments,
    save_frame_results_csv,
    save_summary_json,
    save_text_report,
    save_vtt_file,
    summarize_same_posture_segments,
    summarize_segments,
)
from .video_io import VideoReader, VideoWriter


class ErgoVideoAnalyzer:
    def __init__(
        self,
        model_path: str | Path,
        output_dir: str | Path,
        person_index: int = 0,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        write_video: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.person_index = person_index
        self.write_video = write_video
        self.progress_callback = progress_callback

        self.pose_estimator = PoseEstimator(
            model_path=self.model_path,
            person_index=self.person_index,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.rule_engine = ErgoRuleEngine()

    def _progress(self, value: int, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(value, message)

    def run(self, input_video_path: str | Path) -> dict:
        input_video_path = Path(input_video_path)
        stem = input_video_path.stem

        csv_path = self.output_dir / f"{stem}_frame_results.csv"
        summary_path = self.output_dir / f"{stem}_summary.json"
        report_txt_path = self.output_dir / f"{stem}_summary_report.txt"
        overlay_video_path = self.output_dir / f"{stem}_overlay.mp4"
        temp_overlay_video_path = self.output_dir / f"{stem}_overlay_raw.mp4"
        vtt_path = self.output_dir / f"{stem}_summary.vtt"

        reader = VideoReader(input_video_path)
        self._progress(5, "動画情報を取得しました")

        writer = None
        if self.write_video:
            writer = VideoWriter(
                output_path=temp_overlay_video_path,
                fps=reader.fps,
                width=reader.width,
                height=reader.height,
            )

        rows: list[dict] = []

        try:
            total_frames = max(1, reader.frame_count)

            for frame_idx, frame in reader:
                analyze_progress = 10 + int((frame_idx / total_frames) * 68)
                if frame_idx % 10 == 0 or frame_idx == total_frames - 1:
                    self._progress(
                        analyze_progress,
                        f"フレーム解析中... {frame_idx + 1}/{total_frames}"
                    )

                timestamp_ms = int((frame_idx / reader.fps) * 1000)

                pose_result = self.pose_estimator.detect(frame, timestamp_ms=timestamp_ms)
                landmarks_px = pose_result.landmarks_px if pose_result else None

                ergo_result = self.rule_engine.evaluate(landmarks_px)

                row = {
                    "frame": frame_idx,
                    "time_sec": frame_idx / reader.fps,
                    "label": ergo_result.label,
                    "risk_score": ergo_result.risk_score,
                    "trunk_angle": ergo_result.metrics.trunk_angle,
                    "left_elbow_angle": ergo_result.metrics.left_elbow_angle,
                    "right_elbow_angle": ergo_result.metrics.right_elbow_angle,
                    "left_knee_angle": ergo_result.metrics.left_knee_angle,
                    "right_knee_angle": ergo_result.metrics.right_knee_angle,
                    "wrist_above_shoulder": ergo_result.metrics.wrist_above_shoulder,
                    "squat_like": ergo_result.metrics.squat_like,
                    "one_arm_overextended": ergo_result.metrics.one_arm_overextended,
                    "visible": ergo_result.metrics.visible,
                    "posture_signature": ergo_result.metrics.posture_signature,
                }
                rows.append(row)

                if writer:
                    vis = frame.copy()
                    if landmarks_px:
                        self._draw_pose(vis, landmarks_px)
                    self._draw_header(vis, row)
                    writer.write(vis)

        finally:
            reader.release()
            if writer:
                writer.release()
            self.pose_estimator.close()

        self._progress(82, "CSVを書き出しています")
        save_frame_results_csv(rows, csv_path)

        self._progress(86, "イベント区間を集約しています")
        event_segments = summarize_segments(rows, min_duration_sec=1.0)

        self._progress(90, "同一姿勢継続を集約しています")
        same_posture_segments = summarize_same_posture_segments(rows, min_duration_sec=3.0)

        aggregated_stats = build_aggregated_stats(event_segments, same_posture_segments)

        summary = {
            "video_name": input_video_path.name,
            "aggregated_stats": aggregated_stats,
            "event_segments": event_segments,
            "same_posture_segments": same_posture_segments,
        }

        self._progress(93, "JSONサマリーを書き出しています")
        save_summary_json(summary, summary_path)

        report_text = build_natural_language_summary(
            input_video_name=input_video_path.name,
            aggregated_stats=aggregated_stats,
            event_segments=event_segments,
            same_posture_segments=same_posture_segments,
        )

        self._progress(96, "日本語サマリーを作成しています")
        save_text_report(report_text, report_txt_path)

        self._progress(98, "字幕ファイルを作成しています")
        vtt_text = build_vtt_from_segments(event_segments, same_posture_segments)
        save_vtt_file(vtt_text, vtt_path)

        final_video_path = None
        if self.write_video:
            self._progress(99, "ブラウザ再生用動画を生成しています")
            final_video_path = self._ensure_browser_playable_video(
                temp_input_path=temp_overlay_video_path,
                final_output_path=overlay_video_path,
            )

        self._progress(100, "完了しました")

        return {
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "report_txt_path": str(report_txt_path),
            "video_path": str(final_video_path) if final_video_path else None,
            "vtt_path": str(vtt_path),
            "summary_text": report_text,
        }

    def _ensure_browser_playable_video(
        self,
        temp_input_path: Path,
        final_output_path: Path,
    ) -> Path:
        ffmpeg_path = Path(r"C:\Users\PJ\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe")

        if not ffmpeg_path.exists():
            raise RuntimeError(
                f"ffmpeg が見つかりません: {ffmpeg_path}"
            )

        if not temp_input_path.exists():
            raise RuntimeError(
                f"変換元動画が見つかりません: {temp_input_path}"
            )

        if final_output_path.exists():
            try:
                final_output_path.unlink()
            except OSError:
                pass

        cmd = [
            str(ffmpeg_path),
            "-y",
            "-i", str(temp_input_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(final_output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "ffmpeg によるブラウザ再生用動画の生成に失敗しました。"
            ) from e
        finally:
            if temp_input_path.exists():
                try:
                    temp_input_path.unlink()
                except OSError:
                    pass

        if not final_output_path.exists():
            raise RuntimeError(
                f"ブラウザ再生用動画の生成結果が見つかりません: {final_output_path}"
            )

        return final_output_path

    def _draw_pose(self, frame, landmarks_px: dict[str, tuple[float, float]]) -> None:
        for p1_name, p2_name in POSE_CONNECTIONS:
            if p1_name not in landmarks_px or p2_name not in landmarks_px:
                continue
            p1 = tuple(map(int, landmarks_px[p1_name]))
            p2 = tuple(map(int, landmarks_px[p2_name]))
            cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

        for _, point in landmarks_px.items():
            p = tuple(map(int, point))
            cv2.circle(frame, p, 4, (0, 0, 255), -1, cv2.LINE_AA)

    def _draw_header(self, frame, row: dict) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (760, 150), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        lines = [
            f"time : {row['time_sec']:.2f}s",
            f"label: {row['label']}",
            f"risk : {row['risk_score']}",
            f"trunk: {self._fmt(row['trunk_angle'])}",
            f"knee : {self._fmt(row['left_knee_angle'])} / {self._fmt(row['right_knee_angle'])}",
        ]

        y = 35
        for line in lines:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

    @staticmethod
    def _fmt(value) -> str:
        if value is None:
            return "-"
        return f"{float(value):.1f}"