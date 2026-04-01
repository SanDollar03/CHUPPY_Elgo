from __future__ import annotations

from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, video_path: str | Path) -> None:
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"動画を開けません: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def __iter__(self):
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        self.cap.release()


class VideoWriter:
    def __init__(self, output_path: str | Path, fps: float, width: int, height: int) -> None:
        self.output_path = str(output_path)

        # まずはOpenCVで書ける形式で出す
        # 後段でffmpegがあればH.264へ変換する前提
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        if not self.writer.isOpened():
            raise RuntimeError(f"動画を書き込めません: {self.output_path}")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()