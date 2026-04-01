from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp

# 公式の33ランドマーク名に対応
# nose, eyes, ears, mouth, shoulders, elbows, wrists, fingers, hips, knees, ankles, heels, foot index
LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

# 骨格描画用の接続
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]


@dataclass
class PoseResult:
    landmarks_px: dict[str, tuple[float, float]]
    landmarks_norm: dict[str, tuple[float, float, float, float]]
    world_landmarks: dict[str, tuple[float, float, float, float]]


class PoseEstimator:
    def __init__(
        self,
        model_path: str | Path,
        person_index: int = 0,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.person_index = person_index

        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=max(1, person_index + 1),
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, frame_bgr, timestamp_ms: int) -> PoseResult | None:
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None

        if self.person_index >= len(result.pose_landmarks):
            return None

        lm_list = result.pose_landmarks[self.person_index]
        world_list = result.pose_world_landmarks[self.person_index] if result.pose_world_landmarks else None

        landmarks_px: dict[str, tuple[float, float]] = {}
        landmarks_norm: dict[str, tuple[float, float, float, float]] = {}
        world_landmarks: dict[str, tuple[float, float, float, float]] = {}

        for idx, lm in enumerate(lm_list):
            name = LANDMARK_NAMES[idx]
            x_px = float(lm.x) * w
            y_px = float(lm.y) * h
            landmarks_px[name] = (x_px, y_px)
            landmarks_norm[name] = (
                float(lm.x),
                float(lm.y),
                float(lm.z),
                float(lm.visibility),
            )

        if world_list:
            for idx, lm in enumerate(world_list):
                name = LANDMARK_NAMES[idx]
                visibility = getattr(lm, "visibility", 1.0)
                world_landmarks[name] = (
                    float(lm.x),
                    float(lm.y),
                    float(lm.z),
                    float(visibility),
                )

        return PoseResult(
            landmarks_px=landmarks_px,
            landmarks_norm=landmarks_norm,
            world_landmarks=world_landmarks,
        )