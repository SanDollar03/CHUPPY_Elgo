from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import (
    calc_angle_2d,
    is_above,
    midpoint,
    safe_mean,
    trunk_forward_angle_deg,
)


@dataclass
class FrameErgoMetrics:
    left_elbow_angle: float | None = None
    right_elbow_angle: float | None = None
    left_knee_angle: float | None = None
    right_knee_angle: float | None = None
    trunk_angle: float | None = None
    wrist_above_shoulder: bool = False
    squat_like: bool = False
    one_arm_overextended: bool = False
    visible: bool = False
    posture_signature: str = "unknown"


@dataclass
class FrameErgoResult:
    label: str
    risk_score: int
    metrics: FrameErgoMetrics = field(default_factory=FrameErgoMetrics)


class ErgoRuleEngine:
    """
    優先順位:
      1. 頭上作業
      2. しゃがみ
      3. 片腕過伸展
      4. 強い前屈
      5. 軽い前屈
      6. 通常
    """

    def evaluate(self, landmarks_px: dict[str, tuple[float, float]] | None) -> FrameErgoResult:
        if not landmarks_px:
            return FrameErgoResult(
                label="no_person",
                risk_score=0,
                metrics=FrameErgoMetrics(
                    visible=False,
                    posture_signature="no_person",
                ),
            )

        required = [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        for key in required:
            if key not in landmarks_px:
                return FrameErgoResult(
                    label="partial_pose",
                    risk_score=0,
                    metrics=FrameErgoMetrics(
                        visible=False,
                        posture_signature="partial_pose",
                    ),
                )

        left_elbow_angle = calc_angle_2d(
            landmarks_px["left_shoulder"],
            landmarks_px["left_elbow"],
            landmarks_px["left_wrist"],
        )
        right_elbow_angle = calc_angle_2d(
            landmarks_px["right_shoulder"],
            landmarks_px["right_elbow"],
            landmarks_px["right_wrist"],
        )
        left_knee_angle = calc_angle_2d(
            landmarks_px["left_hip"],
            landmarks_px["left_knee"],
            landmarks_px["left_ankle"],
        )
        right_knee_angle = calc_angle_2d(
            landmarks_px["right_hip"],
            landmarks_px["right_knee"],
            landmarks_px["right_ankle"],
        )

        shoulder_mid = midpoint(landmarks_px["left_shoulder"], landmarks_px["right_shoulder"])
        hip_mid = midpoint(landmarks_px["left_hip"], landmarks_px["right_hip"])
        trunk_angle = trunk_forward_angle_deg(shoulder_mid, hip_mid)

        left_wrist_above = is_above(landmarks_px["left_wrist"], landmarks_px["left_shoulder"])
        right_wrist_above = is_above(landmarks_px["right_wrist"], landmarks_px["right_shoulder"])
        wrist_above_shoulder = left_wrist_above or right_wrist_above

        knee_angle_avg = safe_mean([left_knee_angle, right_knee_angle])
        squat_like = knee_angle_avg is not None and knee_angle_avg < 110.0

        # 片腕過伸展:
        # 片側だけ肘がかなり伸び切っていて、もう片側との差がある場合に検出
        one_arm_overextended = False
        if left_elbow_angle is not None and right_elbow_angle is not None:
            if (
                (left_elbow_angle >= 165.0 and right_elbow_angle <= 145.0)
                or (right_elbow_angle >= 165.0 and left_elbow_angle <= 145.0)
            ):
                one_arm_overextended = True
        elif left_elbow_angle is not None and left_elbow_angle >= 170.0:
            one_arm_overextended = True
        elif right_elbow_angle is not None and right_elbow_angle >= 170.0:
            one_arm_overextended = True

        posture_signature = self._build_posture_signature(
            trunk_angle=trunk_angle,
            wrist_above_shoulder=wrist_above_shoulder,
            squat_like=squat_like,
            one_arm_overextended=one_arm_overextended,
        )

        metrics = FrameErgoMetrics(
            left_elbow_angle=left_elbow_angle,
            right_elbow_angle=right_elbow_angle,
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            trunk_angle=trunk_angle,
            wrist_above_shoulder=wrist_above_shoulder,
            squat_like=squat_like,
            one_arm_overextended=one_arm_overextended,
            visible=True,
            posture_signature=posture_signature,
        )

        if wrist_above_shoulder:
            return FrameErgoResult(label="overhead_work", risk_score=2, metrics=metrics)

        if squat_like:
            return FrameErgoResult(label="squatting", risk_score=2, metrics=metrics)

        if one_arm_overextended:
            return FrameErgoResult(label="one_arm_overextension", risk_score=2, metrics=metrics)

        if trunk_angle is not None and trunk_angle >= 45.0:
            return FrameErgoResult(label="bending", risk_score=3, metrics=metrics)

        if trunk_angle is not None and trunk_angle >= 25.0:
            return FrameErgoResult(label="bending", risk_score=2, metrics=metrics)

        return FrameErgoResult(label="normal", risk_score=0, metrics=metrics)

    def _build_posture_signature(
        self,
        trunk_angle: float | None,
        wrist_above_shoulder: bool,
        squat_like: bool,
        one_arm_overextended: bool,
    ) -> str:
        trunk_bin = "upright"
        if trunk_angle is not None:
            if trunk_angle >= 45.0:
                trunk_bin = "bend_high"
            elif trunk_angle >= 25.0:
                trunk_bin = "bend_mid"

        arm_bin = "arm_normal"
        if wrist_above_shoulder:
            arm_bin = "arm_overhead"
        elif one_arm_overextended:
            arm_bin = "arm_overextend"

        leg_bin = "leg_normal"
        if squat_like:
            leg_bin = "leg_squat"

        return f"{trunk_bin}|{arm_bin}|{leg_bin}"