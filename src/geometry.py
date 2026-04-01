from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def to_np(point: Iterable[float]) -> np.ndarray:
    return np.array(list(point), dtype=float)


def calc_angle_2d(a, b, c) -> float | None:
    """
    点bを頂点とする角度（0-180度）を返す
    """
    a = to_np(a[:2])
    b = to_np(b[:2])
    c = to_np(c[:2])

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return None

    cos_angle = float(np.dot(ba, bc) / denom)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def midpoint(p1, p2):
    return (
        (float(p1[0]) + float(p2[0])) / 2.0,
        (float(p1[1]) + float(p2[1])) / 2.0,
    )


def distance_2d(p1, p2) -> float:
    return math.dist((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))


def trunk_forward_angle_deg(shoulder_mid, hip_mid) -> float | None:
    """
    画像上の縦方向を基準にした体幹前傾角（0=直立に近い）。
    真横撮影を想定した簡易指標。
    """
    dx = float(shoulder_mid[0]) - float(hip_mid[0])
    dy = float(shoulder_mid[1]) - float(hip_mid[1])

    norm = math.hypot(dx, dy)
    if norm == 0:
        return None

    # 画像座標系ではyが下方向に増えるため、縦ベクトルとのずれを使う
    angle = abs(math.degrees(math.atan2(dx, dy if abs(dy) > 1e-9 else 1e-9)))
    return angle


def is_above(p1, p2) -> bool:
    """
    画像座標では y が小さい方が上
    """
    return float(p1[1]) < float(p2[1])


def safe_mean(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))