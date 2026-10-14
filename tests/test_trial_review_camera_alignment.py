import numpy as np
import pandas as pd
import pytest

from routes.trial_review import _compute_camera_offsets
from scripts.analysis.ankle_step_stride import _homography_calibration


def _scaled_homography(scale, tx, ty):
    src = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    dst = [
        [tx, ty],
        [tx + scale * 10, ty],
        [tx + scale * 10, ty + scale * 10],
        [tx, ty + scale * 10],
    ]
    return _homography_calibration(src, dst)["homography"]


def test_compute_camera_offsets_aligns_drifted_camera_to_continuous_distance():
    # Camera 0's own homography already agrees with the continuously tracked
    # distance (no calibration error), so its offset should come out ~0.
    homography_cam0 = _scaled_homography(scale=0.01, tx=0.0, ty=0.0)
    # Camera 1's own calibration has a 3m systematic error versus the
    # continuously tracked distance -- e.g. its anchors were measured
    # slightly off, or perspective distortion biased the fit. Its own pixel
    # coordinates also reset to 0 at the handoff, the way a fresh camera clip
    # does, while the continuously tracked distance keeps advancing smoothly.
    drift = 3.0
    homography_cam1 = _scaled_homography(scale=0.01, tx=drift, ty=0.0)

    homographies = {0: homography_cam0, 1: homography_cam1}
    controls = {
        0: np.array([[0.0, 0.0], [0.4, 0.0]]),
        1: np.array([[drift, 0.0], [drift + 0.4, 0.0]]),
    }

    rows = []
    for frame in range(5):
        rows.append(
            {
                "cam": 1,  # metrics.csv is 1-indexed
                "absolute_frame": frame,
                "dist_m": frame * 0.1,
                "image_point_x": frame * 10,
                "image_point_y": 0,
            }
        )
    for frame in range(5, 10):
        rows.append(
            {
                "cam": 2,
                "absolute_frame": frame,
                "dist_m": frame * 0.1,  # never resets at the handoff
                "image_point_x": (frame - 5) * 10,  # this camera's own pixels do reset
                "image_point_y": 0,
            }
        )
    metrics = pd.DataFrame(rows)

    offsets = _compute_camera_offsets(metrics, homographies, controls)

    assert offsets[0] == pytest.approx(0.0, abs=1e-2)
    assert offsets[1] == pytest.approx(0.5 - drift, abs=1e-2)


def test_compute_camera_offsets_falls_back_to_control_geometry_without_samples():
    # No metrics rows reference camera 1 at all (e.g. tracking never picked
    # it up), so there is nothing to measure a drift from; the fallback
    # should place it right after camera 0's calibrated span ends.
    homography_cam0 = _scaled_homography(scale=0.01, tx=0.0, ty=0.0)
    homography_cam1 = _scaled_homography(scale=0.01, tx=100.0, ty=0.0)

    homographies = {0: homography_cam0, 1: homography_cam1}
    controls = {
        0: np.array([[0.0, 0.0], [8.0, 0.0]]),
        1: np.array([[100.0, 0.0], [106.0, 0.0]]),
    }

    metrics = pd.DataFrame(
        [
            {
                "cam": 1,
                "absolute_frame": 0,
                "dist_m": 0.0,
                "image_point_x": 0,
                "image_point_y": 0,
            }
        ]
    )

    offsets = _compute_camera_offsets(metrics, homographies, controls)

    assert offsets[0] == pytest.approx(0.0, abs=1e-2)
    # fallback_x starts at 0, then advances by camera 0's local span (8.0),
    # and camera 1's local min_x is its own control's 100.0, so the fallback
    # offset shifts it back down to continue right where camera 0 ended.
    assert offsets[1] == pytest.approx(8.0 - 100.0, abs=1e-6)
