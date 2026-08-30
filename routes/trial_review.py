"""Endpoints for the long-jump trial-review page: per-camera video playback,
step length/cadence/velocity, and toe-path trajectories.

Reuses the existing `RunSession` lookup/path-resolution helpers from
routes.run instead of duplicating them.
"""

import json
import re
import sys
from uuid import UUID
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from db_models import Video
from config import PIPELINE_ROOT
from response_chemas import StepsOut, ToePathOut
from routes.run import _get_run_session, _session_dir, _analysis_csvs, _with_time_column

pipeline_dir = str(PIPELINE_ROOT)
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)
from scripts.analysis.ankle_step_stride import _homography_calibration, _transform_homography


router = APIRouter()


@router.get("/run_session/{run_session_id}/video/{camera_index}")
async def get_run_session_camera_video(
    run_session_id: UUID,
    camera_index: int,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    stmt = select(Video).where(
        Video.run_session_id == run_session_id,
        Video.camera_index == camera_index,
    )
    video = (await session.execute(stmt)).scalars().first()
    if not video or not video.video_path or not Path(video.video_path).exists():
        raise HTTPException(status_code=404, detail="Camera video not found")

    # Prefer the skeleton + landing-point overlay rendered by
    # overlay_videos_per_camera() (core/overlay.py) if this session was
    # analyzed after that feature was added; older sessions won't have it,
    # so fall back to the raw camera footage.
    #
    # no-store: this same URL can start returning different bytes over time
    # (raw footage -> overlay, once a backfill/re-analysis runs), and browsers
    # cache <video> sources aggressively by URL even across reloads. Without
    # this, a browser that loaded the URL before a backfill can keep playing
    # the stale cached copy indefinitely.
    no_cache_headers = {"Cache-Control": "no-store"}
    run_session = await _get_run_session(run_session_id, session)
    overlay_path = _session_dir(run_session) / f"cam{camera_index + 1}_overlay.mp4"
    if overlay_path.exists():
        return FileResponse(overlay_path, headers=no_cache_headers)
    return FileResponse(Path(video.video_path), headers=no_cache_headers)


@router.get("/run_session/{run_session_id}/topdown_review")
async def get_topdown_review_availability(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
):
    """Return only successfully exported calibrated-camera review videos."""
    run_session = await _get_run_session(run_session_id, session)
    camera_indices = []
    for path in _session_dir(run_session).glob("cam*_topdown_review.mp4"):
        match = re.fullmatch(r"cam(\d+)_topdown_review\.mp4", path.name)
        if match:
            camera_indices.append(int(match.group(1)) - 1)
    full_trial_path = _session_dir(run_session) / "trial_topdown_review.mp4"
    return {
        "cameraIndices": sorted(camera_indices),
        "fullTrialAvailable": full_trial_path.exists(),
    }


@router.get("/run_session/{run_session_id}/topdown_review/full_trial")
async def get_full_trial_topdown_review_video(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    run_session = await _get_run_session(run_session_id, session)
    path = _session_dir(run_session) / "trial_topdown_review.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Full-trial top-down review video not available")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.get("/run_session/{run_session_id}/topdown_review/{camera_index}")
async def get_topdown_review_video(
    run_session_id: UUID,
    camera_index: int,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    run_session = await _get_run_session(run_session_id, session)
    path = _session_dir(run_session) / f"cam{camera_index + 1}_topdown_review.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Top-down review video not available")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


def _step_events_csv(session_dir: Path) -> Path | None:
    matches = sorted(session_dir.glob("*_step_events.csv"))
    return matches[0] if matches else None


def _ankle_positions_csv(session_dir: Path) -> Path | None:
    matches = sorted(session_dir.glob("*_ankle_positions.csv"))
    return matches[0] if matches else None


@router.get("/run_session/{run_session_id}/steps", response_model=StepsOut)
async def get_run_session_steps(
    run_session_id: UUID,
    response: Response,
    session: AsyncSession = Depends(get_session),
):
    # Same reasoning as get_run_session_camera_video's no-store: the numbers
    # this returns can change over time for the same run_session_id (e.g.
    # the timeSec-correction fix below applying retroactively to a session
    # that was already open in a browser tab), so this must never be served
    # from a stale cache.
    response.headers["Cache-Control"] = "no-store"
    run_session = await _get_run_session(run_session_id, session)
    session_dir = _session_dir(run_session)
    steps_csv = _step_events_csv(session_dir)
    if not steps_csv:
        raise HTTPException(status_code=404, detail="Step events file not found")

    df = pd.read_csv(steps_csv)
    wanted_cols = [
        "step_index", "time_s", "cam", "foot", "event_type",
        "step_length_m", "cadence_spm", "avg_cadence_spm",
        "world_x_m", "world_y_m",
    ]
    for col in wanted_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values("time_s")

    # Join per-frame speed (from metrics.csv, a different pipeline stage than
    # step detection) onto each step by nearest timestamp.
    metrics_csv, _ = _analysis_csvs(run_session)
    if metrics_csv:
        speed_df = _with_time_column(pd.read_csv(metrics_csv), run_session.fps)
        if "speed_mps" in speed_df.columns and "time_sec" in speed_df.columns:
            speed_df = speed_df.sort_values("time_sec")
            df = pd.merge_asof(
                df, speed_df[["time_sec", "speed_mps"]],
                left_on="time_s", right_on="time_sec", direction="nearest",
            )
        else:
            df["speed_mps"] = np.nan
    else:
        df["speed_mps"] = np.nan

    # ``time_s`` in the step-event CSV is local to each camera clip, whereas
    # metrics.csv uses the stitched timeline.  A nearest-time join therefore
    # cannot reliably supply the last-camera contact speeds.  Each detected
    # contact already carries its interval-based cadence, so calculate its
    # contact-to-contact speed directly from the corresponding step length.
    # This is also the quantity expected in the last-three approach-step
    # comparison.  Keep the per-frame metrics speed only as a fallback for
    # rows without a valid step interval.
    step_length = pd.to_numeric(df["step_length_m"], errors="coerce")
    cadence = pd.to_numeric(df["cadence_spm"], errors="coerce")
    derived_step_speed = step_length * cadence / 60.0
    valid_step_speed = np.isfinite(derived_step_speed) & (derived_step_speed >= 0)
    df.loc[valid_step_speed, "speed_mps"] = derived_step_speed[valid_step_speed]

    # ``time_s`` above is elapsed time within the RAW per-camera clip
    # (orig_frame / fps). But the video actually served for playback (see
    # get_run_session_camera_video) is the *overlay* export, which only
    # contains the subset of frames the tracking stage kept for that camera
    # -- often far fewer than the raw clip. Using raw time_s as a playback
    # position desyncs in proportion to how many frames that camera's
    # tracked window skipped, which can differ a lot camera to camera.
    # Recompute each step's true position within the exported overlay video:
    # its rank (0-indexed) among all *tracked* frames for the same camera --
    # in the same order overlay_videos_per_camera() (core/overlay.py) wrote
    # them -- divided by that camera's fps.
    ankle_csv = _ankle_positions_csv(session_dir)
    if ankle_csv is not None and "orig_frame" in df.columns:
        ankle_df = pd.read_csv(ankle_csv)
        if {"cam", "orig_frame"}.issubset(ankle_df.columns):
            ankle_df = ankle_df.sort_values(["cam", "orig_frame"]).reset_index(drop=True)
            ankle_df["_local_rank"] = ankle_df.groupby("cam").cumcount()

            fps_by_cam: dict[int, float] = {}
            for cam_idx in ankle_df["cam"].dropna().unique():
                cam_idx = int(cam_idx)
                video_stmt = select(Video).where(
                    Video.run_session_id == run_session_id,
                    Video.camera_index == cam_idx,
                )
                video = (await session.execute(video_stmt)).scalars().first()
                if video and video.video_path and Path(video.video_path).exists():
                    cap = cv2.VideoCapture(video.video_path)
                    fps_by_cam[cam_idx] = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    cap.release()

            ankle_df["overlay_time_s"] = ankle_df.apply(
                lambda r: r["_local_rank"] / fps_by_cam.get(int(r["cam"]), 30.0)
                if pd.notna(r["cam"]) else np.nan,
                axis=1,
            )
            df = df.merge(
                ankle_df[["cam", "orig_frame", "overlay_time_s"]],
                on=["cam", "orig_frame"], how="left",
            )
            df["time_s"] = df["overlay_time_s"].fillna(df["time_s"])

    cleaned = df.replace([np.inf, -np.inf], np.nan)

    def _val(row, col):
        value = row.get(col)
        return None if pd.isna(value) else float(value)

    steps = []
    for _, row in cleaned.iterrows():
        foot = row.get("foot")
        event_type = row.get("event_type")
        steps.append({
            "stepIndex": int(row["step_index"]),
            "timeSec": float(row["time_s"]),
            "cam": int(row["cam"]) if not pd.isna(row.get("cam")) else 0,
            "foot": None if pd.isna(foot) else str(foot),
            "eventType": None if pd.isna(event_type) else str(event_type),
            "stepLengthM": _val(row, "step_length_m"),
            "cadenceSpm": _val(row, "cadence_spm"),
            "velocityMps": _val(row, "speed_mps"),
            "worldXM": _val(row, "world_x_m"),
            "worldYM": _val(row, "world_y_m"),
        })

    step_lengths = cleaned["step_length_m"].dropna()
    cadences = cleaned["avg_cadence_spm"].dropna()
    avg_step_length = float(step_lengths.mean()) if len(step_lengths) else None
    avg_cadence = float(cadences.iloc[-1]) if len(cadences) else None

    return {
        "avgStepLengthM": avg_step_length,
        "avgCadenceSpm": avg_cadence,
        "steps": steps,
    }


_FOOT_KEYPOINT_NAMES = [
    "L_big_toe", "L_small_toe", "L_heel",
    "R_big_toe", "R_small_toe", "R_heel",
]


def _camera_homographies(videos: list[Video]) -> dict[int, np.ndarray]:
    """Build a {camera_index: homography_matrix} map from any camera whose
    Video.anchors is a 6-point homography calibration (see routes/upload.py's
    _camera_config_from_homography_anchors). Cameras with the legacy 4-point
    anchors, no anchors, or a bad calibration are simply omitted.
    """
    homographies: dict[int, np.ndarray] = {}
    for video in videos:
        if not video.anchors:
            continue
        try:
            anchors = json.loads(video.anchors)
        except (TypeError, ValueError):
            continue
        if not isinstance(anchors, list) or len(anchors) != 6:
            continue
        if not all(a.get("world_x_m") is not None and a.get("world_y_m") is not None for a in anchors):
            continue

        cap = cv2.VideoCapture(video.video_path)
        try:
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if cap.isOpened() else 0
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if cap.isOpened() else 0
        finally:
            cap.release()
        if not w or not h:
            continue

        src_points = [[float(a["x"]) * w, float(a["y"]) * h] for a in anchors]
        dst_points = [[float(a["world_x_m"]), float(a["world_y_m"])] for a in anchors]
        try:
            calibration = _homography_calibration(src_points, dst_points)
        except ValueError:
            continue
        if calibration is not None:
            homographies[video.camera_index] = calibration["homography"]

    return homographies


def _camera_world_controls(videos: list[Video]) -> dict[int, np.ndarray]:
    controls: dict[int, np.ndarray] = {}
    for video in videos:
        if not video.anchors:
            continue
        try:
            anchors = json.loads(video.anchors)
        except (TypeError, ValueError):
            continue
        if not isinstance(anchors, list) or len(anchors) != 6:
            continue
        if not all(a.get("world_x_m") is not None and a.get("world_y_m") is not None for a in anchors):
            continue
        controls[video.camera_index] = np.asarray(
            [[float(a["world_x_m"]), float(a["world_y_m"])] for a in anchors],
            dtype=np.float64,
        )
    return controls


@router.get("/run_session/{run_session_id}/trajectory")
async def get_run_session_trajectory(
    run_session_id: UUID,
    response: Response,
    session: AsyncSession = Depends(get_session),
):
    """Return the complete calibrated race path and its playback time map.

    This is the seam consumed by the interactive Flutter chart. Homography,
    cross-camera offsets, geometry inference, and frame alignment stay here;
    the caller only receives world-space values ready to draw.
    """
    response.headers["Cache-Control"] = "no-store"
    run_session = await _get_run_session(run_session_id, session)
    metrics_csv, _ = _analysis_csvs(run_session)
    if not metrics_csv:
        raise HTTPException(status_code=404, detail="Trajectory metrics not found")

    metrics = pd.read_csv(metrics_csv)
    required = {"cam", "absolute_frame", "dist_m", "image_point_x", "image_point_y"}
    if not required.issubset(metrics.columns):
        raise HTTPException(status_code=404, detail="Calibrated trajectory columns not found")

    videos_stmt = select(Video).where(Video.run_session_id == run_session_id)
    videos = list((await session.execute(videos_stmt)).scalars().all())
    homographies = _camera_homographies(videos)
    controls = _camera_world_controls(videos)
    if not homographies or set(controls) != set(homographies):
        raise HTTPException(status_code=404, detail="Complete 6-point Homography not found")

    projected_rows = []
    offset_samples: dict[int, list[float]] = {index: [] for index in homographies}
    for _, row in metrics.sort_values("absolute_frame").iterrows():
        camera_index = int(row["cam"]) - 1
        homography = homographies.get(camera_index)
        if homography is None:
            continue
        values = [row.get("dist_m"), row.get("image_point_x"), row.get("image_point_y")]
        if not all(pd.notna(value) and np.isfinite(float(value)) for value in values):
            continue
        local_x, world_y = _transform_homography(
            (float(row["image_point_x"]), float(row["image_point_y"])),
            homography,
        )
        if not np.isfinite(local_x) or not np.isfinite(world_y):
            continue
        global_x = float(row["dist_m"])
        offset_samples[camera_index].append(global_x - float(local_x))
        projected_rows.append((row, camera_index, global_x, float(world_y)))

    camera_offsets: dict[int, float] = {}
    fallback_x = 0.0
    for camera_index in sorted(controls):
        world = controls[camera_index]
        local_min_x = float(world[:, 0].min())
        local_max_x = float(world[:, 0].max())
        samples = offset_samples[camera_index]
        camera_offsets[camera_index] = (
            float(np.median(np.asarray(samples, dtype=np.float64)))
            if samples
            else fallback_x - local_min_x
        )
        fallback_x += local_max_x - local_min_x

    fps = float(run_session.fps or 60)
    cadence_events = []
    steps_csv = _step_events_csv(_session_dir(run_session))
    if steps_csv:
        steps = pd.read_csv(steps_csv)
        if {"seq_frame", "cadence_spm"}.issubset(steps.columns):
            cadence_events = sorted(
                (int(row["seq_frame"]), float(row["cadence_spm"]))
                for _, row in steps.iterrows()
                if pd.notna(row.get("seq_frame"))
                and pd.notna(row.get("cadence_spm"))
                and np.isfinite(float(row["cadence_spm"]))
            )

    def optional_number(row, column):
        value = row.get(column)
        return float(value) if pd.notna(value) and np.isfinite(float(value)) else None

    samples = []
    local_ranks: dict[int, int] = {}
    cadence_index = 0
    current_cadence = None
    for row, camera_index, world_x, world_y in projected_rows:
        absolute_frame = int(row["absolute_frame"])
        while cadence_index < len(cadence_events) and cadence_events[cadence_index][0] <= absolute_frame:
            current_cadence = cadence_events[cadence_index][1]
            cadence_index += 1
        local_rank = local_ranks.get(camera_index, 0)
        local_ranks[camera_index] = local_rank + 1
        samples.append(
            {
                "globalFrame": absolute_frame,
                "globalTimeSec": absolute_frame / fps,
                "cameraIndex": camera_index,
                "cameraTimeSec": local_rank / fps,
                "worldXM": world_x,
                "worldYM": world_y,
                "speedMps": optional_number(row, "speed_mps"),
                "accelerationMps2": optional_number(row, "accel_mps2"),
                "cadenceSpm": current_cadence,
            }
        )

    if not samples:
        raise HTTPException(status_code=404, detail="No calibrated trajectory samples")

    camera_segments = []
    all_adjusted_controls = []
    for camera_index in sorted(controls):
        adjusted = controls[camera_index].copy()
        adjusted[:, 0] += camera_offsets[camera_index]
        all_adjusted_controls.extend(adjusted.tolist())
        camera_samples = [s for s in samples if s["cameraIndex"] == camera_index]
        camera_segments.append(
            {
                "cameraIndex": camera_index,
                "offsetM": camera_offsets[camera_index],
                "startXM": float(adjusted[:, 0].min()),
                "endXM": float(adjusted[:, 0].max()),
                "minYM": float(adjusted[:, 1].min()),
                "maxYM": float(adjusted[:, 1].max()),
                "globalStartTimeSec": camera_samples[0]["globalTimeSec"] if camera_samples else None,
                "globalEndTimeSec": camera_samples[-1]["globalTimeSec"] if camera_samples else None,
            }
        )

    all_controls = np.asarray(all_adjusted_controls, dtype=np.float64)
    final_camera_index = max(controls)
    final_world = controls[final_camera_index]
    final_unique_x = sorted({float(value) for value in final_world[:, 0]})
    takeoff_board_x = None
    sandpit = None
    if run_session.is_long_jump and len(final_unique_x) >= 3:
        takeoff_board_x = final_unique_x[1] + camera_offsets[final_camera_index]
        sandpit = {
            "startXM": takeoff_board_x,
            "endXM": final_unique_x[-1] + camera_offsets[final_camera_index],
            "minYM": float(final_world[:, 1].min()),
            "maxYM": float(final_world[:, 1].max()),
        }

    return {
        "fps": fps,
        "durationSec": samples[-1]["globalTimeSec"],
        "bounds": {
            "minXM": float(all_controls[:, 0].min()),
            "maxXM": float(all_controls[:, 0].max()),
            "minYM": float(all_controls[:, 1].min()),
            "maxYM": float(all_controls[:, 1].max()),
        },
        "cameras": camera_segments,
        "takeoffBoardXM": takeoff_board_x,
        "sandpit": sandpit,
        "samples": samples,
    }


@router.get("/run_session/{run_session_id}/toe_path", response_model=ToePathOut)
async def get_run_session_toe_path(run_session_id: UUID, session: AsyncSession = Depends(get_session)):
    run_session = await _get_run_session(run_session_id, session)
    session_dir = _session_dir(run_session)

    foot_npz = session_dir / "sequential_tracked" / "input_2D" / "foot_keypoints.npz"
    if not foot_npz.exists():
        raise HTTPException(status_code=404, detail="Foot keypoints not found")

    offsets_matches = sorted(session_dir.glob("*_offsets.npz"))
    if not offsets_matches:
        raise HTTPException(status_code=404, detail="Tracking offsets not found")
    offsets_data = np.load(offsets_matches[0], allow_pickle=True)
    offsets = offsets_data["offsets"]
    orig_frames = offsets_data["orig_frames"]
    cam_indices = (
        offsets_data["cam_indices"].astype(int)
        if "cam_indices" in offsets_data
        else np.zeros(len(orig_frames), dtype=int)
    )

    videos_stmt = select(Video).where(Video.run_session_id == run_session_id)
    videos = (await session.execute(videos_stmt)).scalars().all()
    homographies = _camera_homographies(list(videos))

    feet_data = np.load(foot_npz, allow_pickle=True)
    points = feet_data["keypoints"][0]  # (T, 6, 2)
    scores = feet_data["scores"][0]  # (T, 6)
    valid_frames = np.asarray(feet_data["valid_frames"]).flatten().astype(int)
    keypoint_names = (
        [str(n) for n in feet_data["keypoint_names"]]
        if "keypoint_names" in feet_data
        else _FOOT_KEYPOINT_NAMES
    )

    fps = run_session.fps or 60

    frames = []
    for seq_frame, offset_idx in enumerate(valid_frames):
        if seq_frame >= len(points) or offset_idx >= len(offsets):
            continue
        off_x, off_y = offsets[offset_idx]
        orig_frame = int(orig_frames[offset_idx]) if offset_idx < len(orig_frames) else seq_frame
        cam_idx = int(cam_indices[offset_idx]) if offset_idx < len(cam_indices) else 0
        homography = homographies.get(cam_idx)

        point_out = {}
        for idx, name in enumerate(keypoint_names):
            if idx >= points.shape[1]:
                continue
            px = float(points[seq_frame, idx, 0] + off_x)
            py = float(points[seq_frame, idx, 1] + off_y)
            world_x, world_y = (None, None)
            if homography is not None:
                world_x, world_y = _transform_homography((px, py), homography)
            point_out[name] = {
                "x": px,
                "y": py,
                "worldXM": world_x,
                "worldYM": world_y,
                "score": float(scores[seq_frame, idx]),
            }
        frames.append({
            "seqFrame": int(seq_frame),
            "origFrame": orig_frame,
            "timeSec": orig_frame / fps if fps > 0 else seq_frame / 60.0,
            "points": point_out,
        })

    return {
        "keypointNames": keypoint_names,
        "hasWorldCoords": bool(homographies),
        "frames": frames,
    }
