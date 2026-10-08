"""Build a contract-v1 2D comparison from Server NPZ and Local JSON artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np


H36M_COCO_ORDER = (9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3)
COCO_ORDER = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
WHOLEBODY23_ORDER = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)


def build_2d_comparison_report(
    *,
    comparison_group_id: UUID,
    server_run_id: UUID,
    local_run_id: UUID,
    server_manifest: dict[str, Any],
    local_manifest: dict[str, Any],
    server_root: Path,
    local_root: Path,
) -> dict[str, Any]:
    input_hashes_match = _input_hashes(server_manifest) == _input_hashes(local_manifest)
    warnings: list[str] = []
    metrics: list[dict[str, Any]] = []

    if input_hashes_match:
        try:
            metrics.extend(_wholebody23_metrics(server_root, local_root))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            warnings.append(f"wholebody23 comparison unavailable: {error}")
        try:
            metrics.extend(_pose_metrics(server_root, local_root))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            warnings.append(f"pose2d comparison unavailable: {error}")
        try:
            metrics.extend(_bbox_metrics(server_root, local_root))
        except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
            warnings.append(f"bbox comparison unavailable: {error}")
    else:
        warnings.append("Server and Local input video hashes differ; numeric comparison skipped")

    server_duration = _stage_duration(server_manifest)
    local_duration = _stage_duration(local_manifest)
    ratio = (
        local_duration / server_duration
        if server_duration is not None and server_duration > 0 and local_duration is not None
        else None
    )
    status = "not_comparable" if not input_hashes_match else ("complete" if metrics else "partial")
    return {
        "schema_version": "1.0.0",
        "comparison_group_id": str(comparison_group_id),
        "request_id": str(local_manifest["request_id"]),
        "server_run_id": str(server_run_id),
        "local_run_id": str(local_run_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_hashes_match": input_hashes_match,
        "metric_differences": metrics,
        "performance": {
            "server_duration_seconds": server_duration,
            "local_duration_seconds": local_duration,
            "local_to_server_ratio": ratio,
        },
        "warnings": warnings,
    }


def _input_hashes(manifest: dict[str, Any]) -> list[tuple[int, str]]:
    return sorted(
        (int(item["camera_index"]), str(item["sha256"]))
        for item in manifest["input_videos"]
    )


def _stage_duration(manifest: dict[str, Any]) -> float | None:
    durations = [
        float(stage["duration_seconds"])
        for stage in manifest.get("stages", [])
        if stage.get("duration_seconds") is not None
    ]
    return sum(durations) if durations else None


def _find_one(root: Path, patterns: tuple[str, ...]) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    files = sorted({path.resolve() for path in matches if path.is_file()})
    if not files:
        raise FileNotFoundError(f"missing artifact matching {patterns}")
    return files[0]


def _server_pose_frames(root: Path) -> dict[tuple[int, int], np.ndarray]:
    pose_path = _find_one(root, ("**/input_2D/keypoints.npz",))
    offsets_path = _find_one(root, ("**/*_offsets.npz",))
    with np.load(pose_path, allow_pickle=False) as pose_data:
        reconstruction = np.asarray(pose_data["reconstruction"], dtype=np.float64)[0]
        valid = set(np.asarray(pose_data["valid_frames"]).reshape(-1).astype(int).tolist())
    with np.load(offsets_path, allow_pickle=False) as offset_data:
        offsets = np.asarray(offset_data["offsets"], dtype=np.float64)
        source_frames = np.asarray(offset_data["orig_frames"]).astype(int)
        camera_indices = (
            np.asarray(offset_data["cam_indices"]).astype(int)
            if "cam_indices" in offset_data
            else np.zeros(len(source_frames), dtype=int)
        )
    count = min(len(reconstruction), len(offsets), len(source_frames), len(camera_indices))
    result = {}
    for index in range(count):
        if valid and index not in valid:
            continue
        joints = reconstruction[index].copy()
        joints[:, :2] += offsets[index]
        result[(int(camera_indices[index]), int(source_frames[index]))] = joints
    return result


def _local_pose_frames(root: Path) -> dict[tuple[int, int], np.ndarray]:
    path = _find_one(root, ("pose/keypoints_2d.json", "**/keypoints_2d.json"))
    document = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for frame in document["frames"]:
        if not frame.get("valid") or len(frame.get("joints", [])) < 17:
            continue
        coco = np.asarray(
            [[joint["x"], joint["y"], joint["score"]] for joint in frame["joints"][:17]],
            dtype=np.float64,
        )
        result[(int(frame["camera_index"]), int(frame["source_frame"]))] = _coco_to_h36m(coco)
    return result


def _canonical_pose_document(root: Path) -> dict[str, Any]:
    path = _find_one(root, ("pose/keypoints_2d.json",))
    document = json.loads(path.read_text(encoding="utf-8"))
    if tuple(document.get("joint_order", [])) != WHOLEBODY23_ORDER:
        raise ValueError("canonical pose artifact joint_order is not WholeBody23")
    return document


def _canonical_wholebody23_frames(root: Path) -> dict[tuple[int, int], np.ndarray]:
    document = _canonical_pose_document(root)
    result = {}
    for frame in document["frames"]:
        joints = frame.get("joints", [])
        if not frame.get("valid") or len(joints) != 23:
            continue
        result[(int(frame["camera_index"]), int(frame["source_frame"]))] = np.asarray(
            [[joint["x"], joint["y"], joint["score"]] for joint in joints],
            dtype=np.float64,
        )
    return result


def _wholebody23_metrics(server_root: Path, local_root: Path) -> list[dict[str, Any]]:
    server = _canonical_wholebody23_frames(server_root)
    local = _canonical_wholebody23_frames(local_root)
    common = sorted(server.keys() & local.keys())
    if not common:
        raise ValueError("no matching valid WholeBody23 frames")
    all_deltas = np.concatenate([
        np.linalg.norm(server[key][:, :2] - local[key][:, :2], axis=1)
        for key in common
    ])
    foot_deltas = np.concatenate([
        np.linalg.norm(server[key][17:, :2] - local[key][17:, :2], axis=1)
        for key in common
    ])
    confidence_deltas = np.concatenate([
        np.abs(server[key][:, 2] - local[key][:, 2])
        for key in common
    ])
    denominator = max(len(server), len(local), 1)
    return [
        _delta("wholebody23_frame_match_rate", len(common) / denominator, 0.95, "ratio"),
        _delta("wholebody23_mean_pixel_delta", float(np.mean(all_deltas)), 2.0, "pixel"),
        _delta("wholebody23_p95_pixel_delta", float(np.percentile(all_deltas, 95)), 5.0, "pixel"),
        _delta(
            "wholebody23_mean_confidence_delta",
            float(np.mean(confidence_deltas)),
            0.05,
            "score",
        ),
        _delta("foot6_mean_pixel_delta", float(np.mean(foot_deltas)), 2.0, "pixel"),
        _delta("foot6_p95_pixel_delta", float(np.percentile(foot_deltas, 95)), 5.0, "pixel"),
    ]


def _coco_to_h36m(coco: np.ndarray) -> np.ndarray:
    output = np.zeros((17, 3), dtype=np.float64)
    output[list(H36M_COCO_ORDER)] = coco[list(COCO_ORDER)]
    output[10, :2] = np.mean(coco[1:5, :2], axis=0)
    output[10, 2] = np.mean(coco[1:5, 2])
    output[8, :2] = np.mean(coco[5:7, :2], axis=0)
    output[8, :2] += (coco[0, :2] - output[8, :2]) / 3
    output[8, 2] = np.mean(coco[5:7, 2])
    output[0] = np.mean(coco[11:13], axis=0)
    output[7] = np.mean(coco[[5, 6, 11, 12]], axis=0)
    output[7, 0] += 2 * (output[7, 0] - np.mean(output[[0, 8], 0]))
    output[7, 2] = np.mean(output[[0, 8], 2])
    output[9, :2] -= (output[9, :2] - np.mean(coco[5:7, :2], axis=0)) / 4
    return output


def _pose_metrics(server_root: Path, local_root: Path) -> list[dict[str, Any]]:
    server = _server_pose_frames(server_root)
    local = _local_pose_frames(local_root)
    common = sorted(server.keys() & local.keys())
    if not common:
        raise ValueError("no matching valid (camera_index, source_frame) pairs")
    pixel_deltas = np.concatenate(
        [np.linalg.norm(server[key][:, :2] - local[key][:, :2], axis=1) for key in common]
    )
    confidence_deltas = np.concatenate(
        [np.abs(server[key][:, 2] - local[key][:, 2]) for key in common]
    )
    denominator = max(len(server), len(local), 1)
    return [
        _delta("pose2d_frame_match_rate", len(common) / denominator, 0.95, "ratio"),
        _delta("pose2d_mean_pixel_delta", float(np.mean(pixel_deltas)), 2.0, "pixel"),
        _delta("pose2d_median_pixel_delta", float(np.median(pixel_deltas)), 2.0, "pixel"),
        _delta("pose2d_p95_pixel_delta", float(np.percentile(pixel_deltas, 95)), 5.0, "pixel"),
        _delta("pose2d_mean_confidence_delta", float(np.mean(confidence_deltas)), 0.05, "score"),
    ]


def _bbox_metrics(server_root: Path, local_root: Path) -> list[dict[str, Any]]:
    try:
        server_document = _canonical_pose_document(server_root)
        server = {
            (int(frame["camera_index"]), int(frame["source_frame"])):
            tuple(float(frame["bbox"][name]) for name in ("x1", "y1", "x2", "y2"))
            for frame in server_document["frames"] if frame.get("bbox") is not None
        }
    except FileNotFoundError:
        bbox_path = _find_one(server_root, ("**/*_bbox_map.csv",))
        with bbox_path.open(newline="", encoding="utf-8-sig") as source:
            server = {}
            for row in csv.DictReader(source):
                source_frame = row.get("source_frame") or row.get("cam_frame")
                if source_frame is None:
                    raise KeyError("bbox map has neither source_frame nor cam_frame")
                server[(int(row.get("cam", 0)), int(source_frame))] = tuple(
                    float(row[name]) for name in ("x1", "y1", "x2", "y2")
                )
    local_path = _find_one(local_root, ("pose/keypoints_2d.json", "**/keypoints_2d.json"))
    document = json.loads(local_path.read_text(encoding="utf-8"))
    local = {
        (int(frame["camera_index"]), int(frame["source_frame"])):
        tuple(float(frame["bbox"][name]) for name in ("x1", "y1", "x2", "y2"))
        for frame in document["frames"] if frame.get("bbox") is not None
    }
    common = server.keys() & local.keys()
    if not common:
        raise ValueError("no matching bbox frames")
    ious = np.asarray([_iou(server[key], local[key]) for key in common])
    return [
        _delta("bbox_frame_match_rate", len(common) / max(len(server), len(local), 1), 0.95, "ratio"),
        _delta("bbox_mean_iou", float(np.mean(ious)), 0.80, "ratio"),
    ]


def _iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _delta(metric: str, value: float, tolerance: float, unit: str) -> dict[str, Any]:
    higher_is_better = metric.endswith("match_rate") or metric.endswith("mean_iou")
    within = value >= tolerance if higher_is_better else value <= tolerance
    return {
        "metric": metric,
        "server_value": None,
        "local_value": None,
        "absolute_difference": value,
        "relative_difference_percent": None,
        "tolerance": tolerance,
        "within_tolerance": within,
        "unit": unit,
    }
