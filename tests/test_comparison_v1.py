import csv
import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from db_models import AnalysisRun, ComparisonReport, Runner, RunSession, User
from routes.analysis_run import generate_and_store_comparison_report
from utils.comparison_v1 import WHOLEBODY23_ORDER, build_2d_comparison_report
from utils.contract_v1 import validate_comparison_report


def _manifest(video_hash, request_id, duration):
    return {
        "request_id": str(request_id),
        "input_videos": [{"camera_index": 0, "sha256": video_hash}],
        "stages": [{"name": "pose2d", "duration_seconds": duration}],
    }


def test_builds_aligned_pose_and_bbox_report(tmp_path):
    server_root = tmp_path / "server"
    local_root = tmp_path / "local"
    pose_dir = server_root / "run" / "input_2D"
    pose_dir.mkdir(parents=True)
    local_pose_dir = local_root / "pose"
    local_pose_dir.mkdir(parents=True)
    server_pose_dir = server_root / "pose"
    server_pose_dir.mkdir(parents=True)

    joints = np.asarray([[[10.0 + i, 20.0 + i, 0.9] for i in range(17)]])
    np.savez(pose_dir / "keypoints.npz", reconstruction=joints[None], valid_frames=np.asarray([[0]]))
    np.savez(
        server_root / "cam1_offsets.npz",
        offsets=np.asarray([[3, 4]]),
        orig_frames=np.asarray([7]),
        cam_indices=np.asarray([0]),
    )
    with (server_root / "cam1_tracked_bbox_map.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=["cam", "source_frame", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerow({"cam": 0, "source_frame": 7, "x1": 1, "y1": 2, "x2": 20, "y2": 30})

    # The comparator performs the same COCO->H36M conversion as production;
    # this fixture only needs valid finite joints to exercise alignment.
    local_document = {
        "joint_order": list(WHOLEBODY23_ORDER),
        "frames": [{
            "camera_index": 0,
            "source_frame": 7,
            "valid": True,
            "bbox": {"x1": 1, "y1": 2, "x2": 20, "y2": 30},
            "joints": [{"x": 13.0 + i, "y": 24.0 + i, "score": 0.9} for i in range(23)],
        }]
    }
    (local_pose_dir / "keypoints_2d.json").write_text(json.dumps(local_document))
    server_document = {
        "joint_order": list(WHOLEBODY23_ORDER),
        "frames": [{
            "camera_index": 0,
            "source_frame": 7,
            "valid": True,
            "bbox": {"x1": 1, "y1": 2, "x2": 20, "y2": 30},
            "joints": [{"x": 13.0 + i, "y": 24.0 + i, "score": 0.9} for i in range(23)],
        }],
    }
    (server_pose_dir / "keypoints_2d.json").write_text(json.dumps(server_document))

    request_id = uuid4()
    report = build_2d_comparison_report(
        comparison_group_id=uuid4(),
        server_run_id=uuid4(),
        local_run_id=uuid4(),
        server_manifest=_manifest("a" * 64, request_id, 4.0),
        local_manifest=_manifest("a" * 64, request_id, 2.0),
        server_root=server_root,
        local_root=local_root,
    )

    validate_comparison_report(report)
    assert report["status"] == "complete"
    assert report["input_hashes_match"] is True
    assert report["performance"]["local_to_server_ratio"] == 0.5
    names = {item["metric"] for item in report["metric_differences"]}
    assert {
        "pose2d_mean_pixel_delta",
        "wholebody23_mean_pixel_delta",
        "foot6_mean_pixel_delta",
        "bbox_mean_iou",
    } <= names
    metrics = {item["metric"]: item for item in report["metric_differences"]}
    assert metrics["wholebody23_mean_pixel_delta"]["absolute_difference"] == 0
    assert metrics["foot6_mean_pixel_delta"]["absolute_difference"] == 0


def test_mismatched_inputs_are_not_compared(tmp_path):
    request_id = uuid4()
    report = build_2d_comparison_report(
        comparison_group_id=uuid4(),
        server_run_id=uuid4(),
        local_run_id=uuid4(),
        server_manifest=_manifest("a" * 64, request_id, 4.0),
        local_manifest=_manifest("b" * 64, request_id, 2.0),
        server_root=tmp_path,
        local_root=tmp_path,
    )
    validate_comparison_report(report)
    assert report["status"] == "not_comparable"
    assert report["metric_differences"] == []


def test_completed_pair_is_automatically_persisted(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(SQLModel.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user = User(username=f"user-{uuid4()}", hashed_password="hash")
            session.add(user)
            await session.flush()
            runner = Runner(name="Runner", user_id=user.id)
            session.add(runner)
            await session.flush()
            run_session = RunSession(
                runner_id=runner.id,
                date=datetime.now(timezone.utc),
                camera_count=1,
                fps=60,
                note="compare",
            )
            session.add(run_session)
            await session.flush()
            group_id = uuid4()
            server_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="server",
                comparison_group_id=group_id,
                status="completed",
            )
            local_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="local",
                comparison_group_id=group_id,
                status="degraded",
            )
            session.add_all([server_run, local_run])
            await session.commit()

            request_id = uuid4()
            server_root = tmp_path / str(runner.id) / str(run_session.id)
            local_root = tmp_path / str(run_session.id) / "local"
            server_root.mkdir(parents=True)
            local_root.mkdir(parents=True)
            (server_root / "manifest.json").write_text(
                json.dumps(_manifest("a" * 64, request_id, 4.0))
            )
            (local_root / "manifest.json").write_text(
                json.dumps(_manifest("b" * 64, request_id, 2.0))
            )

            report = await generate_and_store_comparison_report(group_id, session)
            assert report["status"] == "not_comparable"
            stored = await session.get(ComparisonReport, group_id)
            assert stored is not None
            assert stored.payload["input_hashes_match"] is False

        await engine.dispose()

    asyncio.run(scenario())
