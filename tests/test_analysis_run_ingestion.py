import asyncio
import csv
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
import pytest

from db_models import AnalysisMeta, AnalysisRun, ComparisonReport, Runner, RunSession, User, Video
from response_chemas import CreateLocalAnalysisRunIn
from routes.analysis_run import (
    create_local_analysis_run,
    get_analysis_run,
    get_comparison_report,
    ingest_analysis_run_manifest,
    submit_comparison_report,
)
from routes.run import (
    get_run_session_angles,
    get_run_session_graphs,
    get_run_session_info,
    get_run_session_video,
)
from routes.trial_review import get_run_session_camera_video, get_run_session_steps
from utils.contract_v1 import ContractValidationError, validate_artifact_document

JOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel",
]


def test_canonical_artifact_validation_rejects_schema_mismatch():
    with pytest.raises(ContractValidationError):
        validate_artifact_document(
            {
                "schema_version": "1.0.0",
                "frames": [{"speed_mps": "not-a-number"}],
            },
            artifact_type="metrics",
        )


def _valid_manifest(*, compute_location="local", camera_count=1, artifacts=None, status="completed"):
    return {
        "schema_version": "1.0.0",
        "request_id": str(uuid4()),
        "run_id": str(uuid4()),
        "comparison_group_id": None,
        "compute_location": compute_location,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": "a" * 64,
        "engine": {
            "name": "runner-pose-ondevice",
            "version": "spike-0.1",
            "models": [],
        },
        "input_videos": [
            {"camera_index": i, "sha256": "b" * 64} for i in range(camera_count)
        ],
        "coordinate_system": {
            "pose2d_origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "pose2d_unit": "original_video_pixel",
            "bbox_format": "x1_y1_x2_y2",
            "joint_order": JOINT_ORDER,
        },
        "stages": [{"name": "pose2d", "status": "completed"}],
        "summary": {
            "total_time_seconds": 12.3,
            "average_speed_mps": 8.1,
            "average_acceleration_mps2": 0.4,
            "detected_steps": 14,
            "average_cadence_spm": 281.5,
            "average_step_length_m": 1.85,
            "maximum_speed_mps": 9.7,
            "official_jump_distance_m": None,
            "takeoff_speed_mps": None,
            "flight_time_seconds": None,
        },
        "artifacts": artifacts or [],
        "warnings": [],
    }


async def _make_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _setup_runner_and_user(session):
    user = User(username=f"user-{uuid4()}", hashed_password="hash")
    session.add(user)
    await session.flush()
    runner = Runner(name="Runner", user_id=user.id)
    session.add(runner)
    await session.commit()
    return user, runner


def test_create_local_analysis_run_then_ingest_manifest_marks_completed(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)

            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1, fps=60),
                session=session,
                current_user=user,
            )

            manifest_doc = _valid_manifest(camera_count=1)
            manifest_file = UploadFile(
                file=BytesIO(json.dumps(manifest_doc).encode()), filename="manifest.json"
            )

            result = await ingest_analysis_run_manifest(
                created.analysisRunId,
                manifest=manifest_file,
                artifacts=[],
                idempotency_key="req-1",
                session=session,
                current_user=user,
            )

            assert result.status == "completed"
            assert result.resultSummary["average_speed_mps"] == 8.1

            # AnalysisMeta got the mirrored summary (non-Compare run).
            meta = (
                await session.execute(
                    select(AnalysisMeta).where(
                        AnalysisMeta.run_session_id == created.runSessionId
                    )
                )
            ).scalars().first()
            assert meta is not None
            assert meta.avg_velocity == 8.1

            manifest_on_disk = tmp_path / str(created.runSessionId) / "local" / "manifest.json"
            assert manifest_on_disk.exists()

        await engine.dispose()

    asyncio.run(scenario())


def test_compare_local_run_reuses_server_run_session():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)
            group_id = uuid4()
            run_session = RunSession(
                runner_id=runner.id,
                date=datetime.now(timezone.utc),
                camera_count=1,
                fps=60,
                note="compare",
                status="processing",
            )
            session.add(run_session)
            await session.flush()
            server_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="server",
                comparison_group_id=group_id,
                status="processing",
            )
            session.add(server_run)
            await session.commit()

            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(
                    runnerId=runner.id,
                    cameraCount=1,
                    fps=60,
                    comparisonGroupId=group_id,
                ),
                session=session,
                current_user=user,
            )

            assert created.runSessionId == run_session.id
            runs = (
                await session.execute(
                    select(AnalysisRun).where(
                        AnalysisRun.run_session_id == run_session.id
                    )
                )
            ).scalars().all()
            assert {run.compute_location for run in runs} == {"server", "local"}
            assert {run.comparison_group_id for run in runs} == {group_id}

        await engine.dispose()

    asyncio.run(scenario())


def test_local_ingestion_stores_source_video_for_run_session_playback(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module
    import routes.run as run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)
    monkeypatch.setattr(run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)
            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1, fps=60),
                session=session,
                current_user=user,
            )
            overlay_bytes = b"hrnet-overlay-video"
            overlay_artifact = {
                "artifact_id": str(uuid4()),
                "type": "overlay",
                "media_type": "video/mp4",
                "relative_path": "overlay/main.mp4",
                "sha256": hashlib.sha256(overlay_bytes).hexdigest(),
                "size_bytes": len(overlay_bytes),
                "camera_index": 0,
            }
            manifest_doc = _valid_manifest(artifacts=[overlay_artifact])
            manifest_doc["input_videos"][0]["sha256"] = hashlib.sha256(b"local-video").hexdigest()
            manifest_file = UploadFile(
                file=BytesIO(json.dumps(manifest_doc).encode()),
                filename="manifest.json",
            )
            source_video = UploadFile(file=BytesIO(b"local-video"), filename="IMG_0085.MOV")
            overlay_video = UploadFile(file=BytesIO(overlay_bytes), filename="overlay/main.mp4")

            await ingest_analysis_run_manifest(
                created.analysisRunId,
                manifest=manifest_file,
                artifacts=[overlay_video],
                input_videos=[source_video],
                idempotency_key="video-1",
                session=session,
                current_user=user,
            )

            expected = tmp_path / str(runner.id) / str(created.runSessionId) / "cam1.MOV"
            assert expected.read_bytes() == b"local-video"

            response = await get_run_session_video(
                created.runSessionId,
                session=session,
                current_user=user,
            )
            expected_overlay = tmp_path / str(created.runSessionId) / "local" / "overlay/main.mp4"
            assert response.path == expected_overlay

            camera_response = await get_run_session_camera_video(
                created.runSessionId,
                camera_index=0,
                session=session,
            )
            assert camera_response.path == expected_overlay

            steps_response = await get_run_session_steps(
                created.runSessionId,
                response=Response(),
                session=session,
            )
            assert steps_response == {
                "avgStepLengthM": None,
                "avgCadenceSpm": None,
                "steps": [],
                "runwayWidthByCam": None,
            }

            video = (
                await session.execute(
                    select(Video).where(Video.run_session_id == created.runSessionId)
                )
            ).scalars().one()
            assert video.camera_index == 0
            assert Path(video.video_path) == expected

        await engine.dispose()

    asyncio.run(scenario())


def test_local_angles_json_is_available_through_existing_result_api(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)
            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1, fps=60),
                session=session,
                current_user=user,
            )
            angles_doc = {
                "schema_version": "1.0.0",
                "frames": [
                    {
                        "camera_index": 0,
                        "source_frame": 12,
                        "timestamp_seconds": 0.2,
                        "raw_values_degrees": {"left_knee_angle": 130.0},
                        "values_degrees": {
                            "left_knee_angle": 50.0,
                            "right_knee_angle": 55.0,
                        },
                    }
                ],
            }
            angles_bytes = json.dumps(angles_doc).encode()
            artifact = {
                "artifact_id": str(uuid4()),
                "type": "angles",
                "media_type": "application/json",
                "relative_path": "angles.json",
                "sha256": hashlib.sha256(angles_bytes).hexdigest(),
                "size_bytes": len(angles_bytes),
            }
            manifest_doc = _valid_manifest(artifacts=[artifact], status="degraded")

            await ingest_analysis_run_manifest(
                created.analysisRunId,
                manifest=UploadFile(
                    file=BytesIO(json.dumps(manifest_doc).encode()),
                    filename="manifest.json",
                ),
                artifacts=[UploadFile(file=BytesIO(angles_bytes), filename="angles.json")],
                idempotency_key="angles-json-1",
                session=session,
                current_user=user,
            )

            derived_csv = tmp_path / str(created.runSessionId) / "local" / "angles.csv"
            assert derived_csv.exists()
            response = await get_run_session_angles(created.runSessionId, session=session)
            assert response["columns"] == ["left_knee_angle", "right_knee_angle"]
            assert response["samples"][0]["frame"] == 0
            assert response["samples"][0]["timeSec"] == 0.2
            assert response["samples"][0]["values"]["left_knee_angle"] == 50.0

        await engine.dispose()

    asyncio.run(scenario())


def test_local_metrics_json_is_available_through_existing_graph_api(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)
            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1, fps=60),
                session=session,
                current_user=user,
            )
            metrics_doc = {
                "schema_version": "1.0.0",
                "frames": [
                    {
                        "camera_index": 0,
                        "camera_frame": 0,
                        "source_frame": 12,
                        "absolute_frame": 0,
                        "timestamp_seconds": 0.2,
                        "distance_m": 1.2,
                        "raw_distance_m": 1.1,
                        "smooth_distance_m": 1.2,
                        "speed_mps": 7.5,
                        "acceleration_mps2": 2.1,
                        "speed_mode_used": "homography",
                        "pixel_distance_m": 1.0,
                        "pixel_speed_mps": 7.0,
                        "pixel_acceleration_mps2": 2.0,
                        "homography_distance_m": 1.2,
                        "homography_speed_mps": 7.5,
                        "homography_acceleration_mps2": 2.1,
                        "image_point_x": 320.0,
                        "image_point_y": 800.0,
                        "is_interpolated": False,
                        "interpolation_gap_length": 0,
                        "confidence": 1.0,
                    }
                ],
            }
            metrics_bytes = json.dumps(metrics_doc).encode()
            artifact = {
                "artifact_id": str(uuid4()),
                "type": "metrics",
                "media_type": "application/json",
                "relative_path": "metrics.json",
                "sha256": hashlib.sha256(metrics_bytes).hexdigest(),
                "size_bytes": len(metrics_bytes),
            }
            manifest_doc = _valid_manifest(artifacts=[artifact], status="degraded")

            await ingest_analysis_run_manifest(
                created.analysisRunId,
                manifest=UploadFile(
                    file=BytesIO(json.dumps(manifest_doc).encode()),
                    filename="manifest.json",
                ),
                artifacts=[UploadFile(file=BytesIO(metrics_bytes), filename="metrics.json")],
                idempotency_key="metrics-json-1",
                session=session,
                current_user=user,
            )

            derived_csv = tmp_path / str(created.runSessionId) / "local" / "metrics.csv"
            assert derived_csv.exists()
            rows = list(csv.DictReader(derived_csv.open()))
            assert rows[0]["dist_smooth_m"] == "1.2"
            assert rows[0]["speed_mps"] == "7.5"
            assert rows[0]["accel_mps2"] == "2.1"
            assert rows[0]["time_sec"] == "0.2"
            graphs = await get_run_session_graphs(
                created.runSessionId,
                session=session,
                current_user=user,
            )
            assert {graph["title"] for graph in graphs} == {
                "Distance", "Velocity", "Acceleration",
            }

        await engine.dispose()

    asyncio.run(scenario())


def test_degraded_local_manifest_is_preserved_as_usable_result(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)
            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1),
                session=session,
                current_user=user,
            )
            manifest_file = UploadFile(
                file=BytesIO(json.dumps(_valid_manifest(status="degraded")).encode()),
                filename="manifest.json",
            )

            result = await ingest_analysis_run_manifest(
                created.analysisRunId,
                manifest=manifest_file,
                artifacts=[],
                idempotency_key="degraded-1",
                session=session,
                current_user=user,
            )

            assert result.status == "degraded"
            run_session = await session.get(RunSession, created.runSessionId)
            assert run_session.status == "done"
            assert run_session.result_dir is not None

            session_info = await get_run_session_info(
                created.runSessionId,
                session=session,
                current_user=user,
            )
            assert session_info.computeLocations == ["local"]

        await engine.dispose()

    asyncio.run(scenario())


def test_local_steps_json_populates_trial_review(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)
            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1, fps=60),
                session=session,
                current_user=user,
            )
            steps_doc = {
                "schema_version": "1.0.0",
                "average_step_length_m": 1.2,
                "average_stride_length_m": 2.4,
                "average_cadence_spm": 240.0,
                "steps": [
                    {
                        "step_index": 1,
                        "event_type": "step",
                        "foot": "left",
                        "camera_index": 0,
                        "source_frame": 12,
                        "timestamp_seconds": 0.2,
                        "world_x_m": 1.0,
                        "world_y_m": 0.4,
                        "step_length_m": None,
                        "stride_length_m": None,
                        "contact_time_seconds": 0.1,
                        "speed_mps": 7.0,
                        "confidence": 0.9,
                    },
                    {
                        "step_index": 2,
                        "event_type": "step",
                        "foot": "right",
                        "camera_index": 0,
                        "source_frame": 27,
                        "timestamp_seconds": 0.45,
                        "world_x_m": 2.2,
                        "world_y_m": 0.4,
                        "step_length_m": 1.2,
                        "stride_length_m": None,
                        "contact_time_seconds": 0.09,
                        "speed_mps": 7.4,
                        "confidence": 0.88,
                    },
                ],
            }
            steps_bytes = json.dumps(steps_doc).encode()
            artifact = {
                "artifact_id": str(uuid4()),
                "type": "steps",
                "media_type": "application/json",
                "relative_path": "gait/steps.json",
                "sha256": hashlib.sha256(steps_bytes).hexdigest(),
                "size_bytes": len(steps_bytes),
            }
            manifest_doc = _valid_manifest(artifacts=[artifact], status="degraded")

            await ingest_analysis_run_manifest(
                created.analysisRunId,
                manifest=UploadFile(
                    file=BytesIO(json.dumps(manifest_doc).encode()),
                    filename="manifest.json",
                ),
                artifacts=[UploadFile(file=BytesIO(steps_bytes), filename="gait/steps.json")],
                idempotency_key="steps-json-1",
                session=session,
                current_user=user,
            )

            derived = tmp_path / str(created.runSessionId) / "local" / "local_step_events.csv"
            assert derived.exists()
            csv_rows = list(csv.DictReader(derived.open()))
            assert csv_rows[1]["cadence_spm"] == "240.0"

            response = Response()
            result = await get_run_session_steps(
                created.runSessionId,
                response=response,
                session=session,
            )
            assert result["avgStepLengthM"] == 1.2
            assert result["avgCadenceSpm"] == 240.0
            assert len(result["steps"]) == 2
            assert result["steps"][1]["stepLengthM"] == 1.2
            assert result["steps"][1]["cadenceSpm"] == 240.0

        await engine.dispose()

    asyncio.run(scenario())


def test_ingest_manifest_rejects_camera_count_mismatch(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)

            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=2),
                session=session,
                current_user=user,
            )

            manifest_doc = _valid_manifest(camera_count=1)  # session expects 2
            manifest_file = UploadFile(
                file=BytesIO(json.dumps(manifest_doc).encode()), filename="manifest.json"
            )

            try:
                await ingest_analysis_run_manifest(
                    created.analysisRunId,
                    manifest=manifest_file,
                    artifacts=[],
                    idempotency_key=None,
                    session=session,
                    current_user=user,
                )
                raise AssertionError("expected camera count mismatch to be rejected")
            except HTTPException as error:
                assert error.status_code == 422
                assert "camera_count" in error.detail

        await engine.dispose()

    asyncio.run(scenario())


def test_ingest_manifest_rejects_artifact_hash_mismatch(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)

            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1),
                session=session,
                current_user=user,
            )

            tampered_bytes = b"tampered content"
            declared_artifact = {
                "artifact_id": str(uuid4()),
                "type": "metrics",
                "media_type": "text/csv",
                "relative_path": "metrics.csv",
                "sha256": hashlib.sha256(b"original content").hexdigest(),
                "size_bytes": len(b"original content"),
            }
            manifest_doc = _valid_manifest(camera_count=1, artifacts=[declared_artifact])
            manifest_file = UploadFile(
                file=BytesIO(json.dumps(manifest_doc).encode()), filename="manifest.json"
            )
            artifact_file = UploadFile(file=BytesIO(tampered_bytes), filename="metrics.csv")

            try:
                await ingest_analysis_run_manifest(
                    created.analysisRunId,
                    manifest=manifest_file,
                    artifacts=[artifact_file],
                    idempotency_key=None,
                    session=session,
                    current_user=user,
                )
                raise AssertionError("expected artifact hash mismatch to be rejected")
            except HTTPException as error:
                assert error.status_code == 422
                assert "sha256" in error.detail

        await engine.dispose()

    asyncio.run(scenario())


def test_repeated_idempotency_key_replays_stored_result_without_reprocessing(tmp_path, monkeypatch):
    import routes.analysis_run as analysis_run_module

    monkeypatch.setattr(analysis_run_module, "RUN_SESSION_DIR", tmp_path)

    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)

            created = await create_local_analysis_run(
                CreateLocalAnalysisRunIn(runnerId=runner.id, cameraCount=1),
                session=session,
                current_user=user,
            )

            manifest_doc = _valid_manifest(camera_count=1)

            async def ingest_once():
                manifest_file = UploadFile(
                    file=BytesIO(json.dumps(manifest_doc).encode()), filename="manifest.json"
                )
                return await ingest_analysis_run_manifest(
                    created.analysisRunId,
                    manifest=manifest_file,
                    artifacts=[],
                    idempotency_key="same-key",
                    session=session,
                    current_user=user,
                )

            first = await ingest_once()
            second = await ingest_once()

            assert first.status == second.status == "completed"

            metas = (
                await session.execute(
                    select(AnalysisMeta).where(
                        AnalysisMeta.run_session_id == created.runSessionId
                    )
                )
            ).scalars().all()
            assert len(metas) == 1  # replay must not create a duplicate row

        await engine.dispose()

    asyncio.run(scenario())


def test_compare_pair_runs_keep_independent_result_summaries():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)

            run_session = RunSession(runner_id=runner.id, camera_count=1)
            session.add(run_session)
            await session.flush()

            comparison_group_id = uuid4()
            server_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="server",
                comparison_group_id=comparison_group_id,
                status="completed",
                result_summary={"average_speed_mps": 8.1},
            )
            local_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="local",
                comparison_group_id=comparison_group_id,
                status="completed",
                result_summary={"average_speed_mps": 7.9},
            )
            session.add_all([server_run, local_run])
            await session.commit()

            # Neither Compare run should have touched AnalysisMeta - ingestion
            # only mirrors into it for non-Compare runs (see ingest_analysis_run_manifest).
            meta = (
                await session.execute(
                    select(AnalysisMeta).where(AnalysisMeta.run_session_id == run_session.id)
                )
            ).scalars().first()
            assert meta is None

            server_reloaded = await get_analysis_run(server_run.id, session=session, current_user=user)
            local_reloaded = await get_analysis_run(local_run.id, session=session, current_user=user)
            assert server_reloaded.resultSummary["average_speed_mps"] == 8.1
            assert local_reloaded.resultSummary["average_speed_mps"] == 7.9

        await engine.dispose()

    asyncio.run(scenario())


def test_comparison_report_round_trips_and_is_idempotent_on_resubmit():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            user, runner = await _setup_runner_and_user(session)

            run_session = RunSession(runner_id=runner.id, camera_count=1)
            session.add(run_session)
            await session.flush()

            comparison_group_id = uuid4()
            server_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="server",
                comparison_group_id=comparison_group_id,
            )
            local_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="local",
                comparison_group_id=comparison_group_id,
            )
            session.add_all([server_run, local_run])
            await session.commit()

            report = {
                "schema_version": "1.0.0",
                "comparison_group_id": str(comparison_group_id),
                "request_id": str(uuid4()),
                "server_run_id": str(server_run.id),
                "local_run_id": str(local_run.id),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "input_hashes_match": True,
                "metric_differences": [],
                "performance": {
                    "server_duration_seconds": 10.0,
                    "local_duration_seconds": 12.0,
                    "local_to_server_ratio": 1.2,
                },
                "warnings": [],
            }

            await submit_comparison_report(report, session=session, current_user=user)
            # Resubmitting (e.g. after recomputing with more data) must update
            # in place, not create a second row for the same comparison group.
            await submit_comparison_report(report, session=session, current_user=user)

            rows = (await session.execute(select(ComparisonReport))).scalars().all()
            assert len(rows) == 1

            fetched = await get_comparison_report(
                comparison_group_id, session=session, current_user=user
            )
            assert fetched["status"] == "complete"

        await engine.dispose()

    asyncio.run(scenario())
