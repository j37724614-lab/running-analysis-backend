import asyncio
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from db_models import AnalysisMeta, AnalysisRun, ComparisonReport, Runner, RunSession, User
from response_chemas import CreateLocalAnalysisRunIn
from routes.analysis_run import (
    create_local_analysis_run,
    get_analysis_run,
    get_comparison_report,
    ingest_analysis_run_manifest,
    submit_comparison_report,
)
from routes.run import get_run_session_info

JOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
    "right_heel",
]


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
