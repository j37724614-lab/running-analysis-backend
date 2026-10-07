"""Local analysis run ingestion and Compare report storage (規劃書 Step 10,
runner-pose-ondevice/report/integration_guide.md §13).

Deliberately does NOT call into `runner-analysis-pipeline` — ingestion only
accepts an already-computed result and stores it, so a Local result can never
get silently recomputed/overwritten by the Server pipeline (§7).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import RUN_SESSION_DIR
from db.session import get_session
from db_models import (
    AnalysisMeta,
    AnalysisRun,
    ComparisonReport,
    Runner,
    RunSession,
    User,
)
from response_chemas import (
    AnalysisRunOut,
    CreateLocalAnalysisRunIn,
    CreateLocalAnalysisRunOut,
)
from routes.auth import get_current_user
from utils.contract_v1 import (
    ContractValidationError,
    is_safe_relative_path,
    sha256_bytes,
    validate_comparison_report,
    validate_manifest,
)

router = APIRouter()


def _analysis_run_out(run: AnalysisRun) -> AnalysisRunOut:
    return AnalysisRunOut(
        analysisRunId=run.id,
        runSessionId=run.run_session_id,
        computeLocation=run.compute_location,
        comparisonGroupId=run.comparison_group_id,
        status=run.status,
        schemaVersion=run.schema_version,
        engineVersion=run.engine_version,
        resultSummary=run.result_summary,
        createdAt=run.created_at,
        completedAt=run.completed_at,
    )


async def _get_owned_analysis_run(
    analysis_run_id: UUID, session: AsyncSession, current_user: User
) -> AnalysisRun:
    run = (
        await session.execute(
            select(AnalysisRun).where(AnalysisRun.id == analysis_run_id)
        )
    ).scalars().first()
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    run_session = (
        await session.execute(
            select(RunSession).where(RunSession.id == run.run_session_id)
        )
    ).scalars().first()
    runner = (
        await session.execute(select(Runner).where(Runner.id == run_session.runner_id))
    ).scalars().first()
    if not runner or runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return run


@router.post("/analysis_run/local", response_model=CreateLocalAnalysisRunOut)
async def create_local_analysis_run(
    req: CreateLocalAnalysisRunIn,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    runner = (
        await session.execute(
            select(Runner)
            .where(Runner.id == req.runnerId)
            .where(Runner.user_id == current_user.id)
        )
    ).scalars().first()
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found or unauthorized")

    run_session = RunSession(
        runner_id=runner.id,
        date=req.date or datetime.now(timezone.utc),
        camera_count=req.cameraCount,
        fps=req.fps,
        is_long_jump=req.isLongJump,
        note=req.note,
        status="processing",
    )
    session.add(run_session)
    await session.flush()

    analysis_run = AnalysisRun(
        run_session_id=run_session.id,
        compute_location="local",
        comparison_group_id=req.comparisonGroupId,
        status="processing",
    )
    session.add(analysis_run)
    await session.commit()

    return CreateLocalAnalysisRunOut(
        runSessionId=run_session.id, analysisRunId=analysis_run.id
    )


@router.get("/analysis_run/{analysis_run_id}", response_model=AnalysisRunOut)
async def get_analysis_run(
    analysis_run_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    run = await _get_owned_analysis_run(analysis_run_id, session, current_user)
    return _analysis_run_out(run)


@router.post("/analysis_run/{analysis_run_id}/manifest", response_model=AnalysisRunOut)
async def ingest_analysis_run_manifest(
    analysis_run_id: UUID,
    manifest: UploadFile = File(...),
    artifacts: List[UploadFile] = File(default=[]),
    input_videos: Optional[List[UploadFile]] = File(default=None),
    idempotency_key: Optional[str] = Form(default=None),
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    run = await _get_owned_analysis_run(analysis_run_id, session, current_user)

    if (
        idempotency_key
        and run.idempotency_key == idempotency_key
        and run.status in ("completed", "failed")
    ):
        # Replay: a retried request after a dropped response must not reprocess
        # or create duplicate artifacts/rows.
        return _analysis_run_out(run)

    manifest_bytes = await manifest.read()
    try:
        manifest_doc = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail=f"manifest is not valid JSON: {error}")

    try:
        validate_manifest(manifest_doc)
    except ContractValidationError as error:
        run.status = "failed"
        run.idempotency_key = idempotency_key
        await session.commit()
        raise HTTPException(status_code=422, detail=str(error))

    if manifest_doc["compute_location"] != run.compute_location:
        raise HTTPException(
            status_code=422,
            detail=(
                f"manifest.compute_location ({manifest_doc['compute_location']!r}) does "
                f"not match this analysis run ({run.compute_location!r})"
            ),
        )

    run_session = (
        await session.execute(select(RunSession).where(RunSession.id == run.run_session_id))
    ).scalars().first()
    if len(manifest_doc["input_videos"]) != run_session.camera_count:
        raise HTTPException(
            status_code=422,
            detail=(
                f"manifest declares {len(manifest_doc['input_videos'])} input videos but "
                f"this session expects {run_session.camera_count} (camera_count)"
            ),
        )

    # Direct unit calls see FastAPI's File() marker when this optional argument
    # is omitted; HTTP requests supply an actual list.
    uploaded_videos = input_videos if isinstance(input_videos, list) else []
    if uploaded_videos and len(uploaded_videos) != run_session.camera_count:
        raise HTTPException(
            status_code=422,
            detail=(
                f"received {len(uploaded_videos)} input videos but this session expects "
                f"{run_session.camera_count} (camera_count)"
            ),
        )

    artifacts_by_relative_path = {}
    for upload in artifacts:
        data = await upload.read()
        artifacts_by_relative_path[upload.filename] = data

    declared_artifacts = manifest_doc.get("artifacts", [])
    for declared in declared_artifacts:
        relative_path = declared["relative_path"]
        if not is_safe_relative_path(relative_path):
            raise HTTPException(
                status_code=422, detail=f"unsafe artifact relative_path: {relative_path!r}"
            )
        uploaded_bytes = artifacts_by_relative_path.get(relative_path)
        if uploaded_bytes is None:
            raise HTTPException(
                status_code=422,
                detail=f"manifest declares artifact {relative_path!r} but it was not uploaded",
            )
        actual_sha256 = sha256_bytes(uploaded_bytes)
        if actual_sha256 != declared["sha256"]:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"artifact {relative_path!r} content does not match the sha256 "
                    "declared in the manifest"
                ),
            )
        if len(uploaded_bytes) != declared["size_bytes"]:
            raise HTTPException(
                status_code=422,
                detail=f"artifact {relative_path!r} size does not match the manifest",
            )

    result_dir = RUN_SESSION_DIR / str(run_session.id) / run.compute_location
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "manifest.json").write_bytes(manifest_bytes)
    for declared in declared_artifacts:
        relative_path = declared["relative_path"]
        artifact_path = result_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(artifacts_by_relative_path[relative_path])

    # Server uploads already use this canonical directory. Keeping Local input
    # videos in the same layout lets the playback endpoint serve them too.
    video_dir = RUN_SESSION_DIR / str(run_session.runner_id) / str(run_session.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    for index, (upload, declared) in enumerate(
        zip(uploaded_videos, manifest_doc["input_videos"]), start=1
    ):
        suffix = Path(upload.filename or "").suffix
        if not suffix or len(suffix) > 10 or not suffix[1:].isalnum():
            suffix = ".mp4"
        destination = video_dir / f"cam{index}{suffix}"
        partial = destination.with_suffix(f"{destination.suffix}.part")
        digest = hashlib.sha256()
        with partial.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != declared["sha256"]:
            partial.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail=f"input video {index} content does not match the manifest sha256",
            )
        partial.replace(destination)

    manifest_status = manifest_doc["status"]
    run.status = manifest_status if manifest_status in ("completed", "degraded") else "failed"
    run.schema_version = manifest_doc["schema_version"]
    run.engine_version = manifest_doc["engine"]["version"]
    run.result_summary = manifest_doc["summary"]
    run.timings = {"stages": manifest_doc.get("stages", [])}
    run.idempotency_key = idempotency_key
    run.completed_at = datetime.now(timezone.utc)
    run_session.result_dir = str(result_dir)
    run_session.status = "done" if run.status in ("completed", "degraded") else "failed"

    # Only promote this run's summary to the session's single display row
    # (AnalysisMeta) when there is no Compare ambiguity about which run owns
    # it - a RunSession either has one non-Compare run, or a Compare pair
    # whose two summaries must stay independently readable from each
    # `AnalysisRun.result_summary` instead.
    if run.comparison_group_id is None and run.status in ("completed", "degraded"):
        summary = manifest_doc["summary"]
        existing_meta = (
            await session.execute(
                select(AnalysisMeta).where(AnalysisMeta.run_session_id == run_session.id)
            )
        ).scalars().first()
        if existing_meta is None:
            session.add(
                AnalysisMeta(
                    run_session_id=run_session.id,
                    total_time=summary.get("total_time_seconds"),
                    avg_velocity=summary.get("average_speed_mps"),
                    avg_acceleration=summary.get("average_acceleration_mps2"),
                    avg_step_length=summary.get("average_step_length_m"),
                    summary=summary,
                )
            )
        else:
            existing_meta.total_time = summary.get("total_time_seconds")
            existing_meta.avg_velocity = summary.get("average_speed_mps")
            existing_meta.avg_acceleration = summary.get("average_acceleration_mps2")
            existing_meta.avg_step_length = summary.get("average_step_length_m")
            existing_meta.summary = summary

    await session.commit()
    await session.refresh(run)
    return _analysis_run_out(run)


@router.post("/comparison_report")
async def submit_comparison_report(
    report: dict,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    try:
        validate_comparison_report(report)
    except ContractValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))

    server_run_id = UUID(report["server_run_id"])
    local_run_id = UUID(report["local_run_id"])
    # Ownership: both referenced runs must belong to the current user.
    await _get_owned_analysis_run(server_run_id, session, current_user)
    await _get_owned_analysis_run(local_run_id, session, current_user)

    comparison_group_id = UUID(report["comparison_group_id"])
    existing = (
        await session.execute(
            select(ComparisonReport).where(
                ComparisonReport.comparison_group_id == comparison_group_id
            )
        )
    ).scalars().first()
    if existing:
        existing.status = report["status"]
        existing.input_hashes_match = report["input_hashes_match"]
        existing.payload = report
    else:
        session.add(
            ComparisonReport(
                comparison_group_id=comparison_group_id,
                server_run_id=server_run_id,
                local_run_id=local_run_id,
                status=report["status"],
                input_hashes_match=report["input_hashes_match"],
                payload=report,
            )
        )
    await session.commit()
    return report


@router.get("/comparison_report/{comparison_group_id}")
async def get_comparison_report(
    comparison_group_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    stored = (
        await session.execute(
            select(ComparisonReport).where(
                ComparisonReport.comparison_group_id == comparison_group_id
            )
        )
    ).scalars().first()
    if not stored:
        raise HTTPException(status_code=404, detail="Comparison report not found")
    # Ownership check via either referenced run (both share the same owner by construction).
    await _get_owned_analysis_run(stored.server_run_id, session, current_user)
    return stored.payload
