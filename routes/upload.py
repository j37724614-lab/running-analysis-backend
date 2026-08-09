import os
import shutil
import uuid
import asyncio
import glob
import json
import sys
from datetime import datetime
from pathlib import Path
from uuid import UUID

import cv2
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session, async_session
from db_models import RunSession, AnalysisMeta, Video, User, Runner
from routes.auth import get_current_user
from config import ENABLE_MOCK_ON_FAILURE, PIPELINE_ROOT, RUN_SESSION_DIR, TEMP_UPLOAD_DIR
from response_chemas import (
    UploadSeperatelyStatus,
    UploadSeperatelyNewRequest,
    UploadSeperatelySelectRequest,
    UploadAllRequest,
)

pipeline_dir = str(PIPELINE_ROOT)
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)
from core.pipeline import run_analysis


router = APIRouter()


def _camera_config_from_anchors(video_path: str, anchors: list[dict], top_distance_m, bottom_distance_m) -> dict:
    cam_cfg = {"video_path": video_path}
    cap = cv2.VideoCapture(video_path)
    try:
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            w = h = 0
    finally:
        cap.release()

    if w > 0 and h > 0:
        points = [
            [float(anchor["x"]) * float(w), float(anchor["y"]) * float(h)]
            for anchor in anchors
        ]
    else:
        points = [[float(anchor["x"]), float(anchor["y"])] for anchor in anchors]

    # Use pixel projection between the two anchor lines for meter conversion.
    # Homography can distort step length for off-line ankle points, so step
    # analysis intentionally relies on this line scale.
    cam_cfg["start_line"] = [points[0], points[3]]
    cam_cfg["end_line"] = [points[1], points[2]]

    distance_candidates = [
        float(value)
        for value in (top_distance_m, bottom_distance_m)
        if value is not None and float(value) > 0
    ]
    if distance_candidates:
        distance_m = sum(distance_candidates) / len(distance_candidates)
        cam_cfg["distance_m"] = distance_m

    return cam_cfg


def _camera_config_from_homography_anchors(video_path: str, anchors: list[dict]) -> dict:
    """Build a homography-mode cam_cfg from 6 ground-control points.

    Each anchor carries its own world_x_m/world_y_m (unlike the 4-point
    line-projection mode above). The pipeline's own _track_calibration()
    already knows how to turn homography_src_points/homography_dst_world
    into a full 2D homography (see ankle_step_stride.py), so this only
    needs to assemble those two lists -- no cv2 call here.
    """
    cam_cfg = {"video_path": video_path}
    cap = cv2.VideoCapture(video_path)
    try:
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            w = h = 0
    finally:
        cap.release()

    if w > 0 and h > 0:
        src_points = [
            [float(anchor["x"]) * float(w), float(anchor["y"]) * float(h)]
            for anchor in anchors
        ]
    else:
        src_points = [[float(anchor["x"]), float(anchor["y"])] for anchor in anchors]

    cam_cfg["homography_src_points"] = src_points
    cam_cfg["homography_dst_world"] = [
        [float(anchor["world_x_m"]), float(anchor["world_y_m"])] for anchor in anchors
    ]

    # Synthesize a start_line/end_line pair from the two points nearest/furthest
    # along the runway (by world_x_m), so core/tracking.py's existing
    # start_line/end_line-driven logic -- temporal prescan's valid frame-range
    # ROI, "stop output past the end line", etc. -- works the same way it does
    # for the 4-point line-projection mode. distance_m is deliberately left
    # unset so _track_calibration() still falls through to the homography
    # distance calculation for step lengths (see _make_calibration()).
    by_world_x = sorted(zip(anchors, src_points), key=lambda pair: pair[0]["world_x_m"])
    min_x = by_world_x[0][0]["world_x_m"]
    max_x = by_world_x[-1][0]["world_x_m"]
    start_pts = [pt for anchor, pt in by_world_x if anchor["world_x_m"] == min_x]
    end_pts = [pt for anchor, pt in by_world_x if anchor["world_x_m"] == max_x]
    if len(start_pts) >= 2 and len(end_pts) >= 2 and min_x != max_x:
        cam_cfg["start_line"] = start_pts[:2]
        cam_cfg["end_line"] = end_pts[:2]

    return cam_cfg


def move_temp_video_and_del_thumbnail(temp_video_id: str, runner_id: str, run_session_id: str, camera_index: int):
    image_path = os.path.join(TEMP_UPLOAD_DIR, temp_video_id + ".jpg")
    if os.path.exists(image_path):
        os.remove(image_path)

    matches = glob.glob(
        os.path.join(TEMP_UPLOAD_DIR, temp_video_id + "*")
    )
    matches = [f for f in matches if not f.endswith(".jpg")]

    if not matches:
        raise HTTPException(404, f"Temperature video file not found for ID: {temp_video_id}")

    temp_video_path = matches[0]
    ext = os.path.splitext(temp_video_path)[1]

    dest_dir = os.path.join(RUN_SESSION_DIR, runner_id, run_session_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"cam{camera_index + 1}{ext}")
    shutil.move(temp_video_path, dest_path)
    return dest_path


async def analyze_and_save(runner_id: str, run_session_id: str, camera_count: int):
    async with async_session() as session:
        try:
            folder = os.path.join(RUN_SESSION_DIR, runner_id, run_session_id)
            run_session = await session.get(RunSession, UUID(run_session_id))
            if not run_session:
                print(f"RunSession {run_session_id} not found")
                return

            run_session.status = "processing"
            run_session.progress = 0
            await session.commit()

            loop = asyncio.get_running_loop()

            def progress_callback(p):
                async def update_db():
                    async with async_session() as sess:
                        stmt = update(RunSession).where(RunSession.id == UUID(run_session_id)).values(progress=p)
                        await sess.execute(stmt)
                        await sess.commit()
                asyncio.run_coroutine_threadsafe(update_db(), loop)

            stmt = select(Video).where(Video.run_session_id == UUID(run_session_id))
            result = await session.execute(stmt)
            videos = result.scalars().all()

            config_dict = {
                "cameras": [],
                "auto_crop": True,
                "tracking_mode": "two_pass",
                "prescan_enabled": True,
                "prescan_engine_path": "/home/jeter/runner-analysis-pipeline/models/yolo26x_ultralytics_int8.engine",
            }
            if run_session.is_long_jump:
                config_dict["long_jump_final_landing"] = True
            videos_sorted = sorted(videos, key=lambda x: x.camera_index)
            meta_data_cameras = []

            for v in videos_sorted:
                anchors = json.loads(v.anchors) if v.anchors else None
                cam_cfg = {"video_path": v.video_path}
                if anchors and len(anchors) == 4:
                    cam_cfg = _camera_config_from_anchors(
                        v.video_path,
                        anchors,
                        v.top_distance_m,
                        v.bottom_distance_m,
                    )
                elif anchors and len(anchors) == 6 and all(
                    a.get("world_x_m") is not None and a.get("world_y_m") is not None
                    for a in anchors
                ):
                    cam_cfg = _camera_config_from_homography_anchors(v.video_path, anchors)
                config_dict["cameras"].append(cam_cfg)

                meta_data_cameras.append({
                    "camera_index": v.camera_index,
                    "anchors": anchors,
                    "top_distance_m": v.top_distance_m,
                    "bottom_distance_m": v.bottom_distance_m,
                })

            # Use the calibrated world-coordinate speed when every analysed
            # camera has six valid ground-control points. The pipeline still
            # writes the legacy pixel series alongside it for comparison.
            if config_dict["cameras"] and all(
                len(cam.get("homography_src_points", [])) == 6
                and len(cam.get("homography_dst_world", [])) == 6
                for cam in config_dict["cameras"]
            ):
                config_dict["speed_mode"] = "homography"
            else:
                config_dict["speed_mode"] = "pixel"

            meta_data = {
                "run_session_id": run_session_id,
                "camera_count": camera_count,
                "cameras": meta_data_cameras
            }

            with open(os.path.join(folder, "metadata.json"), "w") as f:
                json.dump(meta_data, f, indent=4)

            raw_data = await asyncio.to_thread(
                run_analysis,
                config_dict=config_dict,
                gpu="0",
                only_2d=False,
                skip_track=False,
                output_dest=folder,
                progress_callback=progress_callback,
            )

            metrics_csv = raw_data.get("metrics_csv") if raw_data else None
            total_time = raw_data.get("total_time") if raw_data else None
            avg_velocity = raw_data.get("avg_velocity") if raw_data else None
            avg_acceleration = raw_data.get("avg_acceleration") if raw_data else None
            avg_step_length = raw_data.get("avg_step_length") if raw_data else None
            if avg_step_length is None:
                avg_step_length = 0.0

            summary = {
                "metrics_csv": metrics_csv,
                "angles_csv": raw_data.get("angles_csv") if raw_data else None,
                "uncropped_video": raw_data.get("uncropped_video") if raw_data else None,
            }
            analysis_meta = await session.get(AnalysisMeta, UUID(run_session_id))
            if analysis_meta is None:
                analysis_meta = AnalysisMeta(run_session_id=UUID(run_session_id))
                session.add(analysis_meta)
            analysis_meta.total_time = total_time
            analysis_meta.avg_velocity = avg_velocity
            analysis_meta.avg_acceleration = avg_acceleration
            analysis_meta.avg_step_length = avg_step_length
            analysis_meta.summary = summary
            run_session = await session.get(RunSession, UUID(run_session_id))
            run_session.status = "done"
            run_session.progress = 100

            await session.commit()
        except Exception as e:
            print(f"Error during analysis for session {run_session_id}: {e}")
            await session.rollback()

            if ENABLE_MOCK_ON_FAILURE:
                print(f"Falling back to mock data for session {run_session_id}")
                try:
                    analysis_meta = AnalysisMeta(
                        run_session_id=UUID(run_session_id),
                        total_time=10.0,
                        avg_velocity=5.0,
                        avg_acceleration=0.5,
                        avg_step_length=1.2,
                        summary={"mock": "True", "error": str(e)},
                    )
                    session.add(analysis_meta)

                    run_session = await session.get(RunSession, UUID(run_session_id))
                    if run_session:
                        run_session.status = "done"
                        run_session.progress = 100
                        await session.commit()
                        print(f"Set session {run_session_id} status to done (Mock Data)")
                except Exception as inner_e:
                    print(f"Failed to use mock data for session {run_session_id}: {inner_e}")
            else:
                try:
                    run_session = await session.get(RunSession, UUID(run_session_id))
                    if run_session:
                        run_session.status = "failed"
                        await session.commit()
                        print(f"Set session {run_session_id} status to failed")
                except Exception as inner_e:
                    print(f"Failed to set status to failed for session {run_session_id}: {inner_e}")


@router.post("/temp_video/{index}")
async def upload_video(
    index: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    base_id = uuid.uuid4().hex[:8]
    temp_video_id = f"{base_id}_cam{index + 1}"

    video_path = os.path.join(TEMP_UPLOAD_DIR, temp_video_id + Path(file.filename).suffix.lower())
    image_path = os.path.join(TEMP_UPLOAD_DIR, temp_video_id + ".jpg")

    with open(video_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        os.remove(video_path)
        raise HTTPException(400, "Failed to extract first frame")

    cv2.imwrite(image_path, frame)

    return {"tempVideoId": temp_video_id}


@router.post("/upload_all_info")
async def upload_all_info(
    req: UploadAllRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    # Verify runner belongs to current user
    runner = (await session.execute(
        select(Runner).where(Runner.id == UUID(req.runnerId)).where(Runner.user_id == current_user.id)
    )).scalars().first()
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found or unauthorized")

    runSession = RunSession(
        runner_id=UUID(req.runnerId),
        date=datetime.strptime(req.date, "%Y-%m-%d %H:%M:%S"),
        fps=req.fps,
        camera_count=req.cameraCount,
        note=req.note,
        is_long_jump=req.isLongJump,
    )
    session.add(runSession)

    await session.commit()
    await session.refresh(runSession)

    for cameraIndex, info in enumerate(req.videos):
        tempVideoId = info.tempVideoId
        video_stored_path = move_temp_video_and_del_thumbnail(
            tempVideoId,
            req.runnerId,
            str(runSession.id),
            cameraIndex,
        )

        anchors_json = None
        top_d = None
        bot_d = None
        if info.anchors:
            anchors_json = json.dumps([p.dict() for p in info.anchors.points])
            top_d = info.anchors.topDistanceM
            bot_d = info.anchors.bottomDistanceM

        video = Video(
            run_session_id=runSession.id,
            camera_index=cameraIndex,
            video_path=video_stored_path,
            anchors=anchors_json,
            top_distance_m=top_d,
            bottom_distance_m=bot_d,
        )
        session.add(video)
        await session.commit()
        await session.refresh(video)

    asyncio.create_task(analyze_and_save(req.runnerId, str(runSession.id), req.cameraCount))

    return {"runSessionId": runSession.id}


@router.post("/upload_seperately_new")
async def upload_seperately_new(
    req: UploadSeperatelyNewRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> UploadSeperatelyStatus:
    # Verify runner belongs to current user
    runner = (await session.execute(
        select(Runner).where(Runner.id == UUID(req.runnerId)).where(Runner.user_id == current_user.id)
    )).scalars().first()
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found or unauthorized")

    runSession = RunSession(
        runner_id=UUID(req.runnerId),
        date=datetime.strptime(req.date, "%Y-%m-%d %H:%M:%S"),
        fps=req.fps,
        camera_count=req.cameraCount,
        note=req.note,
        is_long_jump=req.isLongJump,
    )
    session.add(runSession)

    await session.commit()
    await session.refresh(runSession)

    videoStoredPath = move_temp_video_and_del_thumbnail(
        req.tempVideoId,
        req.runnerId,
        str(runSession.id),
        req.cameraIndex,
    )

    anchors_json = None
    top_d = None
    bot_d = None
    if req.anchors:
        anchors_json = json.dumps([p.dict() for p in req.anchors.points])
        top_d = req.anchors.topDistanceM
        bot_d = req.anchors.bottomDistanceM

    video = Video(
        run_session_id=runSession.id,
        camera_index=req.cameraIndex,
        video_path=videoStoredPath,
        anchors=anchors_json,
        top_distance_m=top_d,
        bottom_distance_m=bot_d,
    )
    session.add(video)
    await session.commit()
    await session.refresh(video)

    if req.cameraCount == 1:
        asyncio.create_task(analyze_and_save(req.runnerId, str(runSession.id), req.cameraCount))
        return UploadSeperatelyStatus(
            runnerId=req.runnerId,
            runSessionId=str(runSession.id),
            isAllUploaded=True,
            unuploadedCameraIndexes=[],
        )

    unuploadedCameraIndexes = [i for i in range(req.cameraCount) if i != req.cameraIndex]
    return UploadSeperatelyStatus(
        runnerId=req.runnerId,
        runSessionId=str(runSession.id),
        isAllUploaded=False,
        unuploadedCameraIndexes=unuploadedCameraIndexes,
    )


@router.post("/upload_seperately_select")
async def upload_seperately_select(
    req: UploadSeperatelySelectRequest,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> UploadSeperatelyStatus:
    # Verify runner belongs to current user
    runner = (await session.execute(
        select(Runner).where(Runner.id == UUID(req.runnerId)).where(Runner.user_id == current_user.id)
    )).scalars().first()
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found or unauthorized")

    stored_path = move_temp_video_and_del_thumbnail(
        req.tempVideoId,
        req.runnerId,
        req.runSessionId,
        req.cameraIndex,
    )

    anchors_json = None
    top_d = None
    bot_d = None
    if req.anchors:
        anchors_json = json.dumps([p.dict() for p in req.anchors.points])
        top_d = req.anchors.topDistanceM
        bot_d = req.anchors.bottomDistanceM

    video = Video(
        run_session_id=UUID(req.runSessionId),
        camera_index=req.cameraIndex,
        video_path=stored_path,
        anchors=anchors_json,
        top_distance_m=top_d,
        bottom_distance_m=bot_d,
    )
    session.add(video)
    await session.commit()
    await session.refresh(video)

    runSession = await session.get(RunSession, UUID(req.runSessionId))

    stmt = select(Video.camera_index).where(
        Video.run_session_id == UUID(req.runSessionId)
    )
    result = await session.execute(stmt)
    uploaded_indexes = {row[0] for row in result.all()}

    expected_indexes = set(range(runSession.camera_count))
    unuploadedCameraIndexes = sorted(expected_indexes - uploaded_indexes)

    isAllUploaded = len(unuploadedCameraIndexes) == 0
    if isAllUploaded:
        asyncio.create_task(analyze_and_save(req.runnerId, req.runSessionId, runSession.camera_count))

    return UploadSeperatelyStatus(
        runnerId=req.runnerId,
        runSessionId=req.runSessionId,
        isAllUploaded=isAllUploaded,
        unuploadedCameraIndexes=unuploadedCameraIndexes,
    )
