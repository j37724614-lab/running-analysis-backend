import os
from uuid import UUID
from pathlib import Path
from html import escape

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import numpy as np
import pandas as pd

from config import RUN_SESSION_DIR, TEMP_UPLOAD_DIR
from db.session import get_session
from db_models import Runner, RunSession, User
from routes.auth import get_current_user
from response_chemas import (
    AddRunnerIn,
    AnglesOut,
    GraphOut,
    RunnerInfoOut,
    RunSessionInfoOut,
    UnanalyzedRunSessionInfoOut,
)
from utils.report_generator import generate_pdf_report

router = APIRouter()


def _compute_locations(run: RunSession) -> list[str]:
    locations = sorted({analysis_run.compute_location for analysis_run in run.analysis_runs})
    if locations:
        return locations
    return ["server"] if run.analysis is not None else []


def _unanalyzed_session_out(run: RunSession) -> UnanalyzedRunSessionInfoOut:
    uploaded = {video.camera_index for video in run.videos}
    missing = [index for index in range(run.camera_count) if index not in uploaded]
    video_paths: list[str | None] = [None] * run.camera_count
    for video in run.videos:
        if 0 <= video.camera_index < run.camera_count:
            video_paths[video.camera_index] = video.video_path

    return UnanalyzedRunSessionInfoOut(
        runSessionId=run.id,
        runnerId=run.runner_id,
        runnerName=run.runner.name,
        date=run.date,
        cameraCount=run.camera_count,
        fps=run.fps,
        note=run.note,
        unuploadedCameraIndexes=missing,
        videoPaths=video_paths,
    )


@router.get("/runner", response_model=list[RunnerInfoOut])
async def get_runners(
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user)
) -> list[RunnerInfoOut]:
    runners = (await session.execute(
        select(Runner).where(Runner.user_id == current_user.id)
    )).scalars().all()
    result = []

    for runner in runners:
        last_run = (
            await session.execute(
                select(RunSession)
                .where(RunSession.runner_id == runner.id)
                .where(RunSession.status != "pending")
                .order_by(RunSession.date.desc())
                .limit(1)
            )
        ).scalars().first()

        result.append(
            RunnerInfoOut(
                id=runner.id,
                name=runner.name,
                lastVideoId=last_run.id if last_run else None,
            )
        )
    return result


@router.post("/runner")
async def add_runner(
    data: AddRunnerIn,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> dict:
    runner = Runner(name=data.name, user_id=current_user.id)
    session.add(runner)
    await session.commit()
    await session.refresh(runner)
    return {"id": str(runner.id)}


@router.get(
    "/runner/{runner_id}/run_sessions",
    response_model=list[RunSessionInfoOut],
)
async def get_runner_run_sessions(
    runner_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[RunSessionInfoOut]:
    runner = (await session.execute(
        select(Runner).where(Runner.id == runner_id).where(Runner.user_id == current_user.id)
    )).scalars().first()
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found or unauthorized")

    runs = (await session.execute(
        select(RunSession)
        .where(RunSession.runner_id == runner_id)
        .where(RunSession.status != "pending")
        .options(
            selectinload(RunSession.runner),
            selectinload(RunSession.analysis),
            selectinload(RunSession.analysis_runs),
        )
    )).scalars().all()

    result = []
    for run in runs:
        result.append(
            RunSessionInfoOut(
                runSessionId=run.id,
                runnerId=runner_id,
                runnerName=run.runner.name,
                date=run.date,
                cameraCount=run.camera_count,
                fps=run.fps,
                note=run.note,
                avgVelocity=round(run.analysis.avg_velocity, 3) if run.analysis and run.analysis.avg_velocity is not None else None,
                avgAcceleration=round(run.analysis.avg_acceleration, 3) if run.analysis and run.analysis.avg_acceleration is not None else None,
                avgStepLength=round(run.analysis.avg_step_length, 3) if run.analysis and run.analysis.avg_step_length is not None else None,
                totalTime=round(run.analysis.total_time, 3) if run.analysis and run.analysis.total_time is not None else None,
                status=run.status,
                progress=run.progress,
                isLongJump=run.is_long_jump,
                computeLocations=_compute_locations(run),
            )
        )
    return result


@router.get(
    "/runner/{runner_id}/run_sessions/unanalyzed",
    response_model=list[UnanalyzedRunSessionInfoOut],
)
async def get_unanalyzed_run_sessions(
    runner_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> list[UnanalyzedRunSessionInfoOut]:
    runner = (await session.execute(
        select(Runner).where(Runner.id == runner_id).where(Runner.user_id == current_user.id)
    )).scalars().first()
    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found or unauthorized")

    runs = (await session.execute(
        select(RunSession)
        .where(RunSession.runner_id == runner_id)
        .where(RunSession.status == "pending")
        .options(
            selectinload(RunSession.runner),
            selectinload(RunSession.videos),
        )
    )).scalars().all()

    result = []
    for run in runs:
        uploaded = {v.camera_index for v in run.videos}
        missing = [i for i in range(run.camera_count) if i not in uploaded]

        result.append(
            UnanalyzedRunSessionInfoOut(
                runSessionId=run.id,
                runnerId=runner_id,
                runnerName=run.runner.name,
                date=run.date,
                cameraCount=run.camera_count,
                fps=run.fps,
                note=run.note,
                unuploadedCameraIndexes=missing,
                videoPaths=[v.video_path if v else None for v in run.videos],
            )
        )
    return result


@router.get("/run_session/{run_session_id}")
async def get_run_session_info(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user)
) -> RunSessionInfoOut:
    run_session = (await session.execute(
        select(RunSession)
        .where(RunSession.id == run_session_id)
        .options(
            selectinload(RunSession.runner),
            selectinload(RunSession.analysis),
            selectinload(RunSession.analysis_runs),
        )
    )).scalars().first()
    if not run_session or run_session.runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run session not found or unauthorized")

    if run_session.status != "done" or run_session.analysis is None:
        return RunSessionInfoOut(
            runSessionId=run_session_id,
            runnerId=run_session.runner_id,
            runnerName=run_session.runner.name,
            date=run_session.date,
            cameraCount=run_session.camera_count,
            fps=run_session.fps,
            note=run_session.note,
            status=run_session.status,
            progress=run_session.progress,
            isLongJump=run_session.is_long_jump,
            computeLocations=_compute_locations(run_session),
        )

    analysis = run_session.analysis
    return RunSessionInfoOut(
        runSessionId=run_session_id,
        runnerId=run_session.runner_id,
        runnerName=run_session.runner.name,
        date=run_session.date,
        cameraCount=run_session.camera_count,
        fps=run_session.fps,
        note=run_session.note,
        status=run_session.status,
        progress=run_session.progress,
        isLongJump=run_session.is_long_jump,
        computeLocations=_compute_locations(run_session),
        totalTime=round(analysis.total_time, 3) if analysis.total_time is not None else None,
        avgVelocity=round(analysis.avg_velocity, 3) if analysis.avg_velocity is not None else None,
        avgAcceleration=round(analysis.avg_acceleration, 3) if analysis.avg_acceleration is not None else None,
        avgStepLength=round(analysis.avg_step_length, 3) if analysis.avg_step_length is not None else None,
    )


def _finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[list[float], list[float]]:
    cleaned = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned[x_col].astype(float).tolist(), cleaned[y_col].astype(float).tolist()


def _sample(values: list[float], max_points: int) -> list[float]:
    if len(values) <= max_points:
        return values
    step = max(1, len(values) // max_points)
    return values[::step]


@router.delete("/run_session/{run_session_id}")
async def delete_run_session(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    import shutil
    from db_models import AnalysisMeta

    run_session = (await session.execute(
        select(RunSession)
        .where(RunSession.id == run_session_id)
        .options(
            selectinload(RunSession.runner),
            selectinload(RunSession.videos)
        )
    )).scalars().first()

    if not run_session or run_session.runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run session not found or unauthorized")

    # 1. Delete original camera video files on disk
    for video in run_session.videos:
        if video.video_path and os.path.exists(video.video_path):
            try:
                os.remove(video.video_path)
            except Exception:
                pass

    # 2. Delete result directory on disk (which has CSVs, generated charts, and report PDF)
    if run_session.result_dir and os.path.exists(run_session.result_dir):
        try:
            shutil.rmtree(run_session.result_dir)
        except Exception:
            pass

    # 3. Clean up database entries
    # Delete related analysis_meta row
    analysis = (await session.execute(
        select(AnalysisMeta).where(AnalysisMeta.run_session_id == run_session_id)
    )).scalars().first()
    if analysis:
        await session.delete(analysis)

    # Delete related video rows
    for video in run_session.videos:
        await session.delete(video)

    # Delete the run session row
    await session.delete(run_session)

    await session.commit()
    return {"status": "success", "message": "Run session deleted successfully"}

@router.delete("/runner/{runner_id}")
async def delete_runner(
    runner_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    import shutil
    from db_models import RunSession, AnalysisMeta, Video

    # 1. Fetch runner
    runner = (await session.execute(
        select(Runner).where(Runner.id == runner_id).where(Runner.user_id == current_user.id)
    )).scalars().first()

    if not runner:
        raise HTTPException(status_code=404, detail="Runner not found")

    # 2. Fetch all run sessions for this runner
    runs = (await session.execute(
        select(RunSession)
        .where(RunSession.runner_id == runner_id)
        .options(selectinload(RunSession.videos))
    )).scalars().all()

    # 3. For each run session, delete files and DB child entries
    for run in runs:
        # Delete original camera video files on disk
        for video in run.videos:
            if video.video_path and os.path.exists(video.video_path):
                try:
                    os.remove(video.video_path)
                except Exception:
                    pass

        # Delete result directory on disk
        if run.result_dir and os.path.exists(run.result_dir):
            try:
                shutil.rmtree(run.result_dir)
            except Exception:
                pass

        # Delete related analysis_meta row
        analysis = (await session.execute(
            select(AnalysisMeta).where(AnalysisMeta.run_session_id == run.id)
        )).scalars().first()
        if analysis:
            await session.delete(analysis)

        # Delete related video rows
        for video in run.videos:
            await session.delete(video)

        # Delete the run session row
        await session.delete(run)

    # 4. Delete the runner row
    await session.delete(runner)

    await session.commit()
    return {"status": "success", "message": "Runner and all associated sessions deleted successfully"}


def build_graph(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    y_label: str,
    category: str = "metrics",
    max_points: int = 200,
):
    x, y = _finite_xy(df, x_col, y_col)
    x = _sample(x, max_points)
    y = _sample(y, max_points)

    if not y:
        x = [0.0]
        y = [0.0]

    return {
        "title": title,
        "yLabel": y_label,
        "x": x,
        "yMin": float(np.min(y)),
        "yMax": float(np.max(y)),
        "series": [{"name": "main", "y": y}],
        "category": category,
    }


def build_multi_graph(
    df: pd.DataFrame,
    x_col: str,
    columns: list[str],
    title: str,
    y_label: str,
    category: str = "angles",
    max_points: int = 200,
):
    cleaned = df[[x_col] + columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col])
    x = _sample(cleaned[x_col].astype(float).tolist(), max_points)
    series = []
    all_y = []

    for col in columns:
        y = cleaned[col].fillna(0).astype(float).tolist()
        y = _sample(y, max_points)
        series.append({"name": col.replace("_angle", "").replace("_", " "), "y": y})
        all_y.extend(y)

    if not all_y:
        x = [0.0]
        all_y = [0.0]

    return {
        "title": title,
        "yLabel": y_label,
        "x": x,
        "yMin": float(np.min(all_y)),
        "yMax": float(np.max(all_y)),
        "series": series,
        "category": category,
    }


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


async def _get_run_session(
    run_session_id: UUID,
    session: AsyncSession,
    load_analysis: bool = False,
) -> RunSession:
    query = select(RunSession).where(RunSession.id == run_session_id)
    if load_analysis:
        query = query.options(
            selectinload(RunSession.runner),
            selectinload(RunSession.analysis),
        )
    run_session = (await session.execute(query)).scalars().first()
    if not run_session:
        raise HTTPException(status_code=404, detail="Run session not found")
    return run_session


def _session_dir(run_session: RunSession) -> Path:
    if run_session.result_dir:
        result_dir = Path(run_session.result_dir)
        if result_dir.exists():
            return result_dir
    return RUN_SESSION_DIR / str(run_session.runner_id) / str(run_session.id)


def _analysis_csvs(run_session: RunSession) -> tuple[Path | None, Path | None]:
    session_dir = _session_dir(run_session)
    metrics_csv = _first_existing(
        [
            session_dir / "metrics.csv",
            session_dir / "tracking_results" / "run_metrics.csv",
        ]
    )
    angles_csv = _first_existing(
        [
            session_dir / "angles.csv",
            session_dir / "tracking_results" / "joint_angles.csv",
        ]
    )
    return metrics_csv, angles_csv


def _with_time_column(df: pd.DataFrame, fps: int) -> pd.DataFrame:
    if "time_sec" not in df.columns:
        if "time_s" in df.columns:
            df["time_sec"] = df["time_s"]
        elif "absolute_frame" in df.columns:
            df["time_sec"] = df["absolute_frame"] / (fps or 60)
        elif "frame" in df.columns:
            df["time_sec"] = df["frame"] / (fps or 60)
    if "time_s" not in df.columns and "time_sec" in df.columns:
        df["time_s"] = df["time_sec"]
    if "time_sec" in df.columns:
        df = df.sort_values("time_sec")
    return df


@router.get(
    "/run_session/{run_session_id}/graphs",
    response_model=list[GraphOut],
)
async def get_run_session_graphs(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
):
    run_session = await _get_run_session(run_session_id, session, load_analysis=True)
    if run_session.runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run session not found or unauthorized")
    graphs = []
    metrics_csv, angles_csv = _analysis_csvs(run_session)

    if metrics_csv:
        df = _with_time_column(pd.read_csv(metrics_csv), run_session.fps)
        for y_col, title, ylabel in [
            ("dist_smooth_m" if "dist_smooth_m" in df.columns else "dist_m", "Distance", "Distance (m)"),
            ("speed_mps", "Velocity", "Velocity (m/s)"),
            ("accel_mps2", "Acceleration", "Acceleration (m/s²)"),
        ]:
            if "time_sec" in df.columns and y_col in df.columns:
                graphs.append(
                    build_graph(
                        df,
                        x_col="time_sec",
                        y_col=y_col,
                        title=title,
                        y_label=ylabel,
                    )
                )

    if angles_csv:
        df = _with_time_column(pd.read_csv(angles_csv), run_session.fps)
        if "time_sec" in df.columns:
            angle_groups = [
                ("Knee Angle", "Angle (deg)", ["left_knee_angle", "right_knee_angle"]),
                ("Hip Angle", "Angle (deg)", ["left_hip_angle", "right_hip_angle"]),
                ("Elbow Flexion", "Angle (deg)", ["left_elbow_flexion_angle", "right_elbow_flexion_angle"]),
                ("Shoulder Flexion", "Angle (deg)", ["left_shoulder_flexion", "right_shoulder_flexion"]),
                ("Pelvis Torso Angle", "Angle (deg)", ["pelvis_torso_angle"]),
            ]

            for title, y_label, columns in angle_groups:
                available = [c for c in columns if c in df.columns]
                if available:
                    graphs.append(build_multi_graph(df, "time_sec", available, title, y_label))

    if not graphs:
        raise HTTPException(status_code=404, detail="Analysis graph files not found")

    return graphs


@router.get(
    "/run_session/{run_session_id}/angles",
    response_model=AnglesOut,
)
async def get_run_session_angles(run_session_id: UUID, session: AsyncSession = Depends(get_session)):
    run_session = await _get_run_session(run_session_id, session)
    _, angles_csv = _analysis_csvs(run_session)
    if not angles_csv:
        raise HTTPException(status_code=404, detail="Angle file not found")

    df = _with_time_column(pd.read_csv(angles_csv), run_session.fps)
    if "frame" not in df.columns or "time_sec" not in df.columns:
        raise HTTPException(status_code=422, detail="Angle file lacks frame/time columns")

    columns = [
        col for col in df.columns
        if col not in {"frame", "time_sec", "time_s"}
    ]
    samples = []
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    for _, row in cleaned.iterrows():
        values = {}
        for col in columns:
            value = row[col]
            values[col] = None if pd.isna(value) else float(value)
        samples.append({
            "frame": int(row["frame"]),
            "timeSec": float(row["time_sec"]),
            "values": values,
        })

    return {"columns": columns, "samples": samples}


@router.get("/run_session/{run_session_id}/angles.csv")
async def download_run_session_angles_csv(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    run_session = await _get_run_session(run_session_id, session)
    _, angles_csv = _analysis_csvs(run_session)
    if not angles_csv:
        raise HTTPException(status_code=404, detail="Angle file not found")

    return FileResponse(
        angles_csv,
        media_type="text/csv; charset=utf-8",
        filename=f"angles_{run_session_id}.csv",
    )


def _format_value(value, unit: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}{unit}"
    return f"{value}{unit}"


def _summary_metric(label: str, value: str) -> str:
    return f"""
      <div class="metric">
        <div class="label">{escape(label)}</div>
        <div class="value">{escape(value)}</div>
      </div>
    """


def _svg_line_chart(
    series: list[tuple[str, list[float], list[float], str]],
    width: int = 700,
    height: int = 220,
    x_label: str = "Time (s)",
    y_label: str = "",
) -> str:
    pad = {"top": 28, "right": 24, "bottom": 44, "left": 56}
    all_x = [v for _, x, y, _ in series for v in x]
    all_y = [v for _, x, y, _ in series for v in y]
    if not all_x or not all_y:
        return ""
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_min -= 1
        y_max += 1
    pw = width - pad["left"] - pad["right"]
    ph = height - pad["top"] - pad["bottom"]
    sx = lambda v: pad["left"] + (v - x_min) / (x_max - x_min) * pw
    sy = lambda v: pad["top"] + (1 - (v - y_min) / (y_max - y_min)) * ph
    lines = ""
    for name, x, y, color in series:
        if not x:
            continue
        pts = " ".join(f"{sx(xi):.1f},{sy(yi):.1f}" for xi, yi in zip(x, y))
        lines += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>\n'
    grid = ""
    for i in range(5):
        v = y_min + (y_max - y_min) * i / 4
        vy = sy(v)
        grid += f'<line x1="{pad["left"]}" y1="{vy:.1f}" x2="{pad["left"] + pw}" y2="{vy:.1f}" stroke="#eee" stroke-width="1"/>\n'
        grid += f'<text x="{pad["left"] - 6}" y="{vy:.1f}" text-anchor="end" dominant-baseline="middle" font-size="10" fill="#888">{v:.1f}</text>\n'
    xticks = ""
    n = min(8, max(2, int(x_max - x_min)))
    for i in range(n + 1):
        v = x_min + (x_max - x_min) * i / n
        vx = sx(v)
        xticks += f'<text x="{vx:.1f}" y="{pad["top"] + ph + 14}" text-anchor="middle" font-size="10" fill="#888">{v:.1f}</text>\n'
    legend = ""
    for i, (name, _, _, color) in enumerate(series):
        lx = pad["left"] + i * 130
        legend += f'<rect x="{lx}" y="{pad["top"] - 18}" width="18" height="3" rx="1" fill="{color}"/>\n'
        legend += f'<text x="{lx + 22}" y="{pad["top"] - 12}" font-size="11" fill="#444">{escape(name)}</text>\n'
    mid_x = pad["left"] + pw / 2
    mid_y = pad["top"] + ph / 2
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto">\n'
        f'{grid}{lines}{xticks}{legend}'
        f'<line x1="{pad["left"]}" y1="{pad["top"]}" x2="{pad["left"]}" y2="{pad["top"] + ph}" stroke="#ccc" stroke-width="1.5"/>\n'
        f'<line x1="{pad["left"]}" y1="{pad["top"] + ph}" x2="{pad["left"] + pw}" y2="{pad["top"] + ph}" stroke="#ccc" stroke-width="1.5"/>\n'
        f'<text x="{mid_x:.1f}" y="{height - 4}" text-anchor="middle" font-size="11" fill="#888">{escape(x_label)}</text>\n'
        f'<text x="12" y="{mid_y:.1f}" text-anchor="middle" font-size="11" fill="#888" '
        f'transform="rotate(-90,12,{mid_y:.1f})">{escape(y_label)}</text>\n'
        f'</svg>'
    )


def _chart_section(title: str, svg: str, stats: list[tuple[str, str]]) -> str:
    if not svg:
        return ""
    stats_items = "".join(
        f'<div class="cstat"><div class="cstat-label">{escape(label)}</div>'
        f'<div class="cstat-value">{escape(value)}</div></div>'
        for label, value in stats
    )
    stats_block = f'<div class="cstat-row">{stats_items}</div>' if stats_items else ""
    return f"""
    <section class="chart-section">
      <h2>{escape(title)}</h2>
      <div class="chart-wrap">{svg}</div>
      {stats_block}
    </section>
    """


def _build_report_html(
    run_session: RunSession,
    metrics_csv: Path | None,
    angles_csv: Path | None,
) -> str:
    analysis = run_session.analysis
    runner_name = run_session.runner.name if run_session.runner else "-"

    charts_html = ""

    if metrics_csv:
        try:
            df = _with_time_column(pd.read_csv(metrics_csv), run_session.fps)
            df = df.replace([np.inf, -np.inf], np.nan)
            if "time_sec" in df.columns:
                x = _sample(df["time_sec"].tolist(), 300)

                dist_col = next((c for c in ("dist_smooth_m", "dist_m") if c in df.columns), None)
                if dist_col:
                    y = _sample(df[dist_col].fillna(0).tolist(), 300)
                    col_data = df[dist_col].dropna()
                    charts_html += _chart_section(
                        "Distance",
                        _svg_line_chart([("Distance", x, y, "#6366f1")], y_label="m"),
                        stats=[("Total", f"{col_data.max():.2f} m")] if not col_data.empty else [],
                    )

                if "speed_mps" in df.columns:
                    col_data = df["speed_mps"].dropna()
                    y = _sample(df["speed_mps"].fillna(0).tolist(), 300)
                    charts_html += _chart_section(
                        "Velocity",
                        _svg_line_chart([("Velocity", x, y, "#0ea5e9")], y_label="m/s"),
                        stats=[
                            ("Average", f"{col_data.mean():.3f} m/s"),
                            ("Max", f"{col_data.max():.3f} m/s"),
                            ("Min", f"{col_data.min():.3f} m/s"),
                        ] if not col_data.empty else [],
                    )

                if "accel_mps2" in df.columns:
                    col_data = df["accel_mps2"].dropna()
                    y = _sample(df["accel_mps2"].fillna(0).tolist(), 300)
                    charts_html += _chart_section(
                        "Acceleration",
                        _svg_line_chart([("Acceleration", x, y, "#f97316")], y_label="m/s²"),
                        stats=[
                            ("Average", f"{col_data.mean():.3f} m/s²"),
                            ("Max", f"{col_data.max():.3f} m/s²"),
                            ("Min", f"{col_data.min():.3f} m/s²"),
                        ] if not col_data.empty else [],
                    )
        except Exception as e:
            print(f"[report] metrics chart error: {e}")

    if angles_csv:
        try:
            df = _with_time_column(pd.read_csv(angles_csv), run_session.fps)
            df = df.replace([np.inf, -np.inf], np.nan)
            if "time_sec" in df.columns:
                x = _sample(df["time_sec"].tolist(), 300)
                angle_groups = [
                    ("Knee Angle", [("left_knee_angle", "Left", "#0ea5e9"), ("right_knee_angle", "Right", "#f97316")]),
                    ("Hip Angle", [("left_hip_angle", "Left", "#0ea5e9"), ("right_hip_angle", "Right", "#f97316")]),
                    ("Elbow Flexion", [("left_elbow_flexion_angle", "Left", "#0ea5e9"), ("right_elbow_flexion_angle", "Right", "#f97316")]),
                    ("Shoulder Flexion", [("left_shoulder_flexion", "Left", "#0ea5e9"), ("right_shoulder_flexion", "Right", "#f97316")]),
                    ("Pelvis Torso Angle", [("pelvis_torso_angle", "Pelvis-Torso", "#22c55e")]),
                ]
                for title, col_defs in angle_groups:
                    available = [(col, label, color) for col, label, color in col_defs if col in df.columns]
                    if not available:
                        continue
                    series = []
                    stats = []
                    for col, label, color in available:
                        y = _sample(df[col].fillna(0).tolist(), 300)
                        series.append((label, x, y, color))
                        col_data = df[col].dropna()
                        if not col_data.empty:
                            stats += [
                                (f"{label} Avg", f"{col_data.mean():.1f}°"),
                                (f"{label} Max", f"{col_data.max():.1f}°"),
                                (f"{label} Min", f"{col_data.min():.1f}°"),
                            ]
                    charts_html += _chart_section(
                        title,
                        _svg_line_chart(series, y_label="deg"),
                        stats=stats,
                    )
        except Exception as e:
            print(f"[report] angles chart error: {e}")

    summary_metrics = [
        ("Runner", runner_name),
        ("Date", run_session.date.isoformat(sep=" ", timespec="seconds")),
        ("Status", run_session.status),
        ("FPS", str(run_session.fps)),
        ("Camera Count", str(run_session.camera_count)),
        ("Total Time", _format_value(analysis.total_time if analysis else None, " s")),
        ("Avg Velocity", _format_value(analysis.avg_velocity if analysis else None, " m/s")),
        ("Avg Acceleration", _format_value(analysis.avg_acceleration if analysis else None, " m/s²")),
        ("Avg Step Length", _format_value(analysis.avg_step_length if analysis else None, " m")),
        ("Note", run_session.note or "-"),
    ]
    summary_html = "".join(_summary_metric(label, value) for label, value in summary_metrics)

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Runner Analysis Report</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #f4f6f8;
      color: #17212b;
      font-family: Arial, "Noto Sans TC", sans-serif;
      line-height: 1.5;
    }}
    .page {{
      max-width: 1040px;
      margin: 24px auto;
      background: #fff;
      padding: 40px;
      box-shadow: 0 8px 30px rgba(24, 34, 45, 0.12);
    }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    h2 {{
      margin: 0 0 14px;
      padding-bottom: 8px;
      border-bottom: 2px solid #d9e1ea;
      font-size: 18px;
    }}
    .subtle {{ color: #647487; font-size: 13px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 12px;
      margin: 24px 0 32px;
    }}
    .metric {{
      border: 1px solid #dce4ed;
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }}
    .label {{ color: #65758a; font-size: 12px; margin-bottom: 5px; }}
    .value {{ font-size: 19px; font-weight: 700; word-break: break-word; }}
    .chart-section {{ margin: 32px 0; }}
    .chart-wrap {{
      overflow-x: auto;
      background: #fafbfc;
      border: 1px solid #e8edf2;
      border-radius: 8px;
      padding: 12px;
    }}
    .cstat-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .cstat {{
      background: #f3f6fa;
      border: 1px solid #dce4ed;
      border-radius: 6px;
      padding: 8px 14px;
      min-width: 110px;
    }}
    .cstat-label {{ color: #65758a; font-size: 11px; }}
    .cstat-value {{ font-size: 16px; font-weight: 700; }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ margin: 0; box-shadow: none; max-width: none; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>Runner Analysis Report</h1>
    <div class="subtle">Session: {escape(str(run_session.id))}</div>
    <div class="summary">{summary_html}</div>
    {charts_html if charts_html else "<p>No analysis data available.</p>"}
  </main>
</body>
</html>"""


@router.get("/run_session/{run_session_id}/report")
async def download_run_session_report(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    run_session = await _get_run_session(run_session_id, session, load_analysis=True)
    metrics_csv, angles_csv = _analysis_csvs(run_session)
    if not metrics_csv and not angles_csv:
        raise HTTPException(status_code=404, detail="Analysis files not found")

    html = _build_report_html(run_session, metrics_csv, angles_csv)
    return HTMLResponse(
        content=html,
        headers={
            "Content-Disposition": f'attachment; filename="running_report_{run_session_id}.html"'
        },
    )


@router.get("/run_session/{run_session_id}/video")
async def get_run_session_video(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user)
) -> FileResponse:
    run_session = (await session.execute(
        select(RunSession)
        .options(selectinload(RunSession.runner), selectinload(RunSession.analysis))
        .where(RunSession.id == run_session_id)
    )).scalars().first()
    if not run_session or run_session.runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run session not found or unauthorized")

    video_path = None
    if run_session.analysis and isinstance(run_session.analysis.summary, dict):
        summary_video = run_session.analysis.summary.get("uncropped_video")
        if summary_video and Path(summary_video).exists():
            video_path = Path(summary_video)

    session_dir = RUN_SESSION_DIR / str(run_session.runner_id) / str(run_session_id)
    if video_path is None:
        video_path = _first_existing(
            [
                session_dir / "sequential_tracked.mp4",
                session_dir / "output_final.mp4",
                session_dir / "cam1_uncropped_2D.mp4",
                session_dir / "tracking_results" / "output_final.mp4",
                session_dir / "tracking_results" / "analyzed_video_meta.mp4",
            ]
        )

    if video_path is None:
        uncropped_matches = sorted(session_dir.glob("*_uncropped_2D.mp4"))
        video_path = uncropped_matches[0] if uncropped_matches else None

    if video_path is None:
        # The on-device pipeline currently emits metrics/pose artifacts but no
        # rendered overlay, so Local sessions play their first source camera.
        camera_matches = sorted(path for path in session_dir.glob("cam1.*") if path.is_file())
        video_path = camera_matches[0] if camera_matches else None

    if video_path is None:
        raise HTTPException(status_code=404, detail="Analysis video not found")

    return FileResponse(video_path)


@router.get("/temp_video/{temp_video_id}/thumbnail")
def get_thumbnail(
    temp_video_id: str,
    current_user: User = Depends(get_current_user),
):
    return FileResponse(TEMP_UPLOAD_DIR / f"{temp_video_id}.jpg")

@router.get("/run_session/{run_session_id}/csv")
async def get_run_session_csv(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    run_session = await _get_run_session(run_session_id, session, load_analysis=True)
    if run_session.runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run session not found or unauthorized")
    _, angles_csv = _analysis_csvs(run_session)
    if not angles_csv:
        raise HTTPException(status_code=404, detail="CSV file not found")

    return FileResponse(
        angles_csv,
        media_type="text/csv",
        filename=f"angles_{run_session_id}.csv",
    )


@router.get("/run_session/{run_session_id}/pdf")
async def get_run_session_pdf(
    run_session_id: UUID,
    session: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    run_session = await _get_run_session(run_session_id, session, load_analysis=True)
    if run_session.runner.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Run session not found or unauthorized")
    results_dir = _session_dir(run_session)
    pdf_path = results_dir / "report.pdf"

    # Generate if not cached
    if not pdf_path.exists():
        if run_session.status != "done" or not run_session.analysis:
            raise HTTPException(status_code=400, detail="Run session is not fully analyzed yet")

        # Prepare session info dict
        run_session_info = {
            "id": str(run_session_id),
            "runnerName": run_session.runner.name,
            "date": run_session.date,
            "fps": run_session.fps,
            "cameraCount": run_session.camera_count,
            "totalTime": run_session.analysis.total_time,
            "avgVelocity": run_session.analysis.avg_velocity,
            "avgAcceleration": run_session.analysis.avg_acceleration,
            "avgStepLength": run_session.analysis.avg_step_length,
            "note": run_session.note,
            "status": run_session.status,
        }

        try:
            generate_pdf_report(run_session_info, str(results_dir), str(pdf_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"Runner_Analysis_Report_{run_session_id}.pdf",
    )
