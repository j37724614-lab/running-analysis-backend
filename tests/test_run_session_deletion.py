import asyncio
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from db_models import (
    AnalysisMeta,
    AnalysisRun,
    ComparisonReport,
    Runner,
    RunSession,
    User,
    Video,
)
from routes.run import delete_run_session


async def _database():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


async def _owned_session(session, *, compare: bool):
    user = User(username=f"user-{uuid4()}", hashed_password="hash")
    session.add(user)
    await session.flush()
    runner = Runner(name="Runner", user_id=user.id)
    session.add(runner)
    await session.flush()
    run_session = RunSession(
        runner_id=runner.id,
        camera_count=1,
        status="done",
        result_dir=None,
    )
    session.add(run_session)
    await session.flush()
    session.add(
        AnalysisMeta(
            run_session_id=run_session.id,
            total_time=1.0,
        )
    )
    session.add(
        Video(
            run_session_id=run_session.id,
            camera_index=0,
            video_path="/path/that/does/not/exist.mov",
        )
    )

    group_id = uuid4() if compare else None
    server_run = AnalysisRun(
        run_session_id=run_session.id,
        compute_location="server",
        comparison_group_id=group_id,
        status="completed",
    )
    session.add(server_run)
    if compare:
        local_run = AnalysisRun(
            run_session_id=run_session.id,
            compute_location="local",
            comparison_group_id=group_id,
            status="completed",
        )
        session.add(local_run)
        await session.flush()
        session.add(
            ComparisonReport(
                comparison_group_id=group_id,
                server_run_id=server_run.id,
                local_run_id=local_run.id,
                status="complete",
                input_hashes_match=True,
                payload={"schema_version": "1.0.0"},
            )
        )
    await session.commit()
    return user, run_session.id


def test_delete_local_run_session_removes_analysis_run_dependencies():
    async def scenario():
        engine = await _database()
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            user, run_session_id = await _owned_session(session, compare=False)

            result = await delete_run_session(
                run_session_id,
                session=session,
                current_user=user,
            )

            assert result["status"] == "success"
            assert await _count(session, RunSession) == 0
            assert await _count(session, AnalysisRun) == 0
            assert await _count(session, AnalysisMeta) == 0
            assert await _count(session, Video) == 0
        await engine.dispose()

    asyncio.run(scenario())


def test_delete_compare_run_session_removes_report_before_analysis_runs():
    async def scenario():
        engine = await _database()
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            user, run_session_id = await _owned_session(session, compare=True)

            result = await delete_run_session(
                run_session_id,
                session=session,
                current_user=user,
            )

            assert result["status"] == "success"
            assert await _count(session, ComparisonReport) == 0
            assert await _count(session, AnalysisRun) == 0
            assert await _count(session, RunSession) == 0
        await engine.dispose()

    asyncio.run(scenario())


async def _count(session, model) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())
