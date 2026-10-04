import asyncio
from datetime import datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from db_models import (
    AnalysisMeta,
    AnalysisRun,
    Runner,
    RunSession,
    synthesize_legacy_server_run,
)


async def _make_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(SQLModel.metadata.create_all)
    return engine


def test_analysis_run_table_is_created_and_round_trips():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            runner = Runner(name="Runner A")
            session.add(runner)
            await session.flush()

            run_session = RunSession(runner_id=runner.id)
            session.add(run_session)
            await session.flush()

            analysis_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="server",
                engine_version="97e7fac",
            )
            session.add(analysis_run)
            await session.commit()

            loaded = (
                await session.execute(
                    select(AnalysisRun).where(
                        AnalysisRun.run_session_id == run_session.id
                    )
                )
            ).scalar_one()
            assert loaded.compute_location == "server"
            assert loaded.status == "pending"
            assert loaded.comparison_group_id is None
        await engine.dispose()

    asyncio.run(scenario())


def test_run_session_can_have_two_analysis_runs_for_compare_without_overwriting():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            runner = Runner(name="Runner B")
            session.add(runner)
            await session.flush()

            run_session = RunSession(runner_id=runner.id)
            session.add(run_session)
            await session.flush()

            comparison_group_id = uuid4()
            server_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="server",
                comparison_group_id=comparison_group_id,
                status="completed",
            )
            local_run = AnalysisRun(
                run_session_id=run_session.id,
                compute_location="local",
                comparison_group_id=comparison_group_id,
                status="completed",
            )
            session.add_all([server_run, local_run])
            await session.commit()

            runs = (
                await session.execute(
                    select(AnalysisRun)
                    .where(AnalysisRun.run_session_id == run_session.id)
                    .order_by(AnalysisRun.compute_location)
                )
            ).scalars().all()

            assert [r.compute_location for r in runs] == ["local", "server"]
            assert {r.id for r in runs} == {local_run.id, server_run.id}
            assert all(r.comparison_group_id == comparison_group_id for r in runs)
        await engine.dispose()

    asyncio.run(scenario())


def test_synthesize_legacy_server_run_for_session_without_real_rows():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            runner = Runner(name="Runner C")
            session.add(runner)
            await session.flush()

            run_session = RunSession(
                runner_id=runner.id,
                status="done",
                created_at=datetime(2026, 5, 1, 12, 0),
            )
            session.add(run_session)
            await session.flush()

            meta = AnalysisMeta(run_session_id=run_session.id, avg_velocity=8.5)
            session.add(meta)
            await session.commit()

            # Re-load with relationships populated, the way a real caller would.
            reloaded = (
                await session.execute(
                    select(RunSession).where(RunSession.id == run_session.id)
                )
            ).scalar_one()
            await session.refresh(reloaded, attribute_names=["analysis_runs", "analysis"])

            synthesized = synthesize_legacy_server_run(reloaded)
            assert synthesized is not None
            assert synthesized.compute_location == "server"
            assert synthesized.status == "completed"
            assert synthesized.run_session_id == run_session.id

            # Must be transient - nothing was written to the table.
            persisted_count = (
                await session.execute(select(AnalysisRun))
            ).scalars().all()
            assert persisted_count == []
        await engine.dispose()

    asyncio.run(scenario())


def test_synthesize_legacy_server_run_returns_none_when_real_row_exists():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            runner = Runner(name="Runner D")
            session.add(runner)
            await session.flush()

            run_session = RunSession(runner_id=runner.id, status="done")
            session.add(run_session)
            await session.flush()

            session.add(AnalysisMeta(run_session_id=run_session.id, avg_velocity=7.0))
            session.add(
                AnalysisRun(run_session_id=run_session.id, compute_location="server")
            )
            await session.commit()

            reloaded = (
                await session.execute(
                    select(RunSession).where(RunSession.id == run_session.id)
                )
            ).scalar_one()
            await session.refresh(reloaded, attribute_names=["analysis_runs", "analysis"])

            assert synthesize_legacy_server_run(reloaded) is None
        await engine.dispose()

    asyncio.run(scenario())


def test_synthesize_legacy_server_run_returns_none_for_unanalyzed_session():
    async def scenario():
        engine = await _make_engine()
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        async with session_factory() as session:
            runner = Runner(name="Runner E")
            session.add(runner)
            await session.flush()

            run_session = RunSession(runner_id=runner.id, status="pending")
            session.add(run_session)
            await session.commit()

            reloaded = (
                await session.execute(
                    select(RunSession).where(RunSession.id == run_session.id)
                )
            ).scalar_one()
            await session.refresh(reloaded, attribute_names=["analysis_runs", "analysis"])

            assert synthesize_legacy_server_run(reloaded) is None
        await engine.dispose()

    asyncio.run(scenario())
