import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


def test_analysis_metrics_become_nullable_without_losing_rows():
    from db.init_db import ensure_analysis_meta_nullable

    async def scenario():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.execute(
                text("CREATE TABLE run_session (id CHAR(32) PRIMARY KEY)")
            )
            await connection.execute(
                text(
                    """
                    CREATE TABLE analysis_meta (
                        run_session_id CHAR(32) NOT NULL PRIMARY KEY,
                        total_time FLOAT NOT NULL,
                        avg_velocity FLOAT NOT NULL,
                        avg_acceleration FLOAT NOT NULL,
                        avg_step_length FLOAT NOT NULL,
                        summary JSON,
                        FOREIGN KEY(run_session_id) REFERENCES run_session(id)
                    )
                    """
                )
            )
            await connection.execute(
                text("INSERT INTO run_session (id) VALUES ('trial')")
            )
            await connection.execute(
                text(
                    """
                    INSERT INTO analysis_meta
                    (run_session_id, total_time, avg_velocity, avg_acceleration, avg_step_length)
                    VALUES ('trial', 3.2, 8.1, 0.4, 0.0)
                    """
                )
            )

            await ensure_analysis_meta_nullable(connection)

            columns = (await connection.execute(text("PRAGMA table_info(analysis_meta)"))).all()
            nullable_metrics = {
                row[1]: row[3]
                for row in columns
                if row[1] in {
                    "total_time",
                    "avg_velocity",
                    "avg_acceleration",
                    "avg_step_length",
                }
            }
            preserved = (
                await connection.execute(
                    text("SELECT run_session_id, total_time FROM analysis_meta")
                )
            ).one()

        await engine.dispose()
        assert nullable_metrics == {
            "total_time": 0,
            "avg_velocity": 0,
            "avg_acceleration": 0,
            "avg_step_length": 0,
        }
        assert tuple(preserved) == ("trial", 3.2)

    asyncio.run(scenario())
