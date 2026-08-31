from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import select, update, text
from sqlmodel import SQLModel
from db.engine import engine
from db_models import *
import bcrypt


async def ensure_analysis_meta_nullable(connection):
    """Migrate legacy SQLite analysis metrics to the model's nullable contract."""
    columns = (await connection.execute(text("PRAGMA table_info(analysis_meta)"))).all()
    if not columns:
        return
    optional_metrics = {
        "total_time",
        "avg_velocity",
        "avg_acceleration",
        "avg_step_length",
    }
    if not any(row[1] in optional_metrics and row[3] for row in columns):
        return

    await connection.execute(text("DROP TABLE IF EXISTS analysis_meta_nullable"))
    await connection.execute(
        text(
            """
            CREATE TABLE analysis_meta_nullable (
                run_session_id CHAR(32) NOT NULL,
                total_time FLOAT,
                avg_velocity FLOAT,
                avg_acceleration FLOAT,
                avg_step_length FLOAT,
                summary JSON,
                PRIMARY KEY (run_session_id),
                FOREIGN KEY(run_session_id) REFERENCES run_session (id)
            )
            """
        )
    )
    await connection.execute(
        text(
            """
            INSERT INTO analysis_meta_nullable
                (run_session_id, total_time, avg_velocity, avg_acceleration, avg_step_length, summary)
            SELECT run_session_id, total_time, avg_velocity, avg_acceleration, avg_step_length, summary
            FROM analysis_meta
            """
        )
    )
    await connection.execute(text("DROP TABLE analysis_meta"))
    await connection.execute(
        text("ALTER TABLE analysis_meta_nullable RENAME TO analysis_meta")
    )


def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

async def init_db():
    # 建立所有表格
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await ensure_analysis_meta_nullable(conn)
        # 由於 SQLModel metadata 不會自動對既存的表格增加新欄位，因此我們手動進行 ALTER TABLE
        try:
            await conn.execute(text("ALTER TABLE runner ADD COLUMN user_id CHAR(32) REFERENCES user(id)"))
        except Exception as e:
            # 如果欄位已存在，此處會拋出 OperationalError (duplicate column name)，可以安全忽略
            pass
        
    # 確保預設使用者 test 存在，並將無主資料關聯過去
    from db.session import async_session
    async with async_session() as session:
        # 1. 查找是否存在 test 使用者
        result = await session.execute(select(User).where(User.username == "test"))
        test_user = result.scalars().first()
        
        if not test_user:
            hashed = hash_password("test")
            test_user = User(username="test", hashed_password=hashed)
            session.add(test_user)
            await session.commit()
            await session.refresh(test_user)
            
        # 2. 將所有 user_id 為空（舊資料）的 runner 關聯給 test 使用者
        await session.execute(
            update(Runner)
            .where(Runner.user_id == None)
            .values(user_id=test_user.id)
        )
        await session.commit()
