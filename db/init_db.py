from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy import select, update, text
from sqlmodel import SQLModel
from db.engine import engine
from db_models import *
import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

async def init_db():
    # 建立所有表格
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
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
