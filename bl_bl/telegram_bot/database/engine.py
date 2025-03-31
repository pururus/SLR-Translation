import os
from datetime import datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bl_bl.telegram_bot.database.models import Base, User


class WorkWithDB:
    def __init__(self, path):
        self.engine = create_async_engine(path, echo=True)
        self.session = async_sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)

    async def create_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def is_user_exists(self, tg_id) -> bool:
        async with self.session() as session:
            user = await session.get(User, tg_id)
            return bool(user)
        
    async def add_new_user(self, tg_id) -> None:
        async with self.session() as session:
            session.add(
                User(tg_id=tg_id)
            )
            await session.commit()

    async def add_new_user_if_not_exists(self, tg_id) -> bool:
        result = await self.is_user_exists(tg_id)
        if not result:
            await self.add_new_user(tg_id)
        return result

    async def orm_video_check(self, tg_id: int, duration: int) -> bool:
        '''
        Проверка оставшегося время и запросов пользователя
        '''
        await self.update_params(tg_id)
        async with self.session() as session:
            user = await session.get(User, tg_id)
            if user:
                return duration + user.duration_used <= user.duration_max and \
                       1 + user.requests_used <= user.requests_max        
            return False

    async def orm_video_update(self, user_id: int, duration: int):
        '''
        Обновление времени и количества запросов пользователя в таблицу 
        '''
        async with self.session() as session:
            query = update(User).where(User.tg_id == user_id).values(requests_used=User.requests_used + 1, duration_used = User.duration_used + duration)
            await session.execute(query)
            await session.commit()

    async def update_params(self, tg_id):
        async with self.session() as session:
            user = await session.get(User, tg_id)
            if user:
                if user.last_request.date() != datetime.now().date():
                    user.requests_used = 0
                    user.duration_used = 0
                    user.last_request = datetime.now()
                    await session.commit()
    
    async def get_user(self, tg_id) -> User | None:
        async with self.session() as session:
            user = await session.get(User, tg_id)
            return user
            
client = WorkWithDB(os.getenv('DB_LITE')) 