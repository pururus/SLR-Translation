from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from sqlalchemy.ext.asyncio import async_sessionmaker
import bl_bl.telegram_bot.database.engine 

class DataBaseSession(BaseMiddleware):
    def __init__(self, client: bl_bl.telegram_bot.database.engine.WorkWithDB):
        self.client = client

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        data['client'] = self.client
        return await handler(event, data)