from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from bl_bl.telegram_bot.common.utils import is_subscribed

class Validator(BaseMiddleware):
    def __init__(self, channel_id: str):
        self.channel_id = channel_id

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: Message,
        data: Dict[str, Any],
    ) -> Any:
        clinet = data.get('client')
        bot = data.get('bot')
        if event.text == "/start":
            return await handler(event, data)
        elif clinet and bot and await is_subscribed(event.from_user.id, bot, self.channel_id):
            await clinet.add_new_user_if_not_exists(event.from_user.id)
            return await handler(event, data)
        else:
            await event.answer("Вы не подписаны на канал @slr_translation. Пожалуйста, подпишитесь для доступа к контенту.")
