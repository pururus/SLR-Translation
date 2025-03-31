from aiogram import F, Router, types
from aiogram.filters import Command, StateFilter, or_f
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

from sqlalchemy.ext.asyncio import AsyncSession

from bl_bl.telegram_bot.filters.chat_types import ChatTypeFilter, IsAdmin

admin_router = Router()
admin_router.message.filter(ChatTypeFilter(["private"]), IsAdmin())


@admin_router.message(Command("admin"))
async def admin_features(message: types.Message):
    await message.answer("Что хотите сделать?")

