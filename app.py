import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from bl_bl.telegram_bot.middlewares.db import DataBaseSession
from bl_bl.telegram_bot.middlewares.validate import Validator

from bl_bl.telegram_bot.database.engine import client

from bl_bl.telegram_bot.handlers.user_private import user_private_router
from bl_bl.telegram_bot.handlers.user_group import user_group_router
from bl_bl.telegram_bot.handlers.admin_private import admin_router

from bl_bl.telegram_bot.common.bot_cmds_list import private


ALLOWED_UPDATES = ['message']

bot = Bot(token=os.getenv('TOKEN'))
bot.my_admins_list = []

dp = Dispatcher()

dp.include_router(user_private_router)
dp.include_router(user_group_router)
dp.include_router(admin_router)

async def on_startup(bot):
    # await client.drop_db()
    await client.create_db()

async def on_shutdown(bot):
    pass


async def main():
    logging.basicConfig(level=logging.INFO)

    dp.startup.register(on_startup)
    dp.shutdown.register(on_shutdown)

    dp.update.middleware(DataBaseSession(client=client))
    user_private_router.message.middleware(Validator("@slr_translation"))

    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_my_commands(commands=private, scope=types.BotCommandScopeAllPrivateChats())
    await dp.start_polling(bot, allowed_updates=ALLOWED_UPDATES)

asyncio.run(main())
