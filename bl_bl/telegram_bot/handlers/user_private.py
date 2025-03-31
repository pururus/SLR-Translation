import logging
import os

from aiogram import Router, F, Bot
from aiogram.types import Message
from aiogram.filters import Command

from bl_bl.telegram_bot.common.neuro import slr_translator
from bl_bl.telegram_bot.database.engine import WorkWithDB
from bl_bl.telegram_bot.filters.chat_types import ChatTypeFilter

from process_video import process_video

user_private_router = Router()
user_private_router.message.filter(ChatTypeFilter(["private"]))

@user_private_router.message(Command("start"))
async def start_handler(message: Message):
    await message.answer("Привет! Я телеграм бот для перевода русского языка жестов в текст")

@user_private_router.message(Command("help"))
async def help_handler(message: Message):
    await message.answer("""
                         Для перевода вам нужно отправить видео удволетворяющее следующим требованиям:
                            1. Видео должено быть горизонтальным.
                            2. На видео должен находиться один человек на расстоянии 1-1.5 метра от экрана, расположеный по середине и смотрящий в экран.
                            3. Все жесты должны быть отчетливо видны.
                         """)

@user_private_router.message(Command("info"))
async def info_handler(message: Message):
    await message.answer("Бот основан на 100000 индусов, которые были обучены русскому сурдопереводу.\n По всем вопросам писать на почту alex@karachun.com и мы накажем индусов плохо выполняющих свою работу!!!")

@user_private_router.message(Command("limits"))
async def info_handler(message: Message, client: WorkWithDB):
    await client.update_params(message.from_user.id)
    user = await client.get_user(message.from_user.id)
    if user:
        await message.answer(f"Использовано {user.duration_used} сек. из {user.duration_max} сек.\nИспользовано {user.requests_used} запросов из {user.requests_max}.\n")

@user_private_router.message(F.video | F.video_note | F.animation)
async def video_handler(message: Message, bot: Bot, client: WorkWithDB):
    file_id = None
    duration = 0
    if message.video:
        file_id = message.video.file_id
        duration = message.video.duration
    elif message.video_note:
        file_id = message.video_note.file_id
        duration = message.video_note.duration
    elif message.animation:
        file_id = message.animation.file_id
        duration = message.animation.duration

    if not await client.orm_video_check(message.from_user.id, duration):
        await message.answer("Ваш лимит запросов исчерпан 😢")
    else:
        await client.orm_video_update(message.from_user.id, duration)

        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path

        logging.info(f"Download file with path: {file_path}")
        
        await bot.download_file(file_path=file_path, destination=f"{file_id}.mp4")
        await message.reply(await process_video(f"{file_id}.mp4", 'Alex_Karachun/trained_models/s3d_1000_gestures_1000_videos_7_epochs_done/s3d_1000_gestures_1000_videos_5_epoch'))
        if os.path.exists(f"{file_id}.mp4"):
            os.remove(f"{file_id}.mp4")

@user_private_router.message()
async def other_handler(message: Message):
    await message.answer("Я умею работать только с видео. Пришлите видео и тогда я смогу его перевести 🤗")
        




