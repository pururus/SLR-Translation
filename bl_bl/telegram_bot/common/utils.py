from aiogram import Bot

async def is_subscribed(user_id: int, bot: Bot, CHANNEL_ID) -> bool:
    try:
        member = await bot.get_chat_member(chat_id=CHANNEL_ID, user_id=user_id)
        if member.status in ["creator", "administrator", "member"]:
            return True
        else:
            return False
    except Exception as e:
        return False