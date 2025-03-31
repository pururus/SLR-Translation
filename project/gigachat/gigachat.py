import requests
import base64
import uuid
import json
import os

from pathlib import Path
import asyncio
import aiohttp

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
cal_credentials = os.getenv('cal_credentials')
import logging
from typing import Dict, Optional

logging.captureWarnings(True)

class GigaChat:
    _token = None
    _model = "GigaChat-2-Max"
    _url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    async def get_token(self, auth_token, scope='GIGACHAT_API_PERS'):
        '''
        Функция возвращает API токен для gigachat
        
        :param auth_token: Закодированные client_id и секретный ключ
        '''
        rq_uid = str(uuid.uuid4())
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': rq_uid,
            'Authorization': f'Basic {auth_token}'
        }
        payload = {
            'scope': scope
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=payload, ssl=False) as response:
                    return await response.json()
                
        except aiohttp.ClientError as e:
            print(f"Ошибка: {str(e)}")
            return None

    async def check_token(self):
        '''
        Проверяет, получен ли API токен, и, если нет, устанавливает токен в соответствующее поле.
        '''
        if self._token is None:
            encoded_credentials = base64.b64encode(cal_credentials.encode('utf-8')).decode('utf-8')
            token = await self.get_token(encoded_credentials)
            if token:
                self._token = (token)["access_token"]
                return 0
            else:
                print("Не удалось получить токен")
                return 1
            
    async def request(self, message: str, max_tockens: int = 50, temp=1):
        '''
        Функция для запросов к API
        
        :param message: Сообщение, отправляемое LLM
        :param max_tockens: Максимальное количество токенов в ответе
        :param temp: Температура. Влияет на ответ
        
        :return: словарь
        '''
        
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": message
                }
            ],
            "stream": False,
            "max_tokens": max_tockens,
            "temerature": temp
        })

        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self._token}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self._url, headers=headers, data=payload, ssl=False) as response:
                    return await response.json()
                
        except aiohttp.ClientError as e:
            print(f"Ошибка: {str(e)}")
    
    async def parse_translation(self, translation: str):
        message = f'''
        Ты профессиональный переводчик с русского жестового языка на русский естественный язык. Напиши только перевод.
        Тебе нужно привести к естественному виду глоссовую расшифровку текста на РЖЯ: {translation}.
        Напиши только перевод. Можешь пропускать слова, не подходящие по контексту.
        ВЕРНИ ТОЛЬКО ПЕРЕВОД, НЕ ПИШИ РАССУЖДЕНИЯ И ПРЕДПОЛОЖЕНИЯ
        '''
        
        response = await self.request(message)
        if response and 'choices' in response:
            return response['choices'][0]['message']['content']
