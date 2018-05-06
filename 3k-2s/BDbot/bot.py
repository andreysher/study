# -*- coding: utf-8 -*-
import config
import telebot
import dataBase

"""
0 - Начальное состояние
1) Добавление дела
1 - Придумывает название дела
2 - Устанавливает DeadLine
3 - Устанавливает приоритет
4 - Подтверждает добавление дела
"""

bot = telebot.TeleBot(config.token)
activeUsers = {}

dbController = dataBase.DBDriver()


@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
    Привет. Я твой помошник в планировании дел.
    Умею добавлять дело в базу(команда "Добавить дело"),
    показывать состояние твоих запланированых дел(команда "Дела"),
    отмечать выполнение дела(команда "Выполнено <Название дела>"),
    показывать твою продуктивность(команда "Продуктивность"),
    ...
    Скоро научусь еще кое-чему;)
    \
""")
    activeUsers[message.chat.id] = {"state": 0}


@bot.message_handler(commands=['addTask'])
@bot.message_handler(func=lambda message: message.text == "Добавить дело")
def add_task(message):
    chat_id = message.chat.id
    bot.send_message(chat_id, "Как назвать дело?")
    activeUsers[chat_id]["state"] = 1
    activeUsers[chat_id]["task"] = dataBase.Task()


@bot.message_handler(func=lambda message: activeUsers[message.chat.id]["state"] == 1)
def parseTaskName(message):
    """добавить проверку корректности имени(вообще запилить отдельную функцию или класс)"""
    name = message.text
    chat_id = message.chat_id
    activeUsers[chat_id]["task"].setName(name)
    activeUsers[chat_id]["state"] = 2


@bot.message_handler(func=lambda message: activeUsers[message.chat_id]["state"] == 2)
def parseDeadLine(message):
    deadline = int(message.text)
    chat_id = message.chat_id
    activeUsers[chat_id]["task"].setDeadline(deadline)
    activeUsers[chat_id]["state"] = 3


@bot.message_handler(func=lambda message: activeUsers[message.chat_id]["state"] == 3)
def parseImportance(message):
    """добавить проверку корректности имени(вообще запилить отдельную функцию или класс)"""
    importance = float(message.text)
    chat_id = message.chat_id
    activeUsers[chat_id]["task"].setImportance(importance)
    activeUsers[chat_id]["state"] = 4


@bot.message_handler(func=lambda message: activeUsers[message.chat_id]["state"] == 4)
def parseValidate(message):
    answer = message.text
    chat_id = message.chat_id
    if answer == "Да":
        dbController.addTask()
    activeUsers[chat_id]["state"] = 0


bot.polling()
