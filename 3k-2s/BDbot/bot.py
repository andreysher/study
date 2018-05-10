# -*- coding: utf-8 -*-

import requests

import vk_api
from vk_api import VkUpload
from vk_api.longpoll import VkLongPoll, VkEventType
from dataBase import DBDriver
import re

def main():
    session = requests.Session()

    # Авторизация пользователя:
    """
    login, password = 'python@vk.com', 'mypassword'
    vk_session = vk_api.VkApi(login, password)
    try:
        vk_session.auth()
    except vk_api.AuthError as error_msg:
        print(error_msg)
        return
    """

    # Авторизация группы:
    # при передаче token вызывать vk_session.auth не нужно
    vk_session = vk_api.VkApi(token='токен с доступом к сообщениям и фото')

    vk = vk_session.get_api()

    upload = VkUpload(vk_session)  # Для загрузки изображений
    longpoll = VkLongPoll(vk_session)
    dbDriver = DBDriver()
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            print('id{}: "{}"'.format(event.user_id, event.text), end=' ')

            if event.text == "Добавить дело":
                vk.messages.send(user_id=event.user_id,
                                 message="Введите название дела,"
                                         +" Дедлайн в формате ГГГГ-ММ-ДД,"
                                          +" Важность дела от 1 до 10, "
                                           +" Описание дела")
            if re.match(r"\w*;[ \t]*\d\d\d\d-\d\d-\d\d;[ \t]*;\d\d?;[ \t]*.*", event.text):
                argsList = event.text.split(';')
                dbDriver.addTask(event.user_id, argsList[0], argsList[1], argsList[2], argsList[3])

        # vk.messages.send(
        #     user_id=event.user_id,
        #     message=text
        # )
        # print('ok')


if __name__ == '__main__':
    main()