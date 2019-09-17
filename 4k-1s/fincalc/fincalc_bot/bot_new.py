# -*- coding: utf-8 -*-

import requests

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_utils import VkKeyboard, VkKeyboardColor
# from data_base import DBDriver, MyDataBaseException
import re
import config
from database_api import DBDriver


def send_message(vk, event, message_text=None, keyboard=None):
    user_id = event.user_id
    if not keyboard and message_text:
        vk.messages.send(user_id=user_id, message=message_text)
    elif keyboard and not message_text:
        vk.messages.send(user_id=user_id, keyboard=keyboard)
    elif keyboard and message_text:
        vk.messages.send(user_id=user_id, message=message_text, keyboard=keyboard)


def show_history(vk, event, user_history):
    for operation in user_history:
        send_message(vk, event, message_text=' '.join([operation['type'], str(operation['amount']), operation['comment']]))


def main():
    global comment_keyboard
    vk_session = vk_api.VkApi(token=config.token)
    vk = vk_session.get_api()

    DB = DBDriver()

    longpoll = VkLongPoll(vk_session)

    user_state = {}
    user_op = {}

    comment_keyboard = VkKeyboard(one_time=True)
    comment_keyboard.add_button('Без комментариев', VkKeyboardColor.PRIMARY)

    keyboard_1 = VkKeyboard(one_time=True)
    keyboard_1.add_button('Доход', VkKeyboardColor.POSITIVE)
    keyboard_1.add_button('Расход', VkKeyboardColor.NEGATIVE)
    keyboard_1.add_button('История', VkKeyboardColor.DEFAULT)

    keyboard_categ = VkKeyboard(one_time=True)
    keyboard_categ.add_button('Продукты', VkKeyboardColor.DEFAULT)
    keyboard_categ.add_button('Одежда', VkKeyboardColor.DEFAULT)
    keyboard_categ.add_button('Транспорт', VkKeyboardColor.DEFAULT)
    keyboard_categ.add_button('Развлечения', VkKeyboardColor.DEFAULT)
    keyboard_categ.add_button('Жилье', VkKeyboardColor.DEFAULT)
    keyboard_categ.add_button('Другое', VkKeyboardColor.DEFAULT)

    keyboard_info = VkKeyboard()
    keyboard_info.add_button('Доходы', VkKeyboardColor.DEFAULT)
    keyboard_info.add_button('Расходы', VkKeyboardColor.DEFAULT)
    keyboard_info.add_button('Вся', VkKeyboardColor.DEFAULT)

    period_keyboard = VkKeyboard()
    period_keyboard.add_button('День', VkKeyboardColor.DEFAULT)
    period_keyboard.add_button('Неделя', VkKeyboardColor.DEFAULT)
    period_keyboard.add_button('Месяц', VkKeyboardColor.DEFAULT)
    period_keyboard.add_button('Год', VkKeyboardColor.DEFAULT)

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:

            if not ((event.text in config.relevant_comands) or (event.user_id in user_state)):
                continue

            if event.text == 'Начать':
                user_state[event.user_id] = 'start'
                send_message(vk, event, message_text='Ну погнали!',
                             keyboard=keyboard_1.get_keyboard())
                continue

            if event.text == 'Доход':
                send_message(vk, event, 'Пожалуйста, введите сумму дохода')
                user_state[event.user_id] = 'sum+'
                user_op[event.user_id] = {'name': 'Доход'}
                continue

            if event.text == 'Расход':
                send_message(vk, event, 'Пожалуйста, введите сумму расхода')
                user_state[event.user_id] = 'sum-'
                user_op[event.user_id] = {'name': 'Расход'}
                continue

            if event.text == 'История':
                user_state[event.user_id] = 'history'
                send_message(vk, event, 'Какая история вас интересует?',
                             keyboard=keyboard_info.get_keyboard())
                continue

            if user_state[event.user_id] == 'sum+':
                plus_money = int(event.text)
                user_op[event.user_id]['amount'] = plus_money
                user_op[event.user_id]['category'] = 'доход'
                send_message(vk, event, 'Вы можете оставить комментарий',
                             keyboard=comment_keyboard.get_keyboard())
                user_state[event.user_id] = 'wait_comment'
                continue

            if user_state[event.user_id] == 'wait_comment':
                comment = event.text
                user_op[event.user_id]['comment'] = comment
                op_info = user_op[event.user_id]
                DB.add_operation(event.user_id, op_info['name'], op_info['amount'], op_info['comment'],
                                 op_info['category'])
                send_message(vk, event, 'Ваша операция записана!',
                             keyboard=keyboard_1.get_keyboard())
                user_state[event.user_id] = 'start'
                continue

            if user_state[event.user_id] == 'sum-':
                minus_money = int(event.text)
                user_op[event.user_id]['amount'] = minus_money
                send_message(vk, event, message_text='Укажите категорию расхода',
                             keyboard=keyboard_categ.get_keyboard())
                user_state[event.user_id] = 'wait_category'
                continue

            if user_state[event.user_id] == 'wait_category':
                category = event.text
                user_op[event.user_id]['category'] = category
                user_state[event.user_id] = 'wait_comment'
                send_message(vk, event, 'Вы можете оставить комментарий',
                             keyboard=comment_keyboard.get_keyboard())
                continue

            if user_state[event.user_id] == 'history':
                info_type = event.text

                if info_type not in ['Доходы', 'Расходы', 'Вся']:
                    send_message(vk, event, 'Нет такой информации')
                    continue

                user_state[event.user_id] = {'history': info_type}
                send_message(vk, event,
                             'Уточните период, за который хотите получить историю',
                             keyboard=period_keyboard.get_keyboard())
                continue

            if 'history' in user_state[event.user_id] and event.text == 'День':
                user_state[event.user_id] = 'start'
                show_history(vk, event, DB.get_user_history(event.user_id))
                send_message(vk, event, 'Что-нибудь еще?', keyboard=keyboard_1.get_keyboard())

            if 'history' in user_state[event.user_id] and event.text == 'Неделя':
                user_state[event.user_id] = 'start'
                show_history(vk, event, DB.get_user_history(event.user_id))
                send_message(vk, event, 'Что-нибудь еще?', keyboard=keyboard_1.get_keyboard())

            if 'history' in user_state[event.user_id] and event.text == 'Месяц':
                user_state[event.user_id] = 'start'
                show_history(vk, event, DB.get_user_history(event.user_id))
                send_message(vk, event, 'Что-нибудь еще?', keyboard=keyboard_1.get_keyboard())

            if 'history' in user_state[event.user_id] and event.text == 'Год':
                user_state[event.user_id] = 'start'
                show_history(vk, event, DB.get_user_history(event.user_id))
                send_message(vk, event, 'Что-нибудь еще?', keyboard=keyboard_1.get_keyboard())


if __name__ == '__main__':
    main()
