# -*- coding: utf-8 -*-

import requests

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from data_base import DBDriver, MyDataBaseException
import re
import config


def main():
    vk_session = vk_api.VkApi(token=config.token)
    vk = vk_session.get_api()

    longpoll = VkLongPoll(vk_session)
    db_driver = DBDriver()

    valid_commands = [u"новое дело", u"удалить дело",
                      u"активные дела", u"редактировать описание",
                      u"новая категория", u"удалить категорию",
                      u"продуктивность", u"дела из категории",
                      u"добавить в категорию", u"удалить из категории",
                      u"продуктивность по категории",
                      u"отметить как выполненное"]

    user_state = dict()
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            print('id{}: "{}"'.format(event.user_id, event.text), end=' ')

            user_id = event.user_id
            text = event.text

            # добавление дела
            if text.lower() == u"новое дело" and user_state.get(user_id) == None:
                vk.messages.send(user_id=user_id,
                                 message=u"Введите\n название дела;\n "
                                         + u"Дедлайн в формате ГГГГ-ММ-ДД;\n "
                                         + u"Важность дела от 1 до 10;\n "
                                         + u"Описание дела;")
                user_state[user_id] = 1
                continue

            if re.search(r'\w*[ \t\n]*;[ \t\n]*\d\d\d\d-\d\d-\d\d[ \t\n]*;[ \t\n]*\d\d?[ \t\n]*;[ \t\n]*.*',
                         text) and user_state.get(user_id) == 1:

                args_list = re.split(r'[ \t\n]*;[ \t\n]*', text)
                # if not re.match(r'\d\d\d\d-\d\d-\d\d', args_list[1]):
                #     vk.messages.send(user_id=user_id, message=u"Неправильный формат даты" + args_list[1])
                #     continue

                db_driver.addTask(user_id, taskName=args_list[0], deadline=args_list[1], importance=args_list[2], description=args_list[3])
                vk.messages.send(user_id=user_id, message=u"Дело добавлено!")
                user_state[user_id] = None

            # удаление дела
            if text.lower() == u"удалить дело" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Введите название дела и его категорию \nВ качестве "
                                                          u"разделителя используйте ;")
                user_state[user_id] = 2
                continue

            if user_state.get(user_id) == 2:
                if not re.match(r"\w*", text) and not re.match(r"\w*[ \t\n]*;[ \t\n]*[\w* ]*", text):
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и ; в качестве "
                                                              u"разделителя")
                    continue
                if re.search(r'\w*[ \t\n]*;[ \t\n]*\w*', text):
                    args_list = re.split(r'[ \t\n]*;[ \t\n]*', text)

                    try:
                        db_driver.deleteTask(userID=user_id, taskName=args_list[0], categoryName=args_list[1])
                    except MyDataBaseException:
                        vk.messages.send(user_id=user_id, message=u"Дело не найдено. Попробуйте ещё раз")
                        continue
                else:
                    try:
                        db_driver.deleteTask(userID=user_id, taskName=text)
                    except MyDataBaseException:
                        vk.messages.send(user_id=user_id, message=u"Дело не найдено. Попробуйте ещё раз")
                        continue

                vk.messages.send(user_id=user_id, message=u"Дело удалено!")
                user_state[user_id] = None
                continue

            # показать активные дела
            if text.lower() == u"активные дела" and user_state.get(user_id) is None:

                tasks_list = db_driver.showActiveTasks(user_id)
                if not tasks_list:
                    vk.messages.send(user_id=user_id, message=u"У вас нет активных дел. Можете отдыхать дальше :)")
                else:
                    vk.messages.send(user_id=user_id, message=u"Вот что еще нужно сделать:")
                    for task in tasks_list:
                        vk.messages.send(user_id=user_id,
                                         message=u"" + task.name + u"\n" + str(task.deadline) + u"\n" + str(
                                             task.importance) + u"\n" + task.description)

                continue

            # редактировать описание
            if text.lower() == u"редактировать описание" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Введите название дела ; его категорию(если есть) ; новое описание")
                user_state[user_id] = 3
                continue
            if user_state.get(user_id) == 3:
                if not re.match(r"\w*[ \t\n]*;[ \t\n]*\w*[ \n\t]*",text) and not re.match(r"\w*[ \t\n]*;[ \t\n]*\w*[ \n\t]", text):
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и ; в качестве "
                                                              u"разделителя")
                    continue
                args_list = re.split(r"[ \t\n]*;[ \t\n]*", text)
                if len(args_list) == 2:
                    db_driver.changeTaskDescription(user_id, taskName=args_list[0], description=args_list[1])
                if len(args_list) == 3:
                    db_driver.changeTaskDescription(user_id, taskName=args_list[0], categoryName=args_list[1],
                                                    description=args_list[3])
                    continue
            # добавить категорию
            if text.lower() == u"новая категория" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Как назвать категорию?")
                user_state[user_id] = 4
                continue

            if user_state.get(user_id) == 4:
                if re.match(r"\w*", text) is None:
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и пробел")
                    continue
                db_driver.createCategory(userID=user_id, categoryName=text)
                vk.messages.send(user_id=user_id, message=u"Категория добавлена!")
                user_state[user_id] = None
                continue

            # удалить категорию
            if text.lower() == u"удалть категорию" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Какую категорию удалить?")
                user_state[user_id] = 5
                continue

            # я-то, конечно, категорию удалю, но что будет с тасками из неё?
            if user_state.get(user_id) == 5:
                if not re.match(r"\w*", text):
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и пробел")
                    continue
                db_driver.deleteCategory(userID=user_id, categoryName=text)
                vk.messages.send(user_id=user_id, message=u"Категория удалена")
                user_state[user_id] = None
                continue

            # добавить дело в категорию
            if text.lower() == u"добавить в категорию" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Введите название дела и его категорию в качестве разделителя используйте ;")
                user_state[user_id] = 6
                continue

            if user_state.get(user_id) == 6:
                if re.search(r'\w*[ \t\n]*;[ \t]*\w*', text):
                    args_list = re.split(r'[ \t]*;[ \t\n]*]', text)

                    try:
                        db_driver.addTaskToCategory(taskName=args_list[0], categoryName=args_list[1], userID=user_id)
                    except MyDataBaseException:
                        vk.messages.send(user_id=user_id, message=u"Дело или категория отсутствует. Попробуйте ещё раз")
                        continue
                    vk.messages.send(user_id=user_id, message=u"Дело добавлено в категорию!")
                    user_state[user_id] = None
                    continue
                else:
                    vk.messages.send(user_id=user_id, message=u"Отсутствует категория! " +
                                                              "Введите название дела и его категорию")

            # if text == u"отмена":
            #     user_state[user_id] = None
            #     vk.messages.send(user_id=user_id, message=u"Окей, попробуем что-нибудь другое")
            #     continue
            # user_state[user_id] = None
            vk.messages.send(user_id=user_id, message=u"Что-то пошло не так :("
                                                      + "\nСписок доступных команд:"
                                                      + "\nновое дело"
                                                      + "\nактивные дела"
                                                      + "\nудалить дело"
                                                      + "\nновая категория"
                                                      + "\nудалить категорию"
                                                      + "\nредактировать описание"
                                                      + "\nдобавить в категорию")

        # vk.messages.send(
        #     user_id=event.user_id,
        #     message=text
        # )
        # print('ok')


if __name__ == '__main__':
    main()
