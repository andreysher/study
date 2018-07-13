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

    user_state = dict()
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me and event.text:
            print('id{}: "{}"'.format(event.user_id, event.text), end=' ')

            user_id = event.user_id
            text = event.text
            print("Current state: " + str(user_state.get(user_id)))

            # добавление дела
            if text.lower() == u"новое дело" and user_state.get(user_id) == None:
                vk.messages.send(user_id=user_id,
                                 message=u"Введите\n название дела;\n "
                                         + u"Дедлайн в формате ГГГГ-ММ-ДД;\n "
                                         + u"Важность дела от 1 до 10;\n "
                                         + u"Описание дела;")
                user_state[user_id] = 1
                continue

            if re.search(r"\w*[ \t\n]*;[ \t\n]*\d\d\d\d-\d\d-\d\d[ \t\n]*;[ \t\n]*\d\d?[ \t\n]*;[ \t\n]*.*",
                         text) and user_state.get(user_id) == 1:
                args_list = re.split(r"[ \t\n]*;[ \t\n]*", text)
                try:
                    db_driver.addTask(user_id, taskName=args_list[0], deadline=args_list[1],
                                  importance=args_list[2], description=args_list[3])
                except MyDataBaseException as e:
                    if e.message == "not unique task for user":
                        vk.messages.send(user_id=user_id, message=u"Используйте уникальные названия дел")
                        continue
                vk.messages.send(user_id=user_id, message=u"Дело добавлено!")
                user_state[user_id] = None
                continue

            # удаление дела
            if text.lower() == u"удалить дело" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Введите название дела и его категорию \nВ качестве "
                                                          u"разделителя используйте ;")
                user_state[user_id] = 2
                continue

            if user_state.get(user_id) == 2 and text.lower() != "отмена":
                if not re.match(r"\w*", text) and not re.match(r"\w*[ \t\n]*;[ \t\n]*[\w* ]*", text):
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и ; в качестве "
                                                              u"разделителя")
                    continue
                if re.search(r"\w*[ \t\n]*;[ \t\n]*\w*", text):
                    args_list = re.split(r"[ \t\n]*;[ \t\n]*", text)

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
                    tasks_str = u"Вот что еще нужно сделать:"
                    for task in tasks_list:
                        tasks_str += u"\n\n" + task.name\
                                     + u"\n" + str(task.deadline)\
                                     + u"\n" + str(task.importance)\
                                     + u"\n" + task.description
                    vk.messages.send(user_id=user_id, message=tasks_str)
                continue

            # редактировать описание
            if text.lower() == u"редактировать описание" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id,
                                 message=u"Введите название дела ; его категорию (если есть) ; новое описание")
                user_state[user_id] = 3
                continue
            if user_state.get(user_id) == 3 and text.lower() != u"отмена":
                if not re.match(r"\w*[ \t\n]*;[ \t\n]*\w*[ \n\t]*", text) \
                        or not re.match(r"\w*[ \t\n]*;[ \t\n]*\w*[ \n\t]", text):
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

            if user_state.get(user_id) == 4 and text.lower() != u"отмена":
                if not re.match(r"\w+", text):
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и пробел")
                    continue
                try:
                    db_driver.createCategory(userID=user_id, categoryName=text)
                except MyDataBaseException as e:
                    vk.messages.send(user_id=user_id, message=u"Используйте уникальные названия категорий")
                    user_state[user_id] = None
                    continue
                vk.messages.send(user_id=user_id, message=u"Категория добавлена!")
                user_state[user_id] = None
                continue

            # удалить категорию
            if text.lower() == u"удалить категорию" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message=u"Какую категорию удалить?")
                user_state[user_id] = 5
                continue

            # я-то, конечно, категорию удалю, но что будет с тасками из неё?
            if user_state.get(user_id) == 5 and text.lower() != u"отмена":
                if not re.match(r"\w+", text):
                    vk.messages.send(user_id=user_id, message=u"Пожалуйста, используйте только буквы и пробел")
                    continue
                try:
                    db_driver.deleteCategory(userID=user_id, categoryName=text)
                except MyDataBaseException as e:
                    vk.messages.send(user_id=user_id, message=u"Категория не найдена")
                    user_state[user_id] = None
                    continue
                vk.messages.send(user_id=user_id, message=u"Категория удалена")
                user_state[user_id] = None
                continue

            # добавить дело в категорию
            if text.lower() == u"добавить в категорию" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id,
                                 message=u"Введите название дела и его категорию. В качестве разделителя используйте ;")
                user_state[user_id] = 6
                continue

            if user_state.get(user_id) == 6 and text.lower() != u"отмена":
                if re.search(r"\w+[ \t\n]*;[ \t]*\w+", text):
                    args_list = re.split(r"[ \t]*;[ \t\n]*", text)
                    print(args_list)

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

            # удалить дело из категории
            if text.lower() == u"удалить из категории" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id,
                                 message=u"Введите название дела и его категорию. В качестве разделителя используйте ;")
                user_state[user_id] = 7
                continue

            if user_state.get(user_id) == 7 and text.lower() != u"отмена":
                if re.search(r"\w+[ \t\n]*;[ \t]*\w+", text):
                    args_list = re.split(r"[ \t]*;[ \t\n]*", text)
                    print(args_list)
                    try:
                        db_driver.deleteTaskFromCategory(taskName=args_list[0], categoryName=args_list[1],
                                                         userID=user_id)
                    except MyDataBaseException:
                        vk.messages.send(user_id=user_id, message=u"Дело или категория отсутствует. Попробуйте ещё раз")
                        continue
                    vk.messages.send(user_id=user_id, message=u"Дело удалено из категории "
                                                              + "\"" + args_list[1] + "\"")
                    user_state[user_id] = None
                    continue
                else:
                    vk.messages.send(user_id=user_id, message=u"Отсутствует категория! " +
                                                              "Введите название дела и его категорию")

            # показать активные дела из категории
            if text.lower() == u"дела из категории" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id,
                                 message=u"Введите название категории")
                user_state[user_id] = 8
                continue

            if user_state.get(user_id) == 8 and text.lower() != u"отмена":
                try:
                    tasks_list = db_driver.getTasksInCategory(userID=user_id, categoryName=text)
                    tasks_str = u"В категории " + "\"" + text + "\"" + " содержатся следующие дела:\n"

                    for task in tasks_list:
                        tasks_str += u"\n🔵 " + task.name \
                                     + u"\n" + str(task.deadline) \
                                     + u"\n" + str(task.importance) \
                                     + u"\n" + task.description

                    vk.messages.send(user_id=user_id, message=tasks_str)
                except MyDataBaseException as e:
                    if e.message == "No tasks in category":
                        vk.messages.send(user_id=user_id, message=u"В данной категории нет дел")
                    if e.message == "No such category":
                        vk.messages.send(user_id=user_id, message=u"Нет такой категории")

                user_state[user_id] = None
                continue

            # вывод списка категорий
            if text.lower() == u"категории" and user_state.get(user_id) is None:
                category_list = db_driver.getCategories(userID=user_id)
                if len(category_list) == 0:
                    vk.messages.send(user_id=user_id, message="Нет категорий")
                    continue
                answer = "Ваши категории:\n"
                for cat in category_list:
                    answer += u"🔵 " + cat.name + u"\n"
                vk.messages.send(user_id=user_id, message=answer)
                continue

            # показ продуктивности
            if text.lower() == u"продуктивность" and user_state.get(user_id) is None:
                answer = u"Ваша продуктивность равна " +\
                                str(int(db_driver.showProductivity(userID=user_id)*100)) + u" попугаев"
                vk.messages.send(user_id=user_id, message=answer)
                continue

            # отметить как выполненное
            if text.lower() == u"выполнено" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message="Введите название дела")
                user_state[user_id] = 9
                continue

            if user_state.get(user_id) == 9 and text.lower() != u"отмена":
                try:
                    db_driver.endTask(userID=user_id, taskName=text)
                except MyDataBaseException:
                    vk.messages.send(user_id=user_id, message=u"Нет дела с таким названием. Попробуйте ещё раз")
                    user_state[user_id] = None
                    continue
                vk.messages.send(user_id=user_id, message="Дело помечено как выполненное!")
                user_state[user_id] = None
                continue

            # продуктивность по категории
            if text.lower() == u"продуктивность по категории" and user_state.get(user_id) is None:
                vk.messages.send(user_id=user_id, message="Введите название категории")
                user_state[user_id] = 10
                continue

            if user_state.get(user_id) == 10 and text.lower() != u"отмена":
                category_list = db_driver.getCategories(userID=user_id)
                names = list(map(lambda x: x.name, category_list))
                if text not in names:
                    vk.messages.send(user_id=user_id, message="Категории с таким именем не существует!")
                else:
                    answer = u"Ваша продуктивность равна " + \
                             str(int(db_driver.showProductivityInCategory(userID=user_id,
                                                                          categoryName=text) * 100)) + u" попугаев"
                    vk.messages.send(user_id=user_id, message=answer)
                user_state[user_id] = None
                continue

            # показать выполненные дела
            if text.lower() == u"выполненные дела" and user_state.get(user_id) is None:

                tasks_list = db_driver.showCompletedTasks(user_id)
                if not tasks_list:
                    vk.messages.send(user_id=user_id, message=u"У вас нет выполненных дел.")
                else:
                    tasks_str = u"Вот что вы уже сделали:"
                    for task in tasks_list:
                        tasks_str += u"\n\n" + task.name\
                                     + u"\n" + str(task.deadline)\
                                     + u"\n" + str(task.importance)\
                                     + u"\n" + task.description
                    vk.messages.send(user_id=user_id, message=tasks_str)
                continue

            if text.lower() == u"отмена":
                user_state[user_id] = None
                vk.messages.send(user_id=user_id, message=u"Окей, попробуем что-нибудь другое")
                user_state[user_id] = None
                continue

            vk.messages.send(user_id=user_id, message=u"Список доступных команд:"
                                                      + "\nновое дело"
                                                      + "\nактивные дела"
                                                      + "\nудалить дело"
                                                      + "\nвыполнено"
                                                      + "\nвыполненные дела"
                                                      + "\nновая категория"
                                                      + "\nудалить категорию"
                                                      + "\nредактировать описание"
                                                      + "\nдобавить в категорию"
                                                      + "\nудалить из категории"
                                                      + "\nкатегории"
                                                      + "\nпродуктивность"
                                                      + "\nдела из категории"
                                                      + "\nпродуктивность по категории")

        # vk.messages.send(
        #     user_id=event.user_id,
        #     message=text
        # )
        # print('ok')


if __name__ == '__main__':
    main()
