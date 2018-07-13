# -*- coding: utf-8 -*-

import datetime
import re
from datetime import datetime

from sqlalchemy import Column, REAL, DATE, TEXT, INTEGER, ForeignKey, \
    func
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Checker:
    def planeText(text):
        res = bool(re.search(r"[():;]", text))
        if res:
            raise MyDataBaseException("Invalid symbols in task name")
        return not res

    def checkDateFormat(dateText):
        # TODO:Сделать проверку на количество дней в месяце
        date = dateText.split('-')
        if len(date) == 3 and int(date[0]) >= 2018 and 0 < int(date[1]) <= 12 \
                and 0 < int(date[2]) <= 31:
            return True


class Tasks(Base):
    __tablename__ = 'Tasks'
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    name = Column(TEXT, nullable=False, unique=True)
    deadline = Column(DATE, nullable=False)
    importance = Column(REAL, nullable=False)
    startTime = Column(DATE, default=datetime.now(), nullable=False)
    endTime = Column(DATE, default=datetime.now(), nullable=True)
    actual = Column(INTEGER, nullable=False, default=1)
    description = Column(TEXT, nullable=True)

    def setName(self, name):
        if Checker.planeText(name):
            self.name = name

    def setDeadline(self, deadline):
        """deadline must be in YYYY-MM-DD format"""
        if Checker.checkDateFormat(deadline):
            self.deadline = func.date(deadline)

    def setImportance(self, importance):
        self.importance = importance

    def setDescription(self, description):
        self.description = description


class Task_Category(Base):
    __tablename__ = 'Task_Category'
    # ???
    categoryID = Column(INTEGER, ForeignKey("Categories.id"), primary_key=True)
    taskID = Column(INTEGER, ForeignKey("Tasks.id"), primary_key=True)


class Task_User(Base):
    __tablename__ = 'Task_User'
    userID = Column(INTEGER, primary_key=True, unique=False)
    taskID = Column(INTEGER, ForeignKey("Tasks.id"), primary_key=True)


class Categories(Base):
    __tablename__ = 'Categories'
    id = Column(INTEGER, primary_key=True, autoincrement=True)
    name = Column(TEXT, nullable=False)
    userID = Column(INTEGER, nullable=False, unique=True)
    productivity = Column(REAL, nullable=False, default=0.0)

    def setName(self, name):
        if Checker.planeText(name):
            self.name = name


class User_Productivity(Base):
    __tablename__ = 'User_Productivity'
    userID = Column(INTEGER, primary_key=True, unique=True)
    updateTime = Column(INTEGER)
    productivity = Column(REAL)


class MyDataBaseException(Exception):
    def __init__(self, text):
        self.message = text

    def __repr__(self):
        print(self.message)


class DBDriver:
    engine = None

    def __init__(self):
        self.engine = create_engine('sqlite:///./ToDoBot.sqlite3', echo=True)
        self.sessionMaker = sessionmaker(bind=self.engine)

    @staticmethod
    def countUrgency(earlyDate, laterDate):
        d1 = earlyDate.split('-')
        d2 = laterDate.split('-')
        date_1 = datetime.datetime(int(d1[0]), int(d1[1]), int(d1[2]))
        date_2 = datetime.datetime(int(d2[0]), int(d2[1]), int(d2[2]))
        delta = date_2 - date_1
        return delta.days

    def addTask(self, userID, taskName, deadline, importance, description):
        # TODO: проверка имени на уникальность
        task = Tasks()
        task.setName(taskName)
        task.setDeadline(deadline)
        task.setImportance(importance)
        task.setDescription(description)
        session = self.sessionMaker()
        if task.name is None:
            raise MyDataBaseException("Add unnamed task")
        if task.deadline is None:
            raise MyDataBaseException("Add task without deadline")
        if task.importance is None:
            raise MyDataBaseException("Add task without importance")
        if session.query(Tasks, Task_User).filter(Task_User.userID == userID) \
                .filter(Tasks.name == task.name).first() is not None:
            raise MyDataBaseException("not unique task for user")
        session.add(task)
        session.commit()
        tu = Task_User()
        tu.taskID = task.id
        tu.userID = userID
        session.add(tu)
        session.commit()

    def showActiveTasks(self, userID):
        session = self.sessionMaker()
        query = session.query(Task_User, Tasks).filter(Task_User.userID == userID)
        query = query.join(Tasks, Tasks.id == Task_User.taskID).filter(Tasks.actual == 1)
        res = query.all()
        # for task in res:
        #     print(task.Tasks.id)
        actionTasksList = list(map(lambda x: x.Tasks, res))
        return actionTasksList

    def deleteTask(self, userID, taskName, categoryName=None):
        session = self.sessionMaker()
        task = self.findTask(userID, taskName, categoryName=categoryName)
        if task is None:
            raise MyDataBaseException("Can not find such task")
        id = task.id
        session.query(Tasks).filter_by(id=id).delete()
        session.query(Task_User).filter(Task_User.userID == userID).filter(Task_User.taskID == id).delete()
        session.query(Task_Category).filter(Task_Category.taskID == id).delete()
        session.commit()

    def findTask(self, userID, taskName, categoryName=None):
        """возвращает task"""
        session = self.sessionMaker()
        if categoryName is None:
            query = session.query(Task_User, Tasks).filter(Task_User.userID == userID)
            query = query.join(Tasks, Tasks.id == Task_User.taskID). \
                filter(Tasks.name == taskName)
            task = query.first()
            if task is None:
                return None
            return task.Tasks
        else:
            task = session.query(Task_User, Tasks).join(Tasks, Task_User.taskID == Tasks.id) \
                .filter(Task_User.userID == userID).filter(Tasks.name == taskName).first().Tasks
            cat = session.query(Categories).filter(Categories.userID == userID)\
                .filter(Categories.name == categoryName).first()
            queryRes = session.query(Task_Category, Tasks).join(Tasks, Task_Category.taskID == Tasks.id)\
                .filter(Task_Category.categoryID == cat.id)\
                .filter(Task_Category.taskID == task.id).first()
            if queryRes is None:
                return None
            return queryRes.Tasks

    def createCategory(self, userID, categoryName):
        session = self.sessionMaker()
        category = Categories()
        category.setName(categoryName)
        category.userID = userID
        category.productivity = 0.0
        print(categoryName)
        print()
        if session.query(Categories).filter(Categories.userID == userID)\
                .filter(Categories.name == categoryName).first() is not None:
            raise MyDataBaseException("not unique category for user")
        session.add(category)
        session.commit()

    def addTaskToCategory(self, taskName, categoryName, userID):
        session = self.sessionMaker()
        category = session.query(Categories).filter(Categories.userID == userID). \
            filter(Categories.name == categoryName).first()
        if category is None:
            raise MyDataBaseException("No such category")
        task = self.findTask(userID, taskName)
        r = self.findTask(userID, taskName, categoryName=categoryName)
        print()
        print(r)
        print(task)
        print()
        if r is None and task is not None:
            print("in if")
            tc = Task_Category()
            tc.categoryID = category.id
            tc.taskID = task.id
            session.add(tc)
            session.commit()

    def changeTaskDescription(self, userID, taskName, description="", categoryName=None):
        session = self.sessionMaker()
        task = self.findTask(userID, taskName, categoryName=categoryName)
        if task is None:
            raise MyDataBaseException("Can not find such task")
        task.setDescription(description)
        session.query(Tasks).filter(Tasks.id == task.id).update({'description': description})
        session.commit()

    def deleteTaskFromCategory(self, userID, taskName, categoryName):
        session = self.sessionMaker()
        task = self.findTask(userID, taskName, categoryName=categoryName)
        category = session.query(Categories).filter(Categories.userID == userID)\
            .filter(Categories.name == categoryName).first()
        if task is None:
            raise MyDataBaseException("no such task in category")
        if category is None:
            raise MyDataBaseException("no such category for user")
        session.query(Task_Category).filter(Task_Category.taskID == task.id)\
            .filter(Task_Category.categoryID == category.id).delete()
        session.commit()

    def getCategories(self, userID):
        session = self.sessionMaker()
        categories = session.query(Categories).filter(Categories.userID == userID).all()
        return categories

    def deleteCategory(self, userID, categoryName):
        session = self.sessionMaker()
        category = session.query(Categories).filter(Categories.userID == userID)\
                .filter(Categories.name == categoryName).first()
        if category is None:
            raise MyDataBaseException("Can not find category")
        session.query(Categories).filter(Categories.id == category.id).delete()
        session.query(Task_Category).filter(Task_Category.categoryID == category.id).delete()
        session.commit()

    def showProductivity(self, userID):
        session = self.sessionMaker()
        query = session.query(Task_User, Tasks).filter(Task_User.userID == userID)
        query = query.join(Tasks, Tasks.id == Task_User.taskID)

        tasks = list(map(lambda x: x.Tasks, query.all()))

        allImportance = sum([t.importance for t in tasks])
        goodImportance = sum([t.importance for t in tasks if t.actual == 0])
        if allImportance == 0:
            return 0
        return goodImportance / allImportance

    def getTasksInCategory(self, userID, categoryName):
        session = self.sessionMaker()
        cat = session.query(Categories).filter(Categories.userID == userID)\
            .filter(Categories.name == categoryName) \
            .first()
        if cat is None:
            raise MyDataBaseException("No such category")
            return
        res = session.query(Tasks, Task_Category).join(Task_Category, Tasks.id == Task_Category.taskID) \
            .filter(Task_Category.categoryID == cat.id).all()
        if res is None:
            raise MyDataBaseException("No tasks in category")
        tasksInCat = list(map(lambda x: x.Tasks, res))
        return tasksInCat

    def showProductivityInCategory(self, userID, categoryName):
        session = self.sessionMaker()
        query = session.query(Categories, Tasks, Task_Category). \
            filter(Categories.userID == userID)
        query = query.join(Task_Category, Task_Category.categoryID == Categories.id)
        query = query.join(Tasks, Task_Category.taskID == Tasks.id).filter(Categories.name == categoryName)

        tasks = list(map(lambda x: x.Tasks, query.all()))

        allImportance = sum([t.importance for t in tasks])
        goodImportance = sum([t.importance for t in tasks if t.actual == 0])
        if allImportance == 0:
            return 0
        return goodImportance / allImportance

    def endTask(self, userID, taskName):
        session = self.sessionMaker()
        query = session.query(Task_User, Tasks).filter(Task_User.userID == userID)
        query = query.join(Tasks, Tasks.id == Task_User.taskID).filter(Tasks.name == taskName)
        task = query.first()
        if task is None:
            raise MyDataBaseException("can not find task for end")
        task = task.Tasks
        session.query(Tasks).filter(Tasks.id == task.id).update({'actual': 0})
        session.commit()

    def showCompletedTasks(self, userID):
        session = self.sessionMaker()
        query = session.query(Task_User, Tasks).filter(Task_User.userID == userID)
        query = query.join(Tasks, Tasks.id == Task_User.taskID).filter(Tasks.actual == 0)
        res = query.all()
        completedTasksList = list(map(lambda x: x.Tasks, res))
        return completedTasksList

    # TODO:Переписать все под имена сущностей всех, создание сущностей тоже под набор их атрибутов
