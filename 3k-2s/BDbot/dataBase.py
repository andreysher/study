from sqlalchemy import Column, REAL, DATE, TEXT, INTEGER, ForeignKey, \
    func, PrimaryKeyConstraint
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
import re

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
    name = Column(TEXT, nullable=False)
    deadline = Column(DATE, nullable=False)
    importance = Column(REAL, nullable=False)
    startTime = Column(DATE, default=func.now(), nullable=False)
    endTime = Column(DATE, nullable=True)
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

    def addTask(self, userID, taskName, deadline, importance, description):
        task = Tasks()
        task.setName(taskName)
        task.setDeadline(deadline)
        task.setImportance(importance)
        task.setDescription(description)
        sesion = self.sessionMaker()
        if task.name is None:
            raise MyDataBaseException("Add unnamed task")
        if task.deadline is None:
            raise MyDataBaseException("Add task without deadline")
        if task.importance is None:
            raise MyDataBaseException("Add task without importance")
        sesion.add(task)
        sesion.commit()
        tu = Task_User()
        tu.taskID = task.id
        tu.userID = userID
        sesion.add(tu)
        sesion.commit()

    def showActiveTasks(self, userID):
        sesion = self.sessionMaker()
        query = sesion.query(Task_User, Tasks).filter(Task_User.userID == userID)
        query = query.join(Tasks, Tasks.id == Task_User.taskID)
        res = query.all()
        # for task in res:
        #     print(task.Tasks.id)
        actionTasksList = list(map(lambda x: x.Tasks, res))
        return actionTasksList

    def deleteTask(self, userID, taskName, categoryName=None):
        sesion = self.sessionMaker()
        task = self.findTask(userID, taskName, categoryName=categoryName)
        if task is None:
            raise MyDataBaseException("Can not find such task")
        id = task.id
        sesion.query(Tasks).filter_by(id=id).delete()
        sesion.query(Task_User).filter(Task_User.userID == userID and Task_User.taskID == id) \
            .delete()
        sesion.query(Task_Category).filter(Task_Category.taskID == id).delete()
        sesion.commit()

    def findTask(self, userID, taskName, categoryName=None):
        """возвращает task"""
        sesion = self.sessionMaker()
        if categoryName is None:
            query = sesion.query(Task_User, Tasks).filter(Task_User.userID == userID)
            query = query.join(Tasks, Tasks.id == Task_User.taskID).\
                filter(Tasks.name == taskName)
            task = query.first()
            if task is None:
                return None
            return task.Tasks
        else:
            query = sesion.query(Task_User, Tasks, Categories, Task_Category) \
                .filter(Task_User.userID == userID and Tasks.id == Task_User.taskID \
                        and Categories.name == categoryName and Task_Category.categoryID == Categories.id \
                        and Categories.userID == userID)
            queryRes = query.first()
            if queryRes is None:
                return None
            return queryRes.Tasks

    def createCategory(self, userID, categoryName):
        sesion = self.sessionMaker()
        category = Categories()
        category.setName(categoryName)
        category.userID = userID
        category.productivity = 0.0
        sesion.add(category)
        sesion.commit()

    def addTaskToCategory(self, taskName, categoryName, userID):
        sesion = self.sessionMaker()
        category = sesion.query(Categories).filter(Categories.userID == userID \
                                                   and Categories.name == categoryName).first()
        if category is None:
            raise MyDataBaseException("No such category")
        task = self.findTask(userID, taskName)
        if self.findTask(userID, taskName, categoryName) is None and task is not None:
            tc = Task_Category()
            tc.categoryID = category.id
            tc.taskID = task.id
            sesion.add(tc)
            sesion.commit()

    def changeTaskDescription(self, userID, taskName, categoryName, description):
        sesion = self.sessionMaker()
        task = self.findTask(userID, taskName, categoryName=categoryName)
        if task is None:
            raise MyDataBaseException("Can not find such task")
        task.setDescription(description)
        sesion.query(Tasks).filter(Tasks.id == task.id).update({'description': description})
        sesion.commit()

    def deleteTaskFromCategory(self, userID, taskName, categoryName):
        sesion = self.sessionMaker()
        task = self.findTask(userID, taskName, categoryName=categoryName)
        category = sesion.query(Categories).filter(Categories.userID == userID \
                                                   and Categories.name == categoryName).first()
        if task is None:
            raise MyDataBaseException("no such task in category")
        if category is None:
            raise MyDataBaseException("no such category for user")
        sesion.query(Task_Category).filter(Task_Category.taskID == task.id and Categories.id == category.id).delete()

    def deleteCategory(self, userID, categoryName):
        sesion = self.sessionMaker()
        category = sesion.query(Categories).filter(Categories.userID == userID \
                                                   and Categories.name == categoryName).first()
        if category is None:
            raise MyDataBaseException("Can not find category")
        sesion.query(Categories).filter(Categories.id == category.id).delete()
        sesion.query(Task_Category).filter(Task_Category.categoryID == category.id).delete()
        sesion.commit()

    def showProductivity(self):
        pass

    def getTasksInCategory(self, userID, categoryName):
        sesion = self.sessionMaker()
        query = sesion.query(Categories, Tasks, Task_Category). \
            filter(Categories.userID == userID and \
                   Categories.name == categoryName)
        query = query.join(Task_Category, Task_Category.categoryID == Categories.id)
        query = query.join(Tasks, Task_Category.taskID == Tasks.id)
        res = query.all()
        if res is None:
            raise MyDataBaseException("No tasks in category")
        tasksInCat = list(map(lambda x: x.Tasks, res))
        return tasksInCat

    def showProductivityInCategory(self):
        pass

    def endTask(self, userID, taskName, categoryName=None):
        sesion = self.sessionMaker()
        query = sesion.query(Task_User, Tasks).filter(Task_User.userID == userID)
        query = query.join(Tasks, Tasks.id == Task_User.userID).filter(Tasks.name == taskName)
        task = query.first()
        if task is None:
            raise MyDataBaseException("can not find task for end")
        task = task.Tasks
        sesion.query(Tasks).filter(Tasks.id == task.id).update({'actual': 0})
        sesion.commit()

    # TODO:Переписать все под имена сущностей всех, создание сущностей тоже под набор их атрибутов


driver = DBDriver()
# driver.addTask(1, taskName="myTask", deadline="2018-11-30", importance=1, description="123456")
tasks = driver.showActiveTasks(1)
# driver.deleteTask(1, "myTask")
# driver.changeTask(1, 27, name="newName")
# driver.createCategory(1, "myCategory")
driver.addTaskToCategory("myTask", "myCategory", 1)
# driver.changeTaskDescription(1, "myTask", "myCategory", "bla-bla-bla")
# driver.deleteCategory(1, "myCategory")
# driver.deleteTaskFromCategory(1, "myTask", "myCategory")
tasksList = driver.getTasksInCategory(1, "myCategory")
driver.endTask(1, "myTask")