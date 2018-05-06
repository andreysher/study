from sqlalchemy import Column, REAL, DATE, TEXT, INTEGER, ForeignKey, \
    func, PrimaryKeyConstraint
from sqlalchemy.orm import relationship, backref, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
import re

Base = declarative_base()


class Checker:
    def checkForInjection(text):
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
        if Checker.checkForInjection(name):
            self.name = name

    def setDeadline(self, deadline):
        """deadline must be in YYYY-MM-DD format"""
        if Checker.checkForInjection(deadline):
            self.deadline = func.date(deadline)

    def setImportance(self, importance):
        self.importance = importance

    def setDescription(self, description):
        self.description = description


class Task_Category(Base):
    __tablename__ = 'Task_Category'
    # ???
    categoryID = Column(INTEGER, ForeignKey("Categories.id"), primary_key=True)
    taskID = Column(INTEGER, ForeignKey("Task.id"), primary_key=True)


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
        print(task.deadline)
        print(task.name)
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
        actionTasksList = list(map(lambda x: x.Tasks, res))
        return actionTasksList

    def deleteTask(self, userID, taskName, categoryName):
        sesion = self.sessionMaker()
        id = self.findTask(userID, taskName, categoryName=categoryName)
        sesion.query(Tasks).filter_by(id=id).delete()
        sesion.commit()

    def findTask(self, userID, taskName, categoryName):
        """возвращает taskID"""
        sesion = self.sessionMaker()
        if categoryName is None:
            query = sesion.query(Task_User, Tasks).filter(Task_User.userID == userID)
            query = query.join(Tasks, Tasks.id == Task_User.taskID).filter(Tasks.name == taskName)
            id = query.first().Tasks.id
            return id
        else:
            query = sesion.query(Task_User, Tasks, Categories, Task_Category) \
                .filter(Task_User.userID == userID and Tasks.id == Task_User.taskID \
                        and Categories.name == categoryName and Task_Category.categoryID == Categories.id \
                        and Categories.userID == userID)
            id = query.first().Tasks.id
            return id

    def changeTask(self, userID, taskID, **kwargs):
        sesion = self.sessionMaker()
        task = sesion.query(Tasks).filter(Tasks.id == taskID).first()
        taskUser = sesion.query(Task_User).filter(Task_User.taskID == taskID and Task_User.userID == userID).first()
        if taskUser.taskID == task.id and task is not None and taskUser is not None:
            sesion.query(Tasks).filter(Tasks.id == taskID).update(kwargs)
            sesion.commit()

    def createCategory(self, userID, categoryName):
        sesion = self.sessionMaker()
        category = Categories()
        category.name = categoryName
        category.userID = userID
        category.productivity = 0.0
        sesion.add(category)
        sesion.commit()

    def _checkCategoryExist(self, sesion, categoryName, userID):
        category = sesion.query(Categories).filter(
            Categories.name == categoryName and Categories.userID == userID).first()
        if category is None:
            raise MyDataBaseException("Add task to category: Category in None")
        else:
            return category

    def _checkTaskForUser(self, sesion, taskName, userID):
        tasks = sesion.query(Tasks).filter(Tasks.name == taskName).all()
        if len(tasks) == 0:
            raise MyDataBaseException("Add task to category: Task name is invalid")
        tasksIDs = map(lambda x: x.id, tasks)
        tu = sesion.query(Task_User).filter(Task_User.userID == userID and Task_User.taskID.in_(tasksIDs)).first()
        if tu is None:
            raise MyDataBaseException("Add task to category: no such task for such user")
        return tu

    def addTaskToCategory(self, taskName, categoryName, userID):
        sesion = self.sessionMaker()
        category = self._checkCategoryExist(sesion, categoryName, userID)
        tu = self._checkTaskForUser(sesion, taskName, userID)
        tCatCheck = sesion.query(Task_Category).filter(
            Task_Category.categoryID == category.id and Task_Category.taskID == tu.taskID).first()
        if tCatCheck is None:
            tc = Task_Category()
            tc.taskID = tu.taskID
            tc.categoryID = category.id
            sesion.add(tc)
            sesion.commit()

    def changeCategoryForTask(self, taskName, oldCategoryName, newCategoryName, userID):
        pass

    def deleteCategory(self):
        pass

    def showProductivity(self):
        pass

    def showProductivityInCategory(self):
        pass

    # TODO:Переписать все под имена сущностей всех, создание сущностей тоже под набор их атрибутов


driver = DBDriver()
# driver.addTask(1, taskName="myTask", deadline="2018-11-30", importance=1, description="123456")
tasks = driver.showActiveTasks(1)
driver.deleteTask(1, "myTask", categoryName="123456789123456789")
# # driver.changeTask(1, 27, name="newName")
# driver.createCategory(1, "myCategory")
# driver.addTaskToCategory("myTask", "myCategory", 1)
