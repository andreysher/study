# -*- coding: utf-8 -*-


from datetime import datetime

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import sqlalchemy as sa


class MyDataBaseException(Exception):
    def __init__(self, text):
        self.message = text

    def __repr__(self):
        print(self.message)


class DBDriver:
    engine = None

    def __init__(self):
        self.engine = create_engine('sqlite:///./fincalc.db', echo=False)
        self.sessionMaker = sessionmaker(bind=self.engine)
        self.md = sa.MetaData(bind=self.engine)

    def add_operation(self, user_id, op_type, amount, comment, category):
        time = datetime.now()
        category_id = self.__create_category(category)
        print(category_id)
        operation_id = self.__create_operation(op_type, amount, comment, category_id)
        print(operation_id)
        self.__create_user_operation(user_id, operation_id, time)

    def get_user_history(self, user_id):
        user_operation = sa.Table('user_operation', self.md, autoload=True, autoload_with=self.engine)
        operation = sa.Table('operation', self.md, autoload=True, autoload_with=self.engine)
        category = sa.Table('category', self.md, autoload=True, autoload_with=self.engine)
        session = self.sessionMaker()
        ds = session\
            .query(user_operation, operation, category)\
            .join(operation)\
            .join(category)\
            .all()
        ds = list(filter(lambda x: x[0] == user_id, ds))
        ds = list(map(lambda x: {'time': x[2].__str__(), 'type': x[4], 'amount': x[5], 'comment': x[6], 'category': x[9]}, ds))

        return ds

    def __create_category(self, name):
        category = sa.Table('category', self.md, autoload=True, autoload_with=self.engine)
        session = self.sessionMaker()
        res = session.execute(sa.insert(category).values(name=name))
        cat_id = res.inserted_primary_key[0]
        session.commit()
        return cat_id

    def __create_operation(self, op_type, amount, comment, category_id):
        operation = sa.Table('operation', self.md, autoload=True, autoload_with=self.engine)
        session = self.sessionMaker()
        res = session.execute(sa.insert(operation).values(type=op_type, amount=amount, comment=comment, category_id=category_id))
        op_id = res.inserted_primary_key[0]
        session.commit()
        return op_id

    def __create_user_operation(self, user_id, operation_id, time):
        user_operation = sa.Table('user_operation', self.md, autoload=True, autoload_with=self.engine)
        session = self.sessionMaker()
        session.execute(
            sa.insert(user_operation).values(user_id=user_id, operation_id=operation_id, time=time))
        session.commit()

    def clear_database_azazaza(self):
        for tbl in reversed(self.md.sorted_tables):
            self.engine.execute(tbl.delete())
