from database_api import DBDriver
import unittest
import pytest


class TestFinCalcBot(unittest.TestCase):
    DB = DBDriver()

    def test_insert_profit(self):
        self.DB.add_operation(1, 'Доход', 345, 'зарплата', '')
        history = self.DB.get_user_history(1)
        operation = history[0]
        self.assertTrue(345 == operation['amount'])
        self.assertTrue('Доход' == operation['type'])
        self.assertTrue('зарплата' == operation['comment'])
        self.assertTrue('' == operation['category'])
        self.DB.clear_database_azazaza()

    def test_insert_expense(self):
        self.DB.add_operation(2, 'Расход', 310, 'Сходил в кино', 'Развлечения')
        history = self.DB.get_user_history(2)
        operation = history[0]
        self.assertTrue(310 == operation['amount'])
        self.assertTrue('Расход' == operation['type'])
        self.assertTrue('Сходил в кино' == operation['comment'])
        self.assertTrue('Развлечения' == operation['category'])
        self.DB.clear_database_azazaza()

    def test_get_user_history(self):
        self.DB.add_operation(3, 'Доход', 1020, 'зарплата', '')
        self.DB.add_operation(3, 'Расход', 720, 'Сходил в театр', 'Развлечения')
        self.DB.add_operation(3, 'Доход', 890, 'Стипендия', '')
        history = self.DB.get_user_history(3)
        self.assertTrue(len(history) == 3)
        self.DB.clear_database_azazaza()


if __name__ == '__main__':
    unittest.main()
