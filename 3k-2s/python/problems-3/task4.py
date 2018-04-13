# coding: utf8
import types
import math


class Vector:
    """Вектор линейной алгебры. Реализует операции умножения на константу при записи a*V,
     где a - константа V - вектор; При записи U*V выполняет операцию скалярного произведения,
     в данном случае U,V - векторы"""

    myVector = []

    def _isInt(self, elem):
        try:
            if isinstance(elem, (float, complex)):
                return False
            int(elem)
            return True
        except ValueError:
            return False

    def _isFloat(self, elem):
        try:
            if isinstance(elem, complex):
                return False
            float(elem)
            return True
        except ValueError:
            return False

    def _isComplex(self, elem):
        try:
            complex(elem)
            return True
        except ValueError:
            return False

    def __getElemType(self, elements):
        elemsType = int
        for elem in elements:
            if self._isInt(elem):
                continue
            if self._isFloat(elem):
                elemsType = float
                continue
            if self._isComplex(elem):
                elemsType = float
                continue
            else:
                raise ValueError
        return elemsType

    def __init__(self, *args):
        """Создает vector из переданного iterable или чисел
        :param iterable
        """
        if len(args) == 1:
            if isinstance(*args, (tuple, list)):
                print("collection")
                t = self.__getElemType(*args)
                self.myVector = [t(e) for e in args[0]]
            if isinstance(*args, types.GeneratorType):
                print("generator")
                l = [e for e in args[0]]
                t = self.__getElemType(l)
                self.myVector = [t(e) for e in l]
            if isinstance(*args, int):
                self.myVector = [*args]
            if isinstance(*args, float):
                self.myVector = [*args]
            if isinstance(*args, complex):
                self.myVector = [*args]
        else:
            t = self.__getElemType(args)
            self.myVector = [t(e) for e in args]

    def __len__(self) -> int:
        """
        :return: Количество элементов вектора
        """
        return len(self.myVector)

    def __setitem__(self, key, value):
        """
        Устанавливает указанное значение в указанную координату
        :param key: Номер координаты в векторе >= 0
        :param value: Значение в этой координате (число)
        """
        self.myVector[key] = value

    def __add__(self, other: "Vector") -> "Vector":
        """Операция сложения векторов
        :param other: Объект этого же класса Vector
        :return: Сумма векторов линейной алгебры (объект класса Vector)
        """
        return Vector([values[0] + values[1] for values in zip(self.myVector, other.myVector)])

    def __sub__(self, other):
        """Операция вычитания векторов
        :param other: Объект класса Vector
        :return: Разность векторов
        """
        return Vector([values[0] - values[1] for values in zip(self.myVector, other.myVector)])

    def __rmul__(self, other):
        """Операция умножения на константу справа
        :param other: Числовая константа
        :return: Вектор, умноженный на константу
        """
        return Vector([val * other for val in self.myVector])

    def __mul__(self, other):
        """Операция скалярного произведения для векторов и умножения на константу слева
        :param other: Объект класса Vector
        :return: Результат скалярного произведения векторов
        """
        if isinstance(other, Vector):
            return sum([val[0] * val[1] for val in zip(self.myVector, other.myVector)])
        if isinstance(other, (int, float)):
            return Vector([val * other for val in self.myVector])

    def __eq__(self, other):
        """Проверка векторов на равенство
        :param other: Объект класса Vector
        :return: true если координаты векторов совпадают, false - иначе
        """
        return self.myVector == other.myVector

    def __getitem__(self, item):
        """Получение элемента по индексу
        :param item: Индекс(номер координаты вектора)
        :return: Число, находящееся в векторе по данному индексу
        """
        return self.myVector[item]

    def __str__(self):
        """Перевод в строку
        :return: Строковое представление вектора
        """
        return str(self.myVector)

    def get_length(self):
        """
        Вычисляет Евклидову длину вектора
        :return: Евклидова длина вектора
        """
        return math.sqrt(sum(v ** 2 for v in self.myVector))


v = Vector((1, 2.0, 3))
print(v)
