# coding: utf8

import sys


def inputNumber():
    print("Введите число:")

    for line in sys.stdin:
        try:
            i = int(line)
            print(i)
            return
        except ValueError:
            print("Это не число")


if __name__ == '__main__':
    inputNumber()
