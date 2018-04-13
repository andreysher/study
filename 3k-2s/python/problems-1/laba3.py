import time
import math
from bitarray import bitarray


def eratosthen_list(number) -> list:
    l = list(range(number + 1))
    l[1] = 0
    max = math.ceil(math.sqrt(number))
    for candidate in range(2, max):
        for cond in range(2 * candidate, number + 1, candidate):
            l[cond] = 0
    return list(filter(lambda x: x != 0, l))


def eratosthen_set(number) -> set:
    s = set(range(2, number + 1))
    max = math.ceil(math.sqrt(number))
    for candidate in range(2, max):
        print(candidate)
        if candidate in s:
            for cur in range(candidate * 2, number + 1, candidate):
                s.discard(cur)
    return s


def eratosthen_bitarray(number):
    arr = bitarray(number + 1)
    arr.setall(True)
    arr[0] = False
    arr[1] = False
    for i in range(math.ceil(math.sqrt(number))):
        if not arr[i]:
            continue
        for j in range(2 * i, number + 1, i):
            arr[j] = False
    return arr


if __name__ == '__main__':
    print("Введите число:")
    number = int(input())
    l_start = time.time()
    l = eratosthen_list(number)
    l_time = time.time() - l_start
    print(l)
    print(l_time)
    s_start = time.time()
    s = eratosthen_set(number)
    s_time = time.time() - s_start
    print(s)
    print(s_time)
    a_start = time.time()
    a = eratosthen_bitarray(number)
    a_time = time.time() - a_start
    print(a)
    print(a_time)
