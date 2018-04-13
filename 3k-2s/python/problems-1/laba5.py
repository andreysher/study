import math


def is_prime(number) -> bool:
    i = 2
    maximum = math.trunc(math.sqrt(number)) + 1
    while i < maximum:
        if number % i == 0:
            return False
        i += 1
    return True


if __name__ == '__main__':
    print("Введите число:")
    n = int(input())
    s = [x for x in range(2, n) if is_prime(x)]
    print(s)
