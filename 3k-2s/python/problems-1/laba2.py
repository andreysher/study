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
    simple = is_prime(1488866241517)
    print(simple)
