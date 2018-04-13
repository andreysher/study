import math


def to_simple_factors(number) -> list:
    factors = []
    current = 2
    power = 0
    if number < 2:
        return factors
    maximo = math.trunc(math.sqrt(number)) + 1
    while current < maximo:
        while number % current == 0:
            power += 1
            number = number // current
        if power > 0:
            factors.append([current, power])
        power = 0
        current += 1
    if number != 1:
        factors.append([number, 1])
    return factors


if __name__ == '__main__':
    fact = to_simple_factors(12345678874658964)
    print(fact)
