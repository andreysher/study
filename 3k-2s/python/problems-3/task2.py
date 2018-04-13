import sys
import re
import math


def countOpens(f):
    n = 0
    for line in f:
        if line.startswith('open'):
            n += 1
    return n


def countDecil(n, file):
    listSize = math.ceil(0.1 * n)
    decil = []
    print(n)
    print(listSize)
    for line in file:
        if line.startswith('open'):
            time = float(re.search(r'\d+ usec', line).group(0).split(' ')[0])
            if len(decil) <= listSize:
                decil.append(time)
            else:
                min = sorted(decil)[0]
                if time > min:
                    decil.remove(min)
                    decil.append(time)
    print(sorted(decil)[0])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Enter file path")
        filename = input()
    else:
        filename = sys.argv[1]
    with open(filename) as file:
        n = countOpens(file)
    with open(filename) as file:
        countDecil(n, file)