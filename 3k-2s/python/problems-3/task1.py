import sys
import re
import math


def countMeanAndDev(file):
    firsttime = True
    sum = 0
    sumSquares = 0
    n = 0
    for line in file:
        if line.startswith('open'):
            if firsttime:
                firsttime = False;
                continue
            n += 1
            time = re.search(r'\d+ usec', line).group(0)
            usecs = float(time.split(' ')[0])
            sum += usecs
            sumSquares += usecs ** 2
    mean = sum / n
    stdDevision = math.sqrt(sumSquares / n - mean ** 2)
    return mean, stdDevision


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Enter file path")
        filename = input()
    else:
        filename = sys.argv[1]
    with open(filename) as file:
        mean, stdDev = countMeanAndDev(file)
    print(mean)
    print(stdDev)
