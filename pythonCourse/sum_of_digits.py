import sys

s = 0
for digit in sys.argv[1]:
    s += int(digit)

print(s)
