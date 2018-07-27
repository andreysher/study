import sys
import math

a = int(sys.argv[1])
b = int(sys.argv[2])
c = int(sys.argv[3])

d = math.sqrt(math.pow(b, 2) - 4 * a * c)

print((-b+d)/2*a)
print((-b-d)/2*a)