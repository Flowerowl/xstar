import math

try:
    import itertools
    combinations = itertools.combinations
except AttributeError:
    def combinations(seq, r=None):
        if r == None:
            r = len(seq)
        if r <= 0:
            yield []
        else:
            for i in xrange(len(seq)):
                for cc in combinations(seq[i + 1:], r - 1):
                    yield [seq[i]] + cc

try:
    factorial = math.factorial
except AttributeError:
    def factorial(x):
        n = abs(int(x))
        if n < 1:
            n = 1
        x = 1
        for i in range(1, n + 1):
            x = i * x
        return x
