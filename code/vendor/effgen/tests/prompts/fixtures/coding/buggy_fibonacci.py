# BSD 2-Clause License
# Buggy Fibonacci: off-by-one in range — memo[2] is never set for n==2.

def fibonacci(n):
    if n == 0:
        return 0
    memo = {0: 0, 1: 1}
    for i in range(2, n):
        memo[i] = memo[i-1] + memo[i-2]
    return memo[n]
