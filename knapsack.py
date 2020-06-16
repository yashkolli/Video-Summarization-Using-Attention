import numpy as np

def knapsack(v, w, W):
    r = len(v) + 1
    c = W + 1

    v = np.r_[[0], v]
    w = np.r_[[0], w]

    dp = [[0 for i in range(c)] for j in range(r)]

    for i in range(1,r):
        for j in range(1,c):
            if w[i] <= j:
                dp[i][j] = max(v[i] + dp[i-1][j-w[i]], dp[i-1][j])
            else:
                dp[i][j] = dp[i-1][j]

    chosen = []
    i = r - 1
    j = c - 1
    while i > 0 and j > 0:
        if dp[i][j] != dp[i-1][j]:
            chosen.append(i-1)
            j = j - w[i]
            i = i - 1
        else:
            i = i - 1

    return dp[r-1][c-1], chosen

if __name__ == '__main__':
    values = list(map(int, input().split()))
    weights = list(map(int, input().split()))
    max_weight = int(input())

    max_value, chosen = knapsack(values, weights, max_weight)

    print("The max value possible is")
    print(max_value)

    print("The index chosen for these are")
    print(' '.join(str(x) for x in chosen))
