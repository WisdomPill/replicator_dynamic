import matplotlib.pyplot as plt
import numpy as np

p0 = .75

x0 = (p0, 1 - p0)

payoff_matrix = np.array([
    [-1, 0],
    [0, -1]
])

alpha = 1
tau = .0001

a = payoff_matrix[0][0]
b = payoff_matrix[0][1]
c = payoff_matrix[1][0]
d = payoff_matrix[1][1]

epsilon = 1 / 10 ** 10

stop = False
max_iterations = 10 ** 6
i = 1

statuses = np.full(shape=[max_iterations], fill_value=-1, dtype=np.float64)
statuses[0] = p0

while not stop:
    p = statuses[i - 1]

    numerator = alpha + tau * (a * p)
    denominator = alpha + tau * (
            (a * (p ** 2)) + (b + c) * (1 - p) + d * ((1 - p) ** 2)
    )

    statuses[i] = p * numerator / denominator

    if i % 100 == 0:
        print(f"Iteration {i} has p -> {p}")

    abs_delta = abs(p - statuses[i])
    if abs_delta < epsilon:
        print(f"Stopping at iteration {i} since absolute delta is {abs_delta}, "
              f"which is less that epsilon {epsilon}")
        stop = True

    i += 1

    if i > max_iterations:
        stop = True

print(statuses[:i])

plt.plot(np.array(list(range(i))), statuses[:i])
plt.show()
