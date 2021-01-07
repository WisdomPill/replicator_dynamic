import matplotlib.pyplot as plt
import numpy as np

r = 1 / 4
s = 1 / 4
t = 1 - r - s

x0 = (r, s, t)

payoff_matrix = np.array([
    [1, 3, 0],
    [0, 1, 3],
    [3, 0, 1]
])

alpha = 1
tau = 1

a = payoff_matrix[0][0]
b = payoff_matrix[0][1]
c = payoff_matrix[0][2]
d = payoff_matrix[1][0]
e = payoff_matrix[1][1]
f = payoff_matrix[1][2]
g = payoff_matrix[2][0]
h = payoff_matrix[2][1]
i = payoff_matrix[2][2]

epsilon = 1 / 10 ** 5

stop = False
max_iterations = 10**5
j = 1

statuses = np.full(shape=[max_iterations, 3], fill_value=-1, dtype=np.float64)
statuses[0] = (r, s, t)

transformations = np.full(shape=[max_iterations, 2], fill_value=-1, dtype=np.float64)
transformations[0] = (-r / 2 + s / 2, -np.sqrt(3) / 2 * (r + s - 1))

while not stop:
    r = statuses[j - 1][0]
    s = statuses[j - 1][1]
    t = statuses[j - 1][2]

    r_numerator = alpha + tau * (a * r)
    s_numerator = alpha + tau * (e * s)
    t_numerator = alpha + tau * (i * t)

    denominator = alpha + tau * (
            (r ** 2) * a + r * (b * s + c * t) + s * (d * r + f * t) + e
            * (s ** 2) + t * (g * r + h * s) + i * (t ** 2)
    )

    statuses[j][0] = r * r_numerator / denominator
    statuses[j][1] = s * s_numerator / denominator
    statuses[j][2] = t * t_numerator / denominator

    transformations[j][0] = -r / 2 + s / 2
    transformations[j][1] = -np.sqrt(3) / 2 * (r + s - 1)

    print(np.sum(statuses[j]))

    # if j % 10 == 0:
    print(f"Iteration {j} has r,s,t -> {statuses[j]} "
              f"and transformation -> {transformations[j]}")

    distance = np.linalg.norm(statuses[j - 1] - statuses[j])
    if distance < epsilon:
        print(f"Stopping at iteration {j} since absolute delta is {distance}, "
              f"which is less that epsilon {epsilon}")
        stop = True

    j += 1

    if j == max_iterations:
        stop = True

print(statuses[j])
print(transformations[:j])

print((transformations[::j][0], transformations[::j][1]))

plt.scatter(transformations[:j,0], transformations[:j,1])
plt.show()
