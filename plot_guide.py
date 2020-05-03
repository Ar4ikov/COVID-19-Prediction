# | Created by Ar4ikov
# | Время: 24.04.2020 - 12:57

import matplotlib.pyplot as plt
import numpy as np
from math import factorial as f


def generate_x(length, step):
    xa = [0]
    for i in range(length):
        xa.append(np.random.randint(xa[-1], xa[-1] + np.random.randint(1, step)))

    return xa[1:]


def generate_plot(max_dots):
    fig, ax = plt.subplots()
    ax.plot(generate_x(max_dots, 100), [np.random.randint(0, 100 - i) for i in range(max_dots)], label="Downgrade")
    ax.plot(generate_x(max_dots, 100), [np.random.randint(0 + i, 100) for i in range(max_dots)], label="Upgrade")
    ax.plot(generate_x(max_dots, 100), [np.random.randint(45, 55) for i in range(max_dots)], label="45 <= x <= 55")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Test data generate")
    ax.legend()


generate_plot(100)
plt.show()
