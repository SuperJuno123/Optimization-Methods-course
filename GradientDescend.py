import numpy as np

from constants import *

from core import derivative


def function(x):
    return 1 / 4 * (x[0] - 1) ** 2 + sum([(x[i + 1] - 2 * x[i] ** 2 + 1) ** 2 for i in range(0, n - 2)])


def first_derivative(x):
    return 1 / 2 * (x[0] - 1) + sum([2 * (x[i + 1] - 2 * x[i] ** 2 + 1) * (1 - 4 * x[i]) for i in range(0, n - 2)])


def minimization(func, grad, N_iter, x_0, early_stop=True):
    if grad is None:
        grad = derivative

    x_all = []
    x_next = x_0
    for i in range(0, N_iter):
        x_prev = x_next
        x_all.append(x_prev)
        x_next = x_prev - learning_rate * grad(func, x_prev)
        if early_stop:
            if np.all(np.abs(x_prev - x_next) < accuracy_epsilon):
                x_next = x_prev
                break
    return x_next, x_all, list(map(func, x_all)), grad(func, x_next)
