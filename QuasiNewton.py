import numpy as np
from constants import *
from core import *


def f2(x):
    return sum([np.power((np.sqrt(2) * x[i] - 1), 2)
                + np.power((np.sqrt(2) * x[i] - 1), 4) * np.sin(np.power((np.sqrt(2) * x[i] - 1), -2))
                for i in range(n)])


def grad_f2(x):
    return np.array([2 * np.sqrt(2) * (np.sqrt(2) * x[i] - 1) + 4 * np.sqrt(2) * (np.sqrt(2) * x[i] - 1) ** 3 * np.sin(
        1 / (np.sqrt(2) * x[i] - 1) ** 2) - 2 * np.sqrt(2) * (np.sqrt(2) * x[i]- 1) * np.cos(1 / (np.sqrt(2) * x[i] - 1) ** 2)
                     for i in range(n)])


def hess_f2(x):
    return np.diagflat(
        [24 * (np.sqrt(2) * x[i] - 1) ** 2 * np.sin(1 / (np.sqrt(2) * x[i] - 1) ** 2) - 32 * np.cos(
            1 / (np.sqrt(2) * x[i] - 1) ** 2) + ((np.sqrt(2) * x[i] - 1) ** 4) * (
                 (12 * np.cos(1 / (np.sqrt(2) * x[i] - 1) ** 2)) / ((np.sqrt(2) * x[i] - 1) ** 4) - (
                 8 * np.sin(1 / (np.sqrt(2) * x[i] - 1) ** 2)) / (np.sqrt(2) * x[i] - 1) ** 6) + 4
         for i in range(n)])


x_min_assert = np.array([np.sqrt(2) / 2] * n)

from constants import coefficient_for_Quasi_Newton as coefficient


def Quasi_Newton_method(func, x_0, grad=None, hess=None, N_iter=500, early_stop=True):
    if grad is None:
        grad = derivative
    if hess is None:
        hess = hessian
    x_all = []
    x_next = x_0
    for i in range(0, N_iter):
        x_prev = x_next
        x_all.append(x_prev)
        current_hessian = hess(x_prev)
        alpha_k = coefficient * np.linalg.norm(current_hessian)
        current_hessian = current_hessian + alpha_k * np.eye(n)

        x_next = x_prev - np.linalg.inv(current_hessian) @ grad(x_prev)
        if early_stop:
            if np.all(np.abs(x_prev - x_next) < accuracy_epsilon):
                x_next = x_prev
                break
    return x_next, x_all, list(map(func, x_all))
