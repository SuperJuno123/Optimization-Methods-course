import numpy as np
from constants import *
from core import *


def f1(x):
    """Первоначальная с пары"""
    return x ** 2 + x ** 4 * np.sin(x ** -2)


def f2(x):
    """Обновленная (последняя) с пары"""
    return sum([np.power((np.sqrt(2) * x[i] - 1), 2)
                + np.power((np.sqrt(2) * x[i] - 1), 4) * np.sin(np.power((np.sqrt(2) * x[i] - 1), -2))
                for i in range(n)])


def grad_f2(x):
    return np.array([2 * np.sqrt(2) * (np.sqrt(2) * x[i] - 1) * (
            2 * np.power((np.sqrt(2) * x[i] - 1), 2) * np.sin(np.power((np.sqrt(2) * x[i] - 1), -2)) - np.cos(
        (np.power((np.sqrt(2) * x[i] - 1), -2))) + 1) for i in range(n)])


def hess_f2(x):
    return np.diagflat(
        [24 * np.power((np.sqrt(2) * x[i] - 1), 2) * np.sin((np.power((np.sqrt(2) * x[i] - 1), -2))) - 8 * np.sin(
            (np.power((np.sqrt(2) * x[i] - 1), -2))) / np.power((np.sqrt(2) * x[i] - 1), 2) - 20 * np.cos(
            (np.power((np.sqrt(2) * x[i] - 1), -2))) + 4
         for i in range(n)])


x_min_assert = np.array([np.sqrt(2) / 2] * n)

func_2 = Function(func=f2,
                  grad=grad_f2,
                  hess=hess_f2)


def f3(x):
    """Простая одномерная функция с википедии"""
    return np.cos(x) - x ** 3


def Newton_method_one_dimensional_wiki(func, x_0, grad=None, N_iter=500, early_stop=True):
    if grad is None:
        grad = lambda x: (func(x + derivative_epsilon) - func(x)) / derivative_epsilon
    x_all = []
    x_next = x_0
    for i in range(0, N_iter):
        x_prev = x_next
        x_all.append(x_prev)
        x_next = x_prev - func(x_prev) / grad(x_prev)
        if early_stop:
            if np.all(np.abs(x_prev - x_next) < accuracy_epsilon):
                x_next = x_prev
                break
    return x_next, x_all, list(map(func, x_all))


def Newton_method(func, x_0, grad=None, hess=None, N_iter=500, early_stop=True):
    if grad is None:
        grad = derivative
    if hess is None:
        hess = hessian
    x_all = []
    x_next = x_0
    for i in range(0, N_iter):
        x_prev = x_next
        x_all.append(x_prev)
        x_next = x_prev - np.linalg.inv(hess(x_prev)) @ grad(x_prev)
        if early_stop:
            if np.all(np.abs(x_prev - x_next) < accuracy_epsilon):
                x_next = x_prev
                break
    return x_next, x_all, list(map(func, x_all))
