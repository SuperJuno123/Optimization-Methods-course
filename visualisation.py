import matplotlib.pyplot as plt
import numpy as np
from core import *


def steps_graphic(x_all, f_all, grad_min):
    plt.plot(list(map(np.linalg.norm, x_all)), f_all, marker='o', markersize=3)
    plt.xlabel('Нормы векторов x_k')
    plt.ylabel('Значения функции')
    x_end = np.linalg.norm(x_all[-1])
    y_end = f_all[-1]
    plt.plot(x_end, y_end, marker='o', markersize=3, color='red')
    plt.title(f'N: {len(x_all)}, x_0 = {format_2digit_array(x_all[0])}')
    plt.annotate(f'|∇f(x_min)|={np.linalg.norm(grad_min):.5e}', (x_end, y_end), horizontalalignment='center')
    plt.show()


def visualize_Newton_one_dimensional_func(x_all, f_all, func):
    """Только для одномерной функции!"""
    range = np.abs(np.min(x_all) - np.max(x_all))

    start = np.min(x_all) - range / 2
    stop = np.max(x_all) + range / 2

    x_for_graphic = np.linspace(start, stop, num=500)
    y_for_graphic = list(map(func, x_for_graphic))
    plt.plot(x_for_graphic, y_for_graphic)
    # plt.plot([x_all[i] if i % 2 else x_all[i] for i in range(2*len(x_all))],
    #          [f_all[i] if i % 2 else 0 for i in range(2*len(x_all))])
    new_list_of_x = []
    new_list_of_y = []
    for item_x, item_y in zip(x_all, f_all):
        new_list_of_x.append(item_x)
        new_list_of_x.append(item_x)
        new_list_of_y.append(0)
        new_list_of_y.append(item_y)
    plt.plot(new_list_of_x, new_list_of_y)
    plt.title(f'График последовательных приближений для функции {func.__name__}')
    plt.grid()
    plt.show()
