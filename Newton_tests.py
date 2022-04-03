from Newton import *

# Простые (одномерные) функции

import visualisation

one_dimensional_functions = [f1, f3]
#
# for one_dim_func in one_dimensional_functions:
#     x_0 = 0.5
#
#     x_min, x_all, f_all = Newton_method_one_dimensional_wiki(one_dim_func, x_0)
#
#     print(
#         f'Точка минимума для функции {one_dim_func.__name__}: {x_min}, f(x_min) = {one_dim_func(x_min)}, '
#         f'grad(x_min) = {derivative(one_dim_func, x_min)}, '
#         f'N_iter = {len(x_all)}')
#
#     visualisation.visualize_Newton_one_dimensional_func(x_all, f_all, one_dim_func)
#

# Непростая функция (с пары)

from constants import n

number_of_points = 100

import sys

f_min_all = sys.maxsize
x_min_all = sys.maxsize

np.random.seed(100)


for i in range(number_of_points):
    x_min, x_all, f_all = Newton_method(f2, np.random.random(n), grad=grad_f2, hess=hess_f2)

    if f2(x_min) < f_min_all:
        f_min_all = f2(x_min)
        x_min_all = x_min

    print(f'Точка №{i + 1}:\nf(x_min) = {f2(x_min)}, |grad(x_min)| = {np.linalg.norm(grad_f2(x_min))}'
          f'\nN_iter = {len(x_all)}')

    # print(f'Точка минимума функции {f2.__name__}: {x_min}, \nf(x_min) = {f2(x_min)}, \ngrad(x_min) = {grad_f2(x_min)}, '
    #       f'|grad(x_min)| = {np.linalg.norm(grad_f2(x_min))}'
    #       f'\nN_iter = {len(x_all)}, \nОжидаемая точка минимума: {x_min_assert}')
    #
    # visualisation.steps_graphic(x_all, f_all, grad_min=derivative(f2, x_min))

print(
    f'Точка минимума функции {f2.__name__}: {x_min_all}, \nf(x_min) = {f2(x_min_all)}, \ngrad(x_min) = {grad_f2(x_min_all)}, '
    f'|grad(x_min)| = {np.linalg.norm(grad_f2(x_min_all))}\nОжидаемая точка минимума: {x_min_assert}, '
    f'разница: {np.linalg.norm(np.all(np.abs(x_min_assert - x_min_all)))}')
