from scipy.stats import qmc
import sys
from Newton import *

sampler = qmc.Sobol(d=n, scramble=False)
sobol_sequence_array = sampler.random_base2(m=7)

f_min_best = sys.maxsize
x_min_best = sys.maxsize

for sobol_point in sobol_sequence_array:
    x_min, x_all, f_all = Newton_method(f2, sobol_point, grad=grad_f2, hess=hess_f2)

    if f2(x_min) < f_min_best:
        f_min_best = f2(x_min)
        x_min_best = x_min

print(f'С использованием последовательностей Соболя (взято {sobol_sequence_array.shape[0]} точек Соболя):\n'
      f'Точка минимума функции {f2.__name__}: {x_min_best}, \nf(x_min) = {f2(x_min_best)}, \ngrad(x_min) = {grad_f2(x_min_best)}, '
      f'|grad(x_min)| = {np.linalg.norm(grad_f2(x_min_best))}\nОжидаемая точка минимума: {x_min_assert}')


print(f'Постараемся "побить" рекорд последовательностей Соболя, равный f(x_min) = {f2(x_min_best)}:\n')

N_of_Newton_try = 200000

np.random.seed(100)

random_points = np.random.rand(N_of_Newton_try, n)

f_min_Newton = sys.maxsize
x_min_Newton = sys.maxsize

from time import time
start = time()

for i in range(N_of_Newton_try):
    x_min, x_all, f_all = Newton_method(f2, random_points[i], grad=grad_f2, hess=hess_f2)

    if f2(x_min) < f_min_Newton:
        f_min_Newton = f2(x_min)
        x_min_Newton = x_min

    if i % 1000 == 0:
        print(f'Прошло {i} итераций, {time() - start:.2f} секунд, результата пока нет. Текущий рекорд для Ньютона: f(x_min)={f_min_Newton} (у Соболя было {f_min_best})')

    if f2(x_min) < f_min_best:
        print(f'Ура! Ньютон побил Соболя, i = {i}, x_min = {x_min}, f(x_min) = {f2(x_min)},'
              f' |grad(x_min)| = {np.linalg.norm(grad_f2(x_min))}')
        break