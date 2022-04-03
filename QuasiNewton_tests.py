from scipy.stats import qmc
import sys
from Newton import *
from QuasiNewton import *

n_of_Quasi_Newton_try = 10

np.random.seed(100)

random_points = np.random.rand(n_of_Quasi_Newton_try, n)

from time import time

start = time()

for point in random_points:
    x_min_N, x_all_N, f_all_N = Newton_method(f2, point, grad=grad_f2, hess=hess_f2)
    x_min_QN, x_all_QN, f_all_QN = Quasi_Newton_method(f2, point, grad=grad_f2, hess=hess_f2)

    x_min_N_for_print = np.array2string(x_min_N, precision=2, suppress_small=True)
    x_min_QN_for_print = np.array2string(x_min_QN, precision=2, suppress_small=True)

    print(f'Рассмотрим точку {np.array2string(point, precision=2, suppress_small=True)}\n'
          f'НЬЮТОН:\n'
          f'x_min = {x_min_N_for_print}\n'
          f'f(x_min) = {f2(x_min_QN)}\n'
          f'Результат был достигнут за {len(x_all_N)} итераций\n\n'
          f'ГОВАРД & МАКГВАРДЛТ:\n'
          f'x_min = {x_min_QN_for_print}\n'
          f'f(x_min) = {f2(x_min_N)}\n'
          f'Результат был достигнут за {len(x_all_QN)} итераций\n{"-"*50}\n')
