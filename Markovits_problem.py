import numpy as np
from scipy.optimize import minimize
from constants import gamma_for_Markovitz as gamma

n = 500

r = np.random.rand(n)

A = np.random.rand(n, n)

sigma = np.eye(n) + gamma * A @ A.T

x0 = np.ones(n)

print(f'r = {np.array2string(r, suppress_small=True, precision=2)},\n'
      f'sigma = \n{np.array2string(sigma, suppress_small=True, precision=2)}\n')


def func(x):
    return (x.T @ sigma @ x) / (r @ x) ** 2


methods = ['BFGS',
           'CG']
results = []

for method in methods:
    print(method)
    result = minimize(func, x0, method=method)
    print(result)
    results.append(result.fun)
    print(f'\nx_min = {np.array2string(result.x, suppress_small=True, precision=2)}, '
          f'f(x_min) = {result.fun}\n')


def cons(x):
    return sum(x) - 1


print('SLSQP with constrains')
constraints1 = {'type': 'eq',
                'fun': cons}
reslut_SLSQP_with_constraints = minimize(func, x0, method='SLSQP', constraints=constraints1)
print(reslut_SLSQP_with_constraints)
methods.append('SLSQP with constrains')
results.append(reslut_SLSQP_with_constraints.fun)

for method, f_min in zip(methods, results):
    print(f'Метод: {method}, f_min = {f_min}')
