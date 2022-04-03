import numpy as np

class Function:
    def __init__(self, func, grad=None, hess=None):
        function = func
        gradient = grad
        hessian = hess


def format_2digit_array(array):
    return list(map(lambda x: f"{x:.2f}", array))


def derivative(func, x):
    from constants import derivative_epsilon
    dimension = 1 if type(x) is np.float64 else len(x)
    gradient = np.zeros(dimension)
    for i in range(dimension):
        cur_delta = np.zeros(dimension)
        cur_delta[i] += derivative_epsilon
        a1=func(x+cur_delta)
        a2=func(x)
        a3=a1-a2
        gradient[i] = (func(x+cur_delta) - func(x)) / derivative_epsilon

    return gradient

def hessian(func, x):
    from constants import hessian_epsilon, n
    hess = np.zeros((n, n))
    x=np.array(x, np.double)
    for i in range(n):
        for j in range(i+1, n):
            delta_i = np.zeros(n, dtype=np.double)
            delta_i[i] = hessian_epsilon
            delta_j = np.zeros(n, dtype=np.double)
            delta_j[j] = hessian_epsilon
            hess[i, j] = hess[j, i] = (func(x + (delta_i + delta_j)) - func(x + delta_i) - func(x + delta_j) + func(x)) \
                            / (hessian_epsilon ** 2)
    return hess

# def hessian1(func, x):
#     from constants import hessian_epsilon, n
#     hess = np.zeros((n, n))
#     hess2=np.zeros((n, n))
#     x=np.array(x, np.double)
#     n_iter=0
#     for i in range(n):
#         for j in range(i+1, n):
#             delta_i = np.zeros(n, dtype=np.double)
#             delta_i[i] = hessian_epsilon
#             delta_j = np.zeros(n, dtype=np.double)
#             delta_j[j] = hessian_epsilon
#             print(f'{(func(x + (delta_i + delta_j)))}\n'
#                   f'{func(x + delta_i)}\n'
#                   f'{func(x + delta_j)}\n'
#                   f'{func(x)}\n'
#                   f'x={x}\n'
#                   f'x+di+dj={(x + (delta_i + delta_j))}\n'
#                   f'{func(x + (delta_i + delta_j)) - func(x + delta_i) - func(x + delta_j) + func(x)}\n'
#                   f'e^2={hessian_epsilon ** 2}')
#             a1=func(x + (delta_i + delta_j))
#             a2=func(x + delta_i)
#             a3=func(x + delta_j)
#             a4=func(x)
#             delta_f1 = a1 - a2
#             delta_f2 = a3 - a4
#             eps_sqr = np.power(hessian_epsilon, 2)
#             res2=delta_f1 / eps_sqr - delta_f2 / eps_sqr
#             res=(func(x + (delta_i + delta_j)) - func(x + delta_i) - func(x + delta_j) + func(x)) \
#                             / (hessian_epsilon ** 2)
#             print(f'res={res}')
#             hess[i, j] = hess[j, i] = (func(x + (delta_i + delta_j)) - func(x + delta_i) - func(x + delta_j) + func(x)) \
#                             / (hessian_epsilon ** 2)
#             hess2[i, j] = hess[j, i] = res2
#             n_iter += 1
#
#     print(n_iter)
#     return hess

