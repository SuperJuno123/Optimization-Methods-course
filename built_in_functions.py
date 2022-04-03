from scipy.optimize import minimize

from GradientDescend_tests import initial_points

from GradientDescend import function as f1

from Newton import f2 as f2

methods_of_built_in_function = ['Nelder-Mead',
                                'Powell',
                                'CG',
                                'BFGS',
                                'Newton-CG',
                                'L-BFGS-B',
                                'TNC',
                                'COBYLA',
                                'SLSQP',
                                'trust-constr',
                                'dogleg',
                                'trust-ncg',
                                'trust-exact',
                                'trust-krylov']

for x_0 in initial_points:
    print(f'Точка начального приближения: {x_0}')
    for method in methods_of_built_in_function:
        try:
            result = minimize(f2, x_0, method=method)
        except Exception:
            print(f'Метод {method} вызвал ошибку')
            continue
        if result['success'] is True:
            print(f'Метод {method} сошёлся и предлагает следующее решение:'
                  f'\n{list(map("{:.2f}".format, result["x"]))}')
        else:
            print(f'Метод {method} не сошёлся')