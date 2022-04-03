from GradientDescend import *
import visualisation

from constants import n
from core import derivative

initial_points = [
    np.ones(n) * 2,
    # np.random.random(n),
    # np.random.random(n) * 2 - 1,
    # np.zeros(n),
    # np.array([-1] + [1] * (n - 1), dtype=np.float32),
]

x_min_assert = np.ones(n)

N_iterations = [
    100,
    1000,
    10000
]


def GradDescendTest():
    for x_0 in initial_points:
        for n_iter in N_iterations:
            print(f'Точка начального приближения: {x_0}, максимальное количество итераций: {n_iter}')
            # x_min, x_all, f_all, grad_min = minimization(function, first_derivative, n_iter, x_0, early_stop=True)
            x_min, x_all, f_all, grad_min = minimization(function, None, n_iter, x_0, early_stop=True)

            print(f'Минимум: {x_min}, минимальное значение функции {function(x_min)}')
            print(f'Пройдено {np.shape(x_all)[0]} итераций (из {n_iter})')
            print(f'Градиент в точке минимума: {derivative(function, x_min)}, его норма: '
                  f'{np.linalg.norm(derivative(function, x_min))}\n')
            visualisation.steps_graphic(x_all, f_all, grad_min)


GradDescendTest()

# Градиент не нулевой (?),
#  Сделать тестики для градиента функций
