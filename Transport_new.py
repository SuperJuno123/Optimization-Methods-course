from scipy.optimize import linprog
import numpy as np

n_customers = 3
m_factories = 2

c_const = np.arange(1, n_customers * m_factories + 1)
a_const = np.array([100, 50])
b_const = np.array([40, 30, 80])
я

# # Простой пример
# c_example = np.reshape(np.arange(9) + 1, (3, 3))
# a_example = np.array([30, 20, 90])
# b_example = np.array([10, 100, 30])
#
# # # Пример, как выглядят A_1 и A_2 для случая 3 x 3
# A_1 = [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1]]
# b_1 = a_example  # ограничения - запасыф
# A_2 = [[1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]]
# b_2 = b_example  # ограничения - заказы
#
# result = linprog(c_example.flatten(), A_1, a_example, A_2, b_example, method='simplex')
# print(c_example, a_example, b_example, result)


def A_a(n, m):
    res = np.zeros(shape=(m, n * m))
    for i in range(m):
        res[i][i * n:(i + 1) * n] = 1
    return res


def A_b(n, m):
    res = np.zeros(shape=(n, n * m))
    for i in range(n):
        res[i][list([i + n * j for j in range(m)])] = 1
    return res


# def A_b(n, m):
#     res = np.zeros(shape=(m, n * m))
#     for i in range(m):
#         res[i][list([i + m * j for j in range(n)])] = 1
#     return res

c = c_const
a = a_const
b = b_const

# # стоимость перевозок
c = np.random.rand(m_factories, n_customers) * 10
c_flat = c.flatten()
#
# # объём производства a_i в пункте производства i
a = np.random.rand(m_factories) * 200
#
# # объём потребления j в пункте потребления j
b = np.random.random(n_customers) * 200
b /= (np.sum(b) / np.sum(a))

A_1 = A_a(n_customers, m_factories)
b_1 = a  # ограничения - запасы
A_2 = A_b(n_customers, m_factories)
b_2 = b  # ограничения - заказы
print(f'Решаем следующую транспортную задачу:\n'
      f'Матрица стоимостей перевозок имеет вид:\n {np.array2string(c, precision=2, suppress_small=True)}\n'
      f'Минимизируем функцию {"".join([str(c_flat[i]) + " * x" + str(i) + " + " for i in range(len(c_flat))])[:-3:]}\n'
      f'Сгенерированы следующие ограничения по потребностям покупателей: b = {b}\n'
      f'Сгенерированы следующие ограничения по запасам производителей: a = {a}\n'
      f'Задача сбалансирована, что означает sum(a) = {np.sum(a)} = {np.sum(b)} = sum(b)')

A = np.vstack((A_1, A_2))
B = np.hstack((b_1, b_2))

result = linprog(c_flat, A_eq=A, b_eq=B, method='simplex')

print(f'Предложен следующий план перевозок: \n'
      f'{np.array2string(result.x.reshape(m_factories, n_customers), precision=2, suppress_small=True)}\n'
      f'Суммарная стоимость доставки: {c_flat.flatten() @ result.x}\n')

print(result)

from Simplex_method import *

evaluate_deltas(c_flat, A, B, result)
