from scipy.optimize import linprog
import numpy as np

n_of_customers = 2
m_of_factories = 2


# # Простой пример
# c_example = np.reshape(np.arange(9) + 1, (3, 3))
# a_example = np.array([30, 20, 90])
# b_example = np.array([10, 100, 30])
#
# # # Пример, как выглядят A_1 и A_2 для случая 3 x 3
# A_1 = [[1, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1]]
# b_1 = a_example  # ограничения - запасы
# A_2 = [[1, 0, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 0, 1]]
# b_2 = b_example  # ограничения - заказы
#
# result = linprog(c_example.flatten(), A_1, a_example, A_2, b_example, method='simplex')
#
# print(c_example, a_example, b_example, result)


def create_A(m, n):
    res = np.zeros(shape=(m, n * m))
    for i in range(m):
        res[i][i * n:(i + 1) * n] = 1
    return res


# стоимость перевозок
c = np.random.rand(m_of_factories, n_of_customers) * 10
c_flat = c.flatten()

# объём производства a_i в пункте производства i
a = np.random.random(m_of_factories) * 200

# объём потребления j в пункте потребления j
b = np.random.random(n_of_customers) * 200
b /= (np.sum(b) / np.sum(a))
print(np.sum(a), np.sum(b))

A_1 = create_A(m_of_factories, n_of_customers)
b_1 = a  # ограничения - запасы
A_2 = create_A(m_of_factories, n_of_customers)
b_2 = b  # ограничения - заказы

print(f'Решаем следующую транспортную задачу:\n'
      f'Матрица стоимостей перевозок имеет вид:\n {np.array2string(c, precision=2, suppress_small=True)}\n'
      f'Минимизируем функцию '
      f'{"".join([np.array2string((c_flat[i]), precision=2, suppress_small=True) + "*x" + str(i) + " + " for i in range(len(c_flat))])[:-3:]}\n'
      f'Сгенерированы следующие ограничения по заказам покупателей: a = {a}\n'
      f'Сгенерированы следующие ограничения по запасам производителей: b = {b}\n'
      f'Задача сбалансирована, что означает sum(a) = sum(b) = {np.sum(a)}')

# result = linprog(c_flat, A_ub=A_1, b_ub=b_1, A_eq=A_2, b_eq=b_2, method='simplex')
result = linprog(c_flat, A_1, b_1, A_2, b_2)

print(
    f'Предложен следующий план перевозок: \n'
    f'{np.array2string(result.x.reshape(m_of_factories, n_of_customers), precision=2, suppress_small=True)}\n'
    f'Суммарная стоимость доставки: {c_flat @ result.x}\n')

print(result)


def simplex_method(c, A_ub, b_ub, A_eq, b_eq):
    pass
