import numpy as np


def evaluate_deltas(c_new, A, B, result):
    """ВНИМАНИЕ! Работает только для "==" ограничений, введение slack-переменных не предусмотрено"""
    n_of_variable = A.shape[1]
    m_of_restrictions = A.shape[0]

    # получение номеров базисных векторов по решению задачи (ненулевые компоненты) - основные вектора
    # result.x = np.round(result.x, decimals=7)
    basis_indices = np.nonzero(result.x)[0]

    # формирование базисной матрицы и столбца базисных коэффициентов целевой функции
    basis = np.zeros((m_of_restrictions, m_of_restrictions))
    c_bas = []
    print(A[:, 4])

    for basis_i in basis_indices:
        # если вектор основной
        basis[basis_i] = A[:, basis_i] # добавление всего столбца, относящегося к базисным
        c_bas.append(c_new[basis_i])


    # вектора добавляются в матрицу как вектора-строки. Нужно - вектора-столбцы
    # проведение транспонирования
    # basis = np.reshape(basis, (m_of_restrictions, m_of_restrictions)).T

    basis = np.array(basis).T

    # получение обратной матрицы
    basis_inverse = np.linalg.inv(basis)

    while True:
        # получение вектора cb*B(-1) для дальнейшего получения оценок
        # cb - коэффициенты целевой функции, соответствующие базисным переменным
        # cb * basis (для каждой компоненты) - c_new = 0   =>   cb * basis = c_new
        cb = np.dot(c_bas, basis_inverse)
        # получение оценок основных векторов по формуле: скалярное произведение cb (цэ базисных) на A (матрицу ограничений)
        # минус соответствующий коэффициент целевой функции
        delta_main = np.dot(cb, A) - c_new
        delta_main = np.round(delta_main, 2)
        # получение коэффициентов разложения вектора p_0 - столбец свободных членов в симпекс-таблице (столбец A0 у Кашубы)
        p_0 = np.dot(basis_inverse, B)

        # получение минимальной оценки
        min_delta_main = np.min(delta_main)

        print(min_delta_main)
    #
    #     # если максимальная оценка равна 0, получен оптимум
    #     if min(min_delta_main, min_delta_slack) >= 0:
    #         # print("optimum")
    #         break
    #     else:
    #         # если отрицательная оценка у основного вектора
    #         if min_delta_main < 0:
    #             # запоминаем индекс основного вектора
    #             ind_to_basis = np.argmin(delta_main)
    #         else:
    #             # запоминаем индекс дополнительного вектора - корректируем номера векторов
    #             ind_to_basis = np.argmin(delta_slack) + n_of_variable
    #
    #     # вычисляем коэффициенты разложения по базису вектора с положительной оценкой
    #     if ind_to_basis < n_of_variable:
    #         p_j = np.dot(basis_inverse, A[:, ind_to_basis])
    #         c_new_basis = c_new[ind_to_basis]
    #     else:
    #         p_j = np.dot(basis_inverse, A_slack[:, ind_to_basis - n_of_variable])
    #         c_new_basis = 0
    #
    #     # находим вектор, который выводится из базиса (среди столбца, который показал положительную оценку, ищу минимальное
    #     # отношение ai0 / aij (столб. св. членов делить на соответствующую компоненту рассматриваемого вектора)
    #     ind = -1
    #     minimum = 100000
    #     for basis_i in range(m_of_restrictions):
    #         if p_j[basis_i] > 0:
    #             if p_0[basis_i] / p_j[basis_i] < minimum:
    #                 ind = basis_i
    #                 minimum = p_0[basis_i] / p_j[basis_i]
    #
    #     # осуществляем пересчет по формулам Гаусса (делим на aij (на котором "остановились", то есть которое принадлежит
    #     # СТОЛБЦУ с положит. оценкой и СТРОКЕ с минимумом отношения), затем вычитаем из таблицы строки таким образом, чтобы
    #     # столбец вектора, только что вошедшего в базис, имел единичный вид
    #     # замена номера базисного вектора
    #     basis_indices[ind] = ind_to_basis
    #     # замена коэффициента целевой функции при базисном векторе
    #     c_bas[ind] = c_new_basis
    #     # пересчет
    #     basis_inverse[ind, :] = basis_inverse[ind, :] / p_j[ind]  # делю, чтобы в новом базисном
    #     for basis_i in range(m_of_restrictions):
    #         if basis_i != ind:
    #             basis_inverse[basis_i, :] = basis_inverse[basis_i, :] - basis_inverse[ind, :] * p_j[basis_i]  # см. онлайн-решалку
    #
    # # печать коэффициентов разложения p_0, из которого можно получить ответ
    # # print(p_0)
    # # print(basis_indices)
    #
    # new_plan = np.zeros(n_of_variable)
    #
    # for basis_i in range(basis_indices.size):
    #     if basis_indices[basis_i] < n_of_variable:  # slack-переменные в оптимальный план не входят
    #         new_plan[basis_indices[basis_i]] = p_0[basis_i]

    # return new_plan

