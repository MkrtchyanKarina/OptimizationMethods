import numpy as np


def input_problem():
    """ Функция для ввода данных задачи """
    task_type = input("Тип задачи (max/min): ").lower()
    function = list(map(float, input("Коэффициенты целевой функции через пробел: ").split()))
    inequalities_count = int(input("Количество ограничений: "))

    inequalities_coefs = []  # коэффициенты при x в неравенствах (A)
    ineq_constant_coefs = []  # свободные коэффициенты в правой части неравенств (b)
    ineq_signs = []  # знак неравенства: =, <=, >=

    for i in range(inequalities_count):
        row = list(map(float, input(f"Коэффициенты {i + 1}-го ограничения: ").split()))
        sign = input(f"Знак ограничения {i + 1} (<=, >=, =): ")
        value = float(input(f"Правая часть {i + 1}-го ограничения: "))

        inequalities_coefs.append(row)
        ineq_signs.append(sign)
        ineq_constant_coefs.append(value)

    return task_type, function, inequalities_coefs, ineq_constant_coefs, ineq_signs


def make_canonical(task_type, function, inequalities_coefs, ineq_constant_coefs, ineq_signs):
    """ Функция для приведения задачи к каноническому виду """

    inequalities_coefs = np.array(inequalities_coefs, dtype=float)
    ineq_constant_coefs = np.array(ineq_constant_coefs, dtype=float)
    function = np.array(function, dtype=float)

    if task_type == "min":
        function = -function  # задачу минимизации преобразуем в максимизацию

    variable_names = [f"x{i + 1}" for i in range(len(function))]  # список используемых переменных
    add_var_number = len(function) + 1  # номер следующей добавочной переменной

    for i, sign in enumerate(ineq_signs):

        if sign == "<=":
            # для <= добавляется дополнительная переменная +s
            new_column = np.zeros(len(ineq_constant_coefs))  # создаем столбец из нулей
            new_column[i] = 1  # в нужной строке (неравенстве) ставим коэффициент 1

            inequalities_coefs = np.column_stack((inequalities_coefs, new_column))  # присоединяем новый столбец к остальным коэффициентам неравенств
            variable_names.append(f"x{add_var_number}")  # добавляем новую переменную в список

            add_var_number += 1  # увеличиваем порядковый номер добавочного коэффициента

        elif sign == ">=":
            # для >= добавляется избыточная переменная -s
            new_column = np.zeros(len(ineq_constant_coefs))  # создаем столбец из нулей
            new_column[i] = -1  # в нужной строке (неравенстве) ставим коэффициент -1

            inequalities_coefs = np.column_stack((inequalities_coefs, new_column))  # присоединяем новый столбец к остальным коэффициентам неравенств
            variable_names.append(f"x{add_var_number}")  # добавляем новую переменную в список

            add_var_number += 1  # увеличиваем порядковый номер добавочного коэффициента

        elif sign == "=":
            pass  # для равенства дополнительная переменная не добавляется

        else:
            raise ValueError("Знак должен быть <=, >= или =")


    return inequalities_coefs, ineq_constant_coefs, function, variable_names


def find_basis(inequalities_coefs):
    """ Функция поиска базисных переменных среди существующих столбцов """

    rows_count, columns_count = inequalities_coefs.shape  # получение числа строк и столбцов матрицы коэффициентов
    basis = [-1] * rows_count  # создаем список для базисных переменных

    for j in range(columns_count):
        column = inequalities_coefs[:, j]  # берем j-ый столбец матрицы

        for i in range(rows_count):
            if basis[i] == -1:  # если мы еще не нашли базисную переменную для данной строки
                unit_column = np.zeros(rows_count)
                unit_column[i] = 1  # создаем столбец с 1 только на месте данной переменной, остальные 0

                if np.array_equal(column, unit_column):  # проверяем равенство взятого столбца с базисным
                    basis[i] = j  # записываем номер базисной переменной, если они равны
                    break

    return basis


def add_artificial_variables(inequalities_coefs, basis, variable_names):
    """ Функция добавления искусственных переменных там, где нет базиса """

    artificial = []

    for i in range(len(basis)):

        if basis[i] == -1:
            # Если для строки нет базисной переменной, добавляем искусственную
            new_column = np.zeros(inequalities_coefs.shape[0])
            new_column[i] = 1  # создаем столбец с 1 только на месте данной переменной, остальные 0

            inequalities_coefs = np.column_stack((inequalities_coefs, new_column))  # присоединяем этот столбец

            basis[i] = inequalities_coefs.shape[1] - 1  # записываем индекс искусственной переменной
            artificial.append(inequalities_coefs.shape[1] - 1)  # сохраняем индекс искусственной переменной

            variable_names.append(f"a{i + 1}")

    return inequalities_coefs, basis, artificial, variable_names


def create_tableau(inequalities_coefs, ineq_constant_coefs, function, basis):
    """ Функция формирования симплекс-таблицы """

    # объединяем A и b, затем добавляем последнюю строку с коэффициентами целевой функции
    tableau = np.vstack((np.column_stack((inequalities_coefs, ineq_constant_coefs)), np.append(-function, 0)))

    for i, basic_variable in enumerate(basis):
        coefficient = tableau[-1, basic_variable]  # для каждой базисной переменной берем ее коэффициент из исходной функции
        tableau[-1] -= coefficient * tableau[i]  # выражаем базисные переменные из ограничений и корректируем целевую функцию

    return tableau


def simplex(tableau, basis, variable_names):
    """ Функция решения задачи симплекс-методом """

    rows_count = len(basis)

    while True:
        print("\nТекущая симплекс-таблица:")
        print(np.round(tableau, 3))
        print("Базис:", [variable_names[i] for i in basis])

        function_row = tableau[-1, :-1]  # коэффициенты целевой функции без свободного члена

        candidates = []

        # Ищем отрицательные коэффициенты целевой функции,
        # для которых в столбце есть положительный коэффициент
        for j in range(tableau.shape[1] - 1):

            if function_row[j] >= -1e-10:
                continue

            for i in range(rows_count):
                if tableau[i, j] > 1e-10:
                    candidates.append(j)
                    break

        if len(candidates) == 0:
            break  # если подходящих столбцов нет, достигнут оптимум

        entering = min(candidates, key=lambda j: function_row[j])  # выбираем самый отрицательный коэффициент

        print("Входит:", variable_names[entering])

        ratios = []

        # Определяем выходящую переменную по минимальному положительному отношению b_i / a_ik
        for i in range(rows_count):
            coefficient = tableau[i, entering]

            if coefficient > 1e-10:
                ratios.append(tableau[i, -1] / coefficient)
            else:
                ratios.append(np.inf)  # строка не участвует в выборе, если коэффициент неположительный

        leaving_row = np.argmin(ratios)  # выбираем минимальное отношение
        leaving = basis[leaving_row]

        print("Отношения:", ratios)
        print("Выходит:", variable_names[leaving])

        pivot = tableau[leaving_row, entering]  # разрешающий элемент — пересечение разрешающей строки и столбца
        print("Разрешающий элемент:", pivot)

        old_tableau = tableau.copy()  # сохраняем старую таблицу, чтобы все расчеты выполнить по исходным значениям

        # Делим разрешающую строку на разрешающий элемент
        tableau[leaving_row] = old_tableau[leaving_row] / pivot

        # Обнуляем разрешающий столбец во всех остальных строках
        for i in range(len(tableau)):
            if i != leaving_row:
                coefficient = old_tableau[i, entering]
                tableau[i] = old_tableau[i] - coefficient * tableau[leaving_row]

        basis[leaving_row] = entering  # заменяем вышедшую переменную на вошедшую в базис

    return tableau, basis


def phase_one(inequalities_coefs, ineq_constant_coefs, basis, artificial, variable_names):
    """ Функция решения вспомогательной задачи """

    print("\nФаза I — вспомогательная задача\n")

    function_auxiliary = np.zeros(inequalities_coefs.shape[1])

    for variable in artificial:
        function_auxiliary[variable] = -1  # вспомогательная функция минимизирует сумму искусственных переменных

    tableau = create_tableau(inequalities_coefs, ineq_constant_coefs, function_auxiliary, basis)

    tableau, basis = simplex(tableau, basis, variable_names)

    auxiliary_value = tableau[-1, -1]

    print("\nЗначение вспомогательной функции:", auxiliary_value)

    if abs(auxiliary_value) > 1e-10:
        raise ValueError("Исходная задача не имеет допустимого решения.")

    return tableau, basis


def remove_artificial(tableau, basis, artificial, variable_names):
    """ Функция удаления искусственных переменных после первой фазы """

    artificial = set(artificial)

    # Оставляем только столбцы исходных и дополнительных переменных
    columns_to_keep = [j for j in range(tableau.shape[1] - 1) if j not in artificial]

    new_tableau = tableau[:, columns_to_keep]
    new_tableau = np.column_stack((new_tableau, tableau[:, -1]))

    new_basis = []

    for basic_variable in basis:

        if basic_variable in columns_to_keep:
            new_basis.append(columns_to_keep.index(basic_variable))  # обновляем индексы после удаления столбцов
        else:
            new_basis.append(-1)  # если искусственная переменная осталась в базисе

    new_variable_names = [variable_names[j] for j in columns_to_keep]

    return new_tableau, new_basis, new_variable_names, columns_to_keep


def fix_basis(tableau, basis, variable_names):
    """ Функция восстановления базиса после удаления искусственных переменных """

    rows_count = len(basis)
    columns_count = tableau.shape[1] - 1

    for i in range(rows_count):

        if basis[i] != -1:
            continue

        for j in range(columns_count):

            if j in basis:
                continue

            column = tableau[:rows_count, j]

            if abs(column[i]) < 1e-10:
                continue

            pivot = tableau[i, j]

            tableau[i] /= pivot  # нормируем строку по выбранному разрешающему элементу

            for k in range(rows_count + 1):
                if k != i:
                    coefficient = tableau[k, j]
                    tableau[k] -= coefficient * tableau[i]  # обнуляем выбранный столбец в остальных строках

            basis[i] = j  # добавляем найденную переменную в базис
            break

    return tableau, basis


def phase_two(inequalities_coefs, ineq_constant_coefs, function, basis, variable_names):
    """ Функция решения основной задачи """
    print("\nФаза II — основная задача\n")

    tableau = create_tableau(inequalities_coefs, ineq_constant_coefs, function, basis)
    tableau, basis = simplex(tableau, basis, variable_names)

    return tableau, basis


def get_solution(tableau, basis, variable_names, task_type, function):
    """ Функция получения итогового решения """

    solution = np.zeros(len(variable_names))

    for i, basic_variable in enumerate(basis):
        if basic_variable != -1:
            solution[basic_variable] = tableau[i, -1]  # свободный член строки равен значению базисной переменной

    function_value = np.dot(function, solution)  # вычисляем исходную целевую функцию по найденным значениям переменных

    if task_type == "min":
        function_value *= - 1  # возвращаем исходную целевую функцию

    print("\nРешение\n")

    for i, value in enumerate(solution):
        print(f"{variable_names[i]} = {value:.6f}")

    print(f"F = {function_value:.6f}")

    return solution, function_value


def main():
    """ Главная функция программы """

    task_type, function, inequalities_coefs, ineq_constant_coefs, ineq_signs = input_problem()

    inequalities_coefs, ineq_constant_coefs, function, variable_names = make_canonical(task_type, function, inequalities_coefs, ineq_constant_coefs, ineq_signs)

    print("\nКанонический вид\n")
    print("A:", inequalities_coefs)
    print("\nb:", ineq_constant_coefs)
    print("\nF:", function)
    print("\nПеременные:", variable_names)
    basis = find_basis(inequalities_coefs)
    print("\nНайденный базис:", [variable_names[i] if i != -1 else "-" for i in basis])

    if -1 in basis:
        # Добавляем искусственные переменные только в строки без базиса
        inequalities_coefs, basis, artificial, variable_names = add_artificial_variables(inequalities_coefs, basis, variable_names)

        print("\nПосле добавления искусственных переменных:")
        print("A:")
        print(inequalities_coefs)
        print("Переменные:", variable_names)
        print("Искусственные:", [variable_names[i] for i in artificial])

        tableau, basis = phase_one(
            inequalities_coefs,
            ineq_constant_coefs,
            basis,
            artificial,
            variable_names
        )

        tableau, basis, variable_names, columns_to_keep = remove_artificial(tableau, basis, artificial, variable_names)
        tableau, basis = fix_basis(tableau, basis, variable_names)

        inequalities_coefs = tableau[:-1, :-1]
        ineq_constant_coefs = tableau[:-1, -1]

        function = np.array([function[j] if j < len(function) else 0 for j in range(len(variable_names))])

    else:
        # Если базис уже найден, вспомогательная задача не требуется
        pass

    tableau, basis = phase_two(inequalities_coefs, ineq_constant_coefs, function, basis, variable_names)
    get_solution(tableau, basis, variable_names, task_type, function)

main()

"""
Тип задачи (max/min): max
Коэффициенты целевой функции через пробел: 3 1 2 4
Количество ограничений: 3
Коэффициенты 1-го ограничения: 2 1 1 0
Знак ограничения 1 (<=, >=, =): <=
Правая часть 1-го ограничения: 8
Коэффициенты 2-го ограничения: 1 0 1 1
Знак ограничения 2 (<=, >=, =): =
Правая часть 2-го ограничения: 6
Коэффициенты 3-го ограничения: 0 1 0 1
Знак ограничения 3 (<=, >=, =): >=
Правая часть 3-го ограничения: 4

Верный ответ x2 = 8, x4 = 6, F = 32
________________________________________________________

Тип задачи (max/min): max
Коэффициенты целевой функции через пробел: 2 3
Количество ограничений: 2
Коэффициенты 1-го ограничения: 1 1
Знак ограничения 1 (<=, >=, =): =
Правая часть 1-го ограничения: 4
Коэффициенты 2-го ограничения: 1 2
Знак ограничения 2 (<=, >=, =): >=
Правая часть 2-го ограничения: 3

Верный ответ x2 = 4, F = 12
"""
