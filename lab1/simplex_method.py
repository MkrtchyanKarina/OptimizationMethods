import numpy as np


class UnboundedError(Exception):
    """ Ошибка, возникающая, когда целевая функция не ограничена и оптимума не существует """
    pass


def ask_negative_variables(variables_count):
    """ Функция, уточняющая у пользователя, какие переменные могут быть отрицательными """

    print("\nПо умолчанию все переменные считаются неотрицательными (x >= 0).")

    answer = input("Есть ли переменные, которые могут быть отрицательными? (y/n): ").lower()

    if answer not in ("y", "n"):
        raise ValueError("Ответ должен быть y или n")

    negative_variables = [False] * variables_count  # по умолчанию ни одна переменная не отрицательна

    if answer == "n":
        return negative_variables

    answer = input("Номера таких переменных через пробел (или all, если отрицательны все): ").lower()

    if answer == "all":
        return [True] * variables_count

    for number in answer.split():
        index = int(number) - 1  # пользователь нумерует переменные с единицы, а мы работаем с нуля

        if not 0 <= index < variables_count:
            raise ValueError(f"Номер переменной должен быть от 1 до {variables_count}")

        negative_variables[index] = True

    return negative_variables


def input_problem():
    """ Функция для ввода данных задачи """
    task_type = input("Тип задачи (max/min): ").lower()

    if task_type not in ("max", "min"):
        raise ValueError("Тип задачи должен быть max или min")

    function = list(map(float, input("Коэффициенты целевой функции через пробел: ").split()))
    inequalities_count = int(input("Количество ограничений: "))

    inequalities_coefs = []  # коэффициенты при x в неравенствах (A)
    ineq_constant_coafs = []  # свободные коэффициенты в правой части неравенств (b)
    ineq_signs = []  # знак неравенства: =, <=, >=

    for i in range(inequalities_count):
        row = list(map(float, input(f"Коэффициенты {i + 1}-го ограничения: ").split()))

        if len(row) != len(function):
            raise ValueError("Число коэффициентов в ограничении должно совпадать с числом переменных")

        sign = input(f"Знак ограничения {i + 1} (<=, >=, =): ").strip()
        value = float(input(f"Правая часть {i + 1}-го ограничения: "))

        inequalities_coefs.append(row)
        ineq_signs.append(sign)
        ineq_constant_coafs.append(value)

    negative_variables = ask_negative_variables(len(function))  # уточняем, какие x могут быть отрицательными

    return task_type, function, inequalities_coefs, ineq_constant_coafs, ineq_signs, negative_variables


def substitute_negative_variables(function, inequalities_coefs, ineq_constant_coafs, negative_variables):
    """ Функция замены отрицательных переменных x = x' - x'' на пару неотрицательных переменных """

    function = np.array(function, dtype=float)
    inequalities_coefs = np.array(inequalities_coefs, dtype=float)

    variable_names = []  # имена переменных после замены
    new_function = []  # коэффициенты целевой функции после замены
    new_columns = []  # столбцы матрицы ограничений после замены

    for j in range(len(function)):

        if negative_variables[j]:
            # отрицательную переменную представляем разностью двух неотрицательных: x = x' - x'',
            # поэтому коэффициент функции и столбец ограничения берутся дважды, с разными знаками
            column = inequalities_coefs[:, j]

            variable_names.append(f"x{j + 1}'")
            new_function.append(function[j])
            new_columns.append(column)

            variable_names.append(f"x{j + 1}''")
            new_function.append(-function[j])
            new_columns.append(-column)

        else:
            variable_names.append(f"x{j + 1}")  # неотрицательная переменная остается без изменений
            new_function.append(function[j])
            new_columns.append(inequalities_coefs[:, j])

    new_function = np.array(new_function, dtype=float)
    new_inequalities_coefs = np.column_stack(new_columns)  # заново собираем матрицу ограничений из новых столбцов

    return new_inequalities_coefs, ineq_constant_coafs, new_function, variable_names


def fix_negative_right_parts(inequalities_coefs, ineq_constant_coafs, ineq_signs):
    """ Функция устранения отрицательных правых частей ограничений """

    ineq_signs = list(ineq_signs)  # работаем с копией, чтобы не менять знаки, заданные пользователем

    for i, value in enumerate(ineq_constant_coafs):

        if value >= -1e-10:
            continue  # неотрицательная правая часть не требует изменений

        # умножаем строку ограничения на -1, чтобы правая часть стала неотрицательной,
        # иначе базис из добавочных переменных сразу окажется недопустимым
        inequalities_coefs[i] = -inequalities_coefs[i]
        ineq_constant_coafs[i] = -value

        if ineq_signs[i] == "<=":
            ineq_signs[i] = ">="  # при умножении на -1 неравенство меняет направление
        elif ineq_signs[i] == ">=":
            ineq_signs[i] = "<="

    return inequalities_coefs, ineq_constant_coafs, ineq_signs


def pad_function(function, columns_count):
    """ Функция дополнения коэффициентов целевой функции нулями до числа столбцов симплекс-таблицы """

    padded_function = np.zeros(columns_count, dtype=float)
    padded_function[:len(function)] = function  # добавочные переменные входят в функцию с нулевыми коэффициентами

    return padded_function


def make_canonical(task_type, function, inequalities_coafs, ineq_constant_coafs, ineq_signs, variable_names):
    """ Функция для приведения задачи к каноническому виду """

    inequalities_coafs = np.array(inequalities_coafs, dtype=float)
    ineq_constant_coafs = np.array(ineq_constant_coafs, dtype=float)
    function = np.array(function, dtype=float)

    if task_type == "min":
        function = -function  # задачу минимизации преобразуем в максимизацию

    # отрицательные правые части убираем до добавления переменных, чтобы сразу учесть смену знака
    inequalities_coafs, ineq_constant_coafs, ineq_signs = fix_negative_right_parts(
        inequalities_coafs, ineq_constant_coafs, ineq_signs
    )

    add_var_number = len(function) + 1  # номер следующей добавочной переменной

    for i, sign in enumerate(ineq_signs):

        if sign == "<=":
            # для <= добавляется дополнительная переменная +s
            new_column = np.zeros(len(ineq_constant_coafs))  # создаем столбец из нулей
            new_column[i] = 1  # в нужной строке (неравенстве) ставим коэффициент 1

            inequalities_coafs = np.column_stack((inequalities_coafs, new_column))  # присоединяем новый столбец к остальным коэффициентам неравенств
            variable_names.append(f"x{add_var_number}")  # добавляем новую переменную в список

            add_var_number += 1  # увеличиваем порядковый номер добавочного коэффициента

        elif sign == ">=":
            # для >= добавляется избыточная переменная -s
            new_column = np.zeros(len(ineq_constant_coafs))  # создаем столбец из нулей
            new_column[i] = -1  # в нужной строке (неравенстве) ставим коэффициент -1

            inequalities_coafs = np.column_stack((inequalities_coafs, new_column))  # присоединяем новый столбец к остальным коэффициентам неравенств
            variable_names.append(f"x{add_var_number}")  # добавляем новую переменную в список

            add_var_number += 1  # увеличиваем порядковый номер добавочного коэффициента

        elif sign == "=":
            pass  # для равенства дополнительная переменная не добавляется

        else:
            raise ValueError("Знак должен быть <=, >= или =")


    return inequalities_coafs, ineq_constant_coafs, function, variable_names, ineq_signs


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


def create_tableau(inequalities_coefs, ineq_constant_coafs, function, basis):
    """ Функция формирования симплекс-таблицы """

    # коэффициенты целевой функции дополняем нулями до ширины таблицы, иначе строка не совпадет с ней по длине
    function_row = pad_function(function, inequalities_coefs.shape[1])

    # объединяем A и b, затем добавляем последнюю строку с коэффициентами целевой функции
    tableau = np.vstack((np.column_stack((inequalities_coefs, ineq_constant_coafs)), np.append(-function_row, 0)))

    for i, basic_variable in enumerate(basis):

        if basic_variable == -1:
            continue  # для строки без базисной переменной строку целевой функции не корректируем

        coefficient = tableau[-1, basic_variable]  # для каждой базисной переменной берем ее коэффициент из исходной функции
        tableau[-1] -= coefficient * tableau[i]  # выражаем базисные переменные из ограничений и корректируем целевую функцию

    return tableau


def pivot_step(tableau, pivot_row, pivot_column):
    """ Функция пересчета симплекс-таблицы по разрешающему элементу """

    pivot = tableau[pivot_row, pivot_column]

    old_tableau = tableau.copy()  # сохраняем старую таблицу, чтобы все расчеты выполнить по исходным значениям

    # разрешающую строку делим на разрешающий элемент
    tableau[pivot_row] = old_tableau[pivot_row] / pivot

    # на месте разрешающего элемента записываем обратное ему число
    tableau[pivot_row, pivot_column] = 1 / pivot

    for i in range(len(tableau)):

        if i == pivot_row:
            continue

        # каждый элемент строки пересчитываем по правилу прямоугольника: (a_ij * a_kk - a_ik * a_kj) / a_kk
        tableau[i] = (old_tableau[i] * pivot - old_tableau[i, pivot_column] * old_tableau[pivot_row]) / pivot

        # разрешающий столбец делим на разрешающий элемент с противоположным знаком
        tableau[i, pivot_column] = -old_tableau[i, pivot_column] / pivot

    return tableau


def simplex(tableau, basis, variable_names):
    """ Функция решения задачи симплекс-методом """

    rows_count = len(basis)
    steps = 0  # число выполненных пересчетов, нужно для защиты от зацикливания
    max_steps = 100 * (rows_count + len(variable_names))  # предел числа пересчетов до перехода к правилу Блэнда

    while True:
        print("\nТекущая симплекс-таблица:")
        print(np.round(tableau, 3))
        print("Базис:", [variable_names[i] for i in basis])

        function_row = tableau[-1, :-1]  # коэффициенты целевой функции без свободного члена

        candidates = []
        has_improving = False  # есть ли столбец, увеличивающий значение целевой функции
        basic_columns = set(basis)  # столбцы текущего базиса входят в базис и не могут быть выбраны снова

        # Ищем отрицательные коэффициенты целевой функции,
        # для которых в столбце есть положительный коэффициент
        for j in range(tableau.shape[1] - 1):

            if j in basic_columns:
                continue  # разрешающий столбец хранит ненулевые значения, поэтому базис проверяем явно

            if function_row[j] >= -1e-10:
                continue

            has_improving = True

            for i in range(rows_count):
                if tableau[i, j] > 1e-10:
                    candidates.append(j)
                    break

        if not has_improving:
            break  # если ни один коэффициент не отрицателен, достигнут оптимум

        if len(candidates) == 0:
            # столбец улучшает функцию, но ограничить его рост нечем,
            # значит допустимое множество не ограничено и оптимума не существует
            raise UnboundedError("Целевая функция не ограничена, оптимальное решение не существует.")

        if steps > max_steps:
            # правило Блэнда: в базис входит переменная с наименьшим номером,
            # оно позволяет гарантированно закончить расчет при вырожденных задачах
            entering = candidates[0]
        else:
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

        leaving_row = int(np.argmin(ratios))  # выбираем минимальное отношение
        leaving = basis[leaving_row]

        print("Отношения:", ratios)
        print("Выходит:", variable_names[leaving])

        print("Разрешающий элемент:", tableau[leaving_row, entering])

        pivot_step(tableau, leaving_row, entering)  # пересчитываем таблицу по разрешающему элементу

        # в этой схеме переменные разрешающей строки и столбца меняются местами,
        # а столбцы таблицы закреплены за переменными, поэтому меняем их содержимое
        tableau[:, [entering, leaving]] = tableau[:, [leaving, entering]]

        basis[leaving_row] = entering  # заменяем вышедшую переменную на вошедшую в базис
        steps += 1  # считаем выполненный пересчет

    return tableau, basis


def phase_one(inequalities_coefs, ineq_constant_coafs, basis, artificial, variable_names):
    """ Функция решения вспомогательной задачи """

    print("\nФаза I — вспомогательная задача\n")

    function_auxiliary = np.zeros(inequalities_coefs.shape[1])

    for variable in artificial:
        function_auxiliary[variable] = -1  # вспомогательная функция минимизирует сумму искусственных переменных

    tableau = create_tableau(inequalities_coefs, ineq_constant_coafs, function_auxiliary, basis)

    try:
        tableau, basis = simplex(tableau, basis, variable_names)
    except UnboundedError:
        # если вспомогательная функция не ограничена, то искусственные переменные не удалось обнулить,
        # а значит, и исходная задача не имеет допустимого решения
        raise ValueError("Исходная задача не имеет допустимого решения.")

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

            pivot_step(tableau, i, j)  # пересчитываем таблицу по найденному разрешающему элементу

            basis[i] = j  # добавляем найденную переменную в базис
            break

    if -1 in basis:
        # если переменная так и не нашлась, то ограничение является избыточным:
        # в этой строке все коэффициенты равны нулю, поэтому удаляем ее из таблицы
        print("\nОбнаружено избыточное ограничение — оно исключено из расчета.")

        rows_to_keep = [i for i in range(len(basis)) if basis[i] != -1]

        # строку целевой функции, которая всегда идет последней, сохраняем обязательно
        tableau = tableau[rows_to_keep + [len(tableau) - 1]]
        basis = [basis[i] for i in rows_to_keep]

    return tableau, basis


def phase_two(inequalities_coefs, ineq_constant_coafs, function, basis, variable_names):
    """ Функция решения основной задачи """
    print("\nФаза II — основная задача\n")

    tableau = np.column_stack((inequalities_coefs, ineq_constant_coafs))

    function = pad_function(function, tableau.shape[1] - 1)

    # индексную строку строим как index_j = -c_j + сумма c_базисных * a_ij:
    # в этой схеме базисные столбцы хранят обратную матрицу, а не единичную,
    # поэтому просто вычесть строки, как в create_tableau, недостаточно
    objective_row = np.append(-function, 0)

    for i, basic_variable in enumerate(basis):
        if basic_variable != -1:
            objective_row += function[basic_variable] * tableau[i]

    tableau = np.vstack((tableau, objective_row))

    tableau, basis = simplex(tableau, basis, variable_names)

    return tableau, basis


def restore_variables(solution, negative_variables):
    """ Функция обратной замены x = x' - x'' для восстановления исходных значений переменных """

    values = []
    index = 0  # указатель на текущую переменную в векторе решения

    for j, is_negative in enumerate(negative_variables):

        if is_negative:
            values.append(solution[index] - solution[index + 1])  # исходная переменная равна разности двух новых
            index += 2
        else:
            values.append(solution[index])  # неотрицательная переменная при замене не изменилась
            index += 1

    return values


def get_solution(tableau, basis, variable_names, task_type, function, negative_variables):
    """ Функция получения итогового решения """

    solution = np.zeros(len(variable_names))

    for i, basic_variable in enumerate(basis):
        if basic_variable != -1:
            solution[basic_variable] = tableau[i, -1]  # свободный член строки равен значению базисной переменной

    function_value = np.dot(function, solution)  # вычисляем исходную целевую функцию по найденным значениям переменных

    if task_type == "min":
        function_value *= - 1  # возвращаем исходную целевую функцию

    original_values = restore_variables(solution, negative_variables)  # возвращаем значения исходных переменных

    print("\nРешение\n")

    for i, value in enumerate(original_values):
        print(f"x{i + 1} = {value:.6f}")

    if any(negative_variables):
        # выводим введенные при замене переменные, чтобы результат замены можно было проверить
        print("\nЗначения введенных переменных\n")

        for name, value in zip(variable_names, solution):
            if "'" in name:
                print(f"{name} = {value:.6f}")

    print(f"\nF = {function_value:.6f}")

    return solution, function_value, original_values


def solve(task_type, function, inequalities_coefs, ineq_constant_coafs, ineq_signs, negative_variables):
    """ Функция решения задачи от канонического вида до получения ответа """

    variable_names = [f"x{i + 1}" for i in range(len(function))]  # имена исходных переменных

    # отрицательные переменные заменяем на разности неотрицательных, иначе симплекс-метод к ним неприменим
    if any(negative_variables):
        inequalities_coefs, ineq_constant_coafs, function, variable_names = substitute_negative_variables(
            function, inequalities_coefs, ineq_constant_coafs, negative_variables
        )

        print("\nПосле замены отрицательных переменных\n")
        print("Коэффициенты функции:", function)
        print("Переменные:", variable_names)

    inequalities_coefs, ineq_constant_coafs, function, variable_names, ineq_signs = make_canonical(
        task_type, function, inequalities_coefs, ineq_constant_coafs, ineq_signs, variable_names
    )

    function = pad_function(function, len(variable_names))  # дополняем функцию нулями до числа столбцов таблицы

    print("\nКанонический вид\n")
    print("A:", inequalities_coefs)
    print("\nb:", ineq_constant_coafs)
    print("\nF:", function)
    print("\nЗнаки ограничений:", ineq_signs)
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
            ineq_constant_coafs,
            basis,
            artificial,
            variable_names
        )

        tableau, basis, variable_names, columns_to_keep = remove_artificial(tableau, basis, artificial, variable_names)
        tableau, basis = fix_basis(tableau, basis, variable_names)

        inequalities_coefs = tableau[:-1, :-1]
        ineq_constant_coafs = tableau[:-1, -1]

    else:
        # Если базис уже найден, вспомогательная задача не требуется
        pass

    tableau, basis = phase_two(inequalities_coefs, ineq_constant_coafs, function, basis, variable_names)
    get_solution(tableau, basis, variable_names, task_type, function, negative_variables)


def report_error(message):
    """ Функция вывода понятного сообщения об ошибке вместо трассировки стека """

    print(f"\nОшибка: {message}")


def main():
    """ Главная функция программы """

    try:
        task_type, function, inequalities_coefs, ineq_constant_coafs, ineq_signs, negative_variables = input_problem()
    except ValueError as error:
        # некорректный ввод: сообщаем причину и предлагаем начать заново
        report_error(error)
        return

    try:
        solve(task_type, function, inequalities_coefs, ineq_constant_coafs, ineq_signs, negative_variables)
    except ValueError as error:
        # в ходе расчета выяснилось, что допустимых решений нет
        report_error(error)
    except UnboundedError as error:
        # целевая функция не имеет конечного оптимума
        report_error(error)


if __name__ == "__main__":
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
Есть ли переменные, которые могут быть отрицательными? (y/n): n

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
Есть ли переменные, которые могут быть отрицательными? (y/n): n

Верный ответ x2 = 4, F = 12
________________________________________________________

Тип задачи (max/min): max
Коэффициенты целевой функции через пробел: -1 -2
Количество ограничений: 1
Коэффициенты 1-го ограничения: 1 2
Знак ограничения 1 (<=, >=, =): =
Правая часть 1-го ограничения: -3

По умолчанию все переменные считаются неотрицательными (x >= 0).
Есть ли переменные, которые могут быть отрицательными? (y/n): n

Программа сообщает, что задача не имеет допустимого решения,
так как x1 + 2x2 = -3 невозможно при x >= 0.
________________________________________________________

Тип задачи (max/min): max
Коэффициенты целевой функции через пробел: 1 -1
Количество ограничений: 2
Коэффициенты 1-го ограничения: 1 1
Знак ограничения 1 (<=, >=, =): =
Правая часть 1-го ограничения: 4
Коэффициенты 2-го ограничения: 1 -1
Знак ограничения 2 (<=, >=, =): =
Правая часть 2-го ограничения: 1

По умолчанию все переменные считаются неотрицательными (x >= 0).
Есть ли переменные, которые могут быть отрицательными? (y/n): y
Номера таких переменных через пробел (или all, если отрицательны все): 2

После замены x2 = x2' - x2'' верный ответ x1 = 2.5, x2 = 1.5, F = 1
"""
