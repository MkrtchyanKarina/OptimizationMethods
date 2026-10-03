import math
import time
import numpy as np
import matplotlib.pyplot as plt


# Вычисляет значение функции по строковому выражению
def f(x, expression):
    return eval(
        expression,
        {
            "x": x,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "sqrt": math.sqrt,
            "log": math.log,
            "pi": math.pi
        }
    )


# Оценивает константу Липшица по численной производной
def estimate_lipschitz(expression, a, b):
    # Используем большое количество точек для оценки максимального наклона
    x = np.linspace(a, b, 10000)
    y = np.array([f(value, expression) for value in x])

    # Максимальный модуль производной приближённо равен максимальному отношению изменения функции к изменению x
    slopes = np.abs(np.diff(y) / np.diff(x))

    # Небольшой запас нужен, чтобы оценка не оказалась меньше настоящей константы Липшица
    L = 1.1 * np.max(slopes)

    return max(L, 1e-10)


# Вычисляет значение миноранты p(x)
def minorant(x, x_points, y_points, L):
    # p(x) = max(W(ui) - L * |x - ui|)
    values = [y_points[i] - L * abs(x - x_points[i]) for i in range(len(x_points))]
    return max(values)


# Находит минимум миноранты p(x)
def find_minorant_minimum(x_points, y_points, L, a, b):
    candidates = []

    # Уже вычисленные точки тоже являются кандидатами
    candidates.extend(x_points)

    # Ищем точки пересечения соседних участков миноранты
    order = np.argsort(x_points)
    xs = np.array(x_points)[order]
    ys = np.array(y_points)[order]

    for i in range(len(xs) - 1):
        x1 = xs[i]
        x2 = xs[i + 1]
        y1 = ys[i]
        y2 = ys[i + 1]

        # Формула точки пересечения двух прямых с противоположными углами наклона:
        # u_cross = (W(u1) - W(u2)) / (2L) + (u1 + u2) / 2
        x_cross = ((y1 - y2) / (2 * L) + (x1 + x2) / 2)

        # Точка пересечения должна находиться внутри рассматриваемого интервала
        if x1 <= x_cross <= x2:
            candidates.append(x_cross)

    # Выбираем кандидата, где миноранта имеет минимальное значение
    best_x = min(candidates, key=lambda x: minorant(x, x_points, y_points, L))
    best_y = minorant(best_x, x_points, y_points, L)

    return best_x, best_y


# Метод Пиявского для поиска глобального минимума
def piyavskiy_method(expression, a, b, eps):
    start_time = time.perf_counter()

    # Оцениваем константу Липшица
    L = estimate_lipschitz(expression, a, b)

    # Начальные точки u0 = a и u1 = b
    x_points = [a, b]
    y_points = [f(a, expression), f(b, expression)]

    iterations = 0

    # Основной итерационный процесс
    while True:

        # Ищем минимум текущей миноранты
        x_new, p_min = find_minorant_minimum(x_points, y_points, L, a, b)

        # Находим лучшее значение исходной функции
        best_index = np.argmin(y_points)
        best_y = y_points[best_index]

        # Проверяем условие остановки: W(u_k) - p_(k-1)(u_k) < eps
        if best_y - p_min < eps:
            break

        # Вычисляем исходную функцию в новой точке
        y_new = f(x_new, expression)

        # Добавляем новую вершину ломаной
        x_points.append(x_new)
        y_points.append(y_new)

        iterations += 1

    # Находим найденный глобальный минимум
    best_index = np.argmin(y_points)

    x_min = x_points[best_index]
    f_min = y_points[best_index]

    elapsed_time = time.perf_counter() - start_time

    return (
        x_min,
        f_min,
        iterations,
        elapsed_time,
        L,
        x_points,
        y_points
    )


# Ввод исходных данных
expression = input("Введите функцию f(x), например x**2 + 10*(1 - cos(2*pi*x)): ")
a = float(input("Введите левую границу отрезка: "))
b = float(input("Введите правую границу отрезка: "))
eps = float(input("Введите точность eps: "))


# Запуск метода Пиявского
(
    x_min,
    f_min,
    iterations,
    elapsed_time,
    L,
    x_points,
    y_points
) = piyavskiy_method(expression, a, b, eps)


# Вывод результатов
print("\nРезультаты:")
print(f"Константа Липшица L = {L:.8f}")
print(f"Приближённое значение x_min = {x_min:.8f}")
print(f"Минимальное значение f(x_min) = {f_min:.8f}")
print(f"Количество итераций = {iterations}")
print(f"Время вычисления = {elapsed_time:.6f} с")


# Строим график исходной функции
x = np.linspace(a, b, 2000)
y = [f(value, expression) for value in x]

plt.figure((12, 7))

# Исходная функция
plt.plot(x, y, label="Исходная функция W(x)", linewidth=2)

# Строим итоговую миноранту p(x)
p = [ minorant(value, x_points, y_points, L) for value in x]

plt.plot(x, p, "--", label="Итоговая ломаная p(x)", linewidth=1.5)

# Все вычисленные точки
plt.scatter(x_points, y_points, s=25, label="Вычисленные точки")

# Найденный минимум
plt.scatter([x_min],[f_min], s=100, label="Найденный минимум", zorder=5)

plt.xlabel("x")
plt.ylabel("W(x)")
plt.title("Метод Пиявского")
plt.grid(True)
plt.legend()
plt.show()