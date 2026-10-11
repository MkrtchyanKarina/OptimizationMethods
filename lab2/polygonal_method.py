
import math
import time
import numpy as np
import matplotlib.pyplot as plt


# Вычисляет значение функции по строковому выражению
def f(x, expression):
    allowed_functions = {"__builtins__": {}, "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "sqrt": math.sqrt, "log": math.log, "pi": math.pi, "abs": abs}
    try:
        result = float(eval(expression, allowed_functions, {"x": float(x)}))
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as error:
        raise ValueError(f"Не удалось вычислить функцию в точке x={x}: {error}") from error

    if not math.isfinite(result):
        raise ValueError(f"Функция вернула NaN или Inf в точке x={x}")

    return result


# Оценивает константу Липшица по двум сеткам разной плотности
def estimate_lipschitz(evaluate, a, b):
    warnings = []
    x_coarse = np.linspace(a, b, 10000)
    y_coarse = np.array([evaluate(x, "lipschitz") for x in x_coarse])
    slopes_coarse = np.abs(np.diff(y_coarse) / np.diff(x_coarse))
    max_coarse = float(np.max(slopes_coarse))

    x_fine = np.linspace(a, b, 20000)
    y_fine = np.array([evaluate(x, "lipschitz") for x in x_fine])
    slopes_fine = np.abs(np.diff(y_fine) / np.diff(x_fine))
    max_fine = float(np.max(slopes_fine))

    if not np.all(np.isfinite(slopes_coarse)) or not np.all(np.isfinite(slopes_fine)):
        raise ValueError("Не удалось получить конечные значения секущих наклонов.")

    denominator = max(max_coarse, max_fine, 1e-12)
    relative_change = abs(max_fine - max_coarse) / denominator

    if relative_change > 0.05:
        warnings.append("Оценка максимального наклона заметно изменилась при измельчении сетки.")

    # Запас 10% не гарантирует, что найдена истинная константа Липшица
    L = 1.1 * max(max_coarse, max_fine, 1e-10)
    return L, warnings, max_coarse, max_fine


# Проверяет наблюдаемые секущие наклоны и при необходимости увеличивает L
def check_lipschitz_constant(x_points, y_points, L, warnings):
    order = np.argsort(x_points)
    xs = np.asarray(x_points)[order]
    ys = np.asarray(y_points)[order]

    if len(xs) < 2:
        return L

    max_slope = float(np.max(np.abs(np.diff(ys) / np.diff(xs))))

    if not math.isfinite(max_slope):
        raise ValueError("Обнаружен некорректный секущий наклон.")

    if max_slope > L * (1 + 1e-10):
        L = 1.1 * max_slope
        warnings.append("Обнаружен секущий наклон выше текущего L. Оценка L увеличена.")

    return L


# Вычисляет значение миноранты p(x)
def minorant(x, x_points, y_points, L):
    values = [y_points[i] - L * abs(x - x_points[i]) for i in range(len(x_points))]
    return max(values)


# Находит минимум текущей миноранты
def find_minorant_minimum(x_points, y_points, L, a, b):
    candidates = [a, b]
    order = np.argsort(x_points)
    xs = np.asarray(x_points)[order]
    ys = np.asarray(y_points)[order]
    candidates.extend(xs)

    # Ищем пересечения минорант соседних точек
    for i in range(len(xs) - 1):
        x1, x2 = xs[i], xs[i + 1]
        y1, y2 = ys[i], ys[i + 1]
        x_cross = (y1 - y2) / (2 * L) + (x1 + x2) / 2
        if x1 <= x_cross <= x2:
            candidates.append(x_cross)

    candidates = [x for x in candidates if a <= x <= b]
    best_x = min(candidates, key=lambda x: minorant(x, x_points, y_points, L))
    best_y = minorant(best_x, x_points, y_points, L)
    return best_x, best_y


# Метод Пиявского для поиска глобального минимума
def piyavskiy_method(expression, a, b, eps, max_iter=10000):
    if not all(math.isfinite(value) for value in (a, b, eps)):
        raise ValueError("Границы и точность должны быть конечными числами.")
    if a >= b:
        raise ValueError("Левая граница a должна быть меньше правой b.")
    if eps <= 0:
        raise ValueError("Точность eps должна быть положительной.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter должен быть положительным целым числом.")

    start_time = time.perf_counter()
    evaluations = {"lipschitz": 0, "algorithm": 0}

    def evaluate(x, stage):
        evaluations[stage] += 1
        return f(x, expression)

    # Оцениваем константу Липшица
    L, warnings, coarse_slope, fine_slope = estimate_lipschitz(evaluate, a, b)
    x_points = [a, b]
    y_points = [evaluate(a, "algorithm"), evaluate(b, "algorithm")]
    L = check_lipschitz_constant(x_points, y_points, L, warnings)
    iterations = 0
    status = "max_iter"
    final_gap = float("inf")

    while iterations < max_iter:
        L = check_lipschitz_constant(x_points, y_points, L, warnings)
        x_new, p_min = find_minorant_minimum(x_points, y_points, L, a, b)
        best_index = int(np.argmin(y_points))
        best_y = y_points[best_index]
        final_gap = best_y - p_min

        # Проверяем разрыв между лучшим значением функции и минимумом миноранты
        if final_gap < eps:
            status = "converged"
            break

        # Защита от повторного выбора уже исследованной точки
        x_tolerance = 1e-12 * max(1.0, abs(a), abs(b))
        if any(abs(x_new - x_old) <= x_tolerance for x_old in x_points):
            status = "stalled"
            warnings.append("Алгоритм не смог выбрать новую отличную точку.")
            break

        y_new = evaluate(x_new, "algorithm")
        x_points.append(x_new)
        y_points.append(y_new)
        iterations += 1

    best_index = int(np.argmin(y_points))
    x_min, f_min = x_points[best_index], y_points[best_index]
    L = check_lipschitz_constant(x_points, y_points, L, warnings)
    _, p_min = find_minorant_minimum(x_points, y_points, L, a, b)
    final_gap = f_min - p_min

    if status == "max_iter" and final_gap < eps:
        status = "converged"

    elapsed_time = time.perf_counter() - start_time

    return (x_min, f_min, iterations, elapsed_time, L, x_points, y_points,
            status, final_gap, evaluations, warnings, coarse_slope, fine_slope)


# Ввод исходных данных
expression = input("Введите функцию f(x), например x**2 + 10*(1 - cos(2*pi*x)): ")

try:
    a = float(input("Введите левую границу отрезка: "))
    b = float(input("Введите правую границу отрезка: "))
    eps = float(input("Введите точность eps: "))
    max_iter_input = input("Введите максимальное число итераций (Enter = 10000): ")
    max_iter = int(max_iter_input) if max_iter_input.strip() else 10000

    (x_min, f_min, iterations, elapsed_time, L, x_points, y_points, status,
     final_gap, evaluations, warnings, coarse_slope, fine_slope) = piyavskiy_method(expression, a, b, eps, max_iter)

except (ValueError, SyntaxError) as error:
    print(f"\nОшибка: {error}")
    raise SystemExit(1)


# Строим график исходной функции
x = np.linspace(a, b, 2000)

try:
    y = [f(value, expression) for value in x]
except ValueError as error:
    print(f"\nНе удалось построить график: {error}")
    raise SystemExit(1)

p = [minorant(value, x_points, y_points, L) for value in x]


# Вывод результатов
print("\nРезультаты:")
print(f"Статус завершения: {status}")
print(f"Константа Липшица L = {L:.8f}")
print(f"Приближённое значение x_min = {x_min:.8f}")
print(f"Минимальное значение f(x_min) = {f_min:.8f}")
print(f"Разрыв f_min - p_min = {final_gap:.8e}")
print(f"Количество итераций = {iterations}")
print(f"Время вычисления алгоритма = {elapsed_time:.6f} с")
print("\nВычисления функции:")
print(f"При оценке L: {evaluations['lipschitz']}")
print(f"В алгоритме: {evaluations['algorithm']}")
print(f"Всего в алгоритме и при оценке L: {sum(evaluations.values())}")
print(f"Для построения графика: {len(x)}")
print("\nПроверка оценки L:")
print(f"Максимальный наклон на грубой сетке: {coarse_slope:.8f}")
print(f"Максимальный наклон на мелкой сетке: {fine_slope:.8f}")

if status == "converged":
    print("\nКритерий остановки выполнен. Строгая гарантия точности зависит от корректности L.")
elif status == "max_iter":
    print("\nДостигнут лимит итераций. Требуемая точность не подтверждена.")
elif status == "stalled":
    print("\nАлгоритм остановлен из-за повторного выбора точки. Требуемая точность не подтверждена.")

if warnings:
    print("\nПредупреждения:")
    for warning in dict.fromkeys(warnings):
        print(f"- {warning}")


# Построение графика
plt.figure(figsize=(12, 7))
plt.plot(x, y, label="Исходная функция W(x)", linewidth=2)
plt.plot(x, p, "--", label="Итоговая миноранта p(x)", linewidth=1.5)
plt.scatter(x_points, y_points, s=25, label="Вычисленные точки")
plt.scatter([x_min], [f_min], s=100, label="Найденный минимум", zorder=5)
plt.xlabel("x")
plt.ylabel("W(x)")
plt.title("Метод Пиявского")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
