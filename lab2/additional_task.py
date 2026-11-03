import ast
import numpy as np
import time
import sympy as sp
from functools import lru_cache
import math
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt


MATH_FUNCTIONS = {"sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "sqrt": math.sqrt, "log": math.log, "abs": abs}
ALLOWED_NAMES = {"x", "pi"} | set(MATH_FUNCTIONS)
ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub)


# Проверяет выражение и компилирует его для вычислений
@lru_cache(maxsize=128)
def validate_expression(expression):
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Синтаксическая ошибка в выражении: {error.msg}") from error

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Недопустимая конструкция в выражении: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id not in ALLOWED_NAMES:
            raise ValueError(f"Неизвестное имя: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in MATH_FUNCTIONS:
                raise ValueError("Разрешены только перечисленные математические функции.")
            if len(node.args) != 1 or node.keywords:
                raise ValueError(f"Функция {node.func.id} должна принимать один аргумент.")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Разрешены только числовые константы.")

            if isinstance(node.value, int) and node.value.bit_length() > 1024:
                raise ValueError("Числовая константа слишком велика.")

    # Преобразуем числовые литералы в float, чтобы избежать огромных целочисленных вычислений
    class FloatConstants(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                try:
                    value = float(node.value)
                except OverflowError as error:
                    raise ValueError("Числовая константа слишком велика.") from error
                if not math.isfinite(value):
                    raise ValueError("Числовая константа должна быть конечной.")
                return ast.copy_location(ast.Constant(value=value), node)
            return node

    tree = FloatConstants().visit(tree)
    return compile(ast.fix_missing_locations(tree), "<expression>", "eval")


# Вычисляет значение функции по строковому выражению
def f(x, expression):
    compiled = validate_expression(expression)
    namespace = {"x": float(x), "pi": math.pi, **MATH_FUNCTIONS}

    try:
        result = float(eval(compiled, {"__builtins__": {}}, namespace))
    except (NameError, ValueError, OverflowError, ZeroDivisionError, TypeError) as error:
        raise ValueError(f"Не удалось вычислить функцию в точке x={x}: {error}") from error

    if not math.isfinite(result):
        raise ValueError(f"Функция вернула NaN или Inf в точке x={x}")

    return result


# Находит корень функции методом бисекции
def find_root_bisection(func, left, right, iterations=80):
    f_left = func(left)
    f_right = func(right)

    for _ in range(iterations):
        middle = (left + right) / 2
        f_middle = func(middle)

        if abs(f_middle) < 1e-12:
            return middle

        if f_left * f_middle < 0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle

    return (left + right) / 2





def estimate_lipschitz(evaluate, a, b, expression_for_derivative):
    """
    Оценивает константу Липшица как max |f'(x)| на [a, b]
      1. Символьно дифференцируем f через SymPy.
      2. Пытаемся найти ВСЕ нули f'' на [a, b] через solveset
         (это ловит и кратные корни).
      3. Если solveset не справился — fallback:
           - смена знака f'' -> бисекция (brentq);
           - касание нуля f'' (кратный корень) -> минимизация |f''|.
      4. Кандидаты = {a, b} U {нули f''}. В них считаем |f'|, берём максимум.
      5. Небольшой запас на погрешность.
    """
    warnings = []


    x = sp.Symbol("x", real=True)
    functions = {
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
        "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log,
        "abs": sp.Abs, "pi": sp.pi,
    }
    try:
        expression = sp.sympify(expression_for_derivative, locals={"x": x, **functions})
    except (sp.SympifyError, SyntaxError) as err:
        raise ValueError(f"SymPy не смог разобрать выражение: {err}") from err

    try:
        first_derivative = sp.diff(expression, x)
        second_derivative = sp.diff(first_derivative, x)
    except Exception as err:
        raise ValueError(f"Не удалось символьно продифференцировать: {err}") from err

    # Есть ли в производных Abs / sign / DiracDelta — признак негладкости
    has_nonsmooth = any(
        expr.has(sp.Abs) or expr.has(sp.sign) or expr.has(sp.DiracDelta)
        for expr in (first_derivative, second_derivative)
    )
    if has_nonsmooth:
        warnings.append(
            "Функция содержит abs/sign/δ: в точках излома f' имеет скачок. "
            "Учитываются односторонние пределы, но гарантия условна."
        )

    modules = [{"DiracDelta": lambda z: np.zeros_like(np.asarray(z, dtype=float))}, "numpy"]

    def safe_lambdify(expr):
        fn = sp.lambdify(x, expr, modules=modules)
        def wrapped(t):
            try:
                val = fn(t)
                arr = np.asarray(val, dtype=float)
                return arr if arr.ndim > 0 else float(arr)
            except Exception:
                return np.nan
        return wrapped

    f_prime = safe_lambdify(first_derivative)
    f_second = safe_lambdify(second_derivative)

    critical_points = []

    solveset_ok = False
    try:
        sol_set = sp.solveset(second_derivative, x, domain=sp.Interval(a, b))
        if isinstance(sol_set, sp.FiniteSet):
            for sol in sol_set:
                if sol.is_real and sol.is_finite:
                    critical_points.append(float(sol))
            solveset_ok = True
        elif sol_set is sp.S.EmptySet:
            solveset_ok = True  # корней нет — это тоже валидный ответ
        else:
            warnings.append(
                "solveset вернул не FiniteSet (возможно, трансцендентное уравнение). "
                "Используется численный fallback."
            )
    except Exception as err:
        warnings.append(f"solveset не справился ({err}). Используется численный fallback.")


    if not solveset_ok:
        n_grid = 4000
        xs = np.linspace(a, b, n_grid)
        vals = np.asarray(f_second(xs), dtype=float)

        if not np.all(np.isfinite(vals)):
            warnings.append("f'' содержит NaN/Inf на части сетки; эти узлы пропущены.")
            vals = np.where(np.isfinite(vals), vals, np.nan)


        for i in range(len(xs) - 1):
            yl, yr = vals[i], vals[i + 1]
            if not (math.isfinite(yl) and math.isfinite(yr)):
                continue
            if yl == 0.0:
                critical_points.append(float(xs[i]))
            elif yl * yr < 0:
                try:
                    root = brentq(lambda t: float(f_second(t)), xs[i], xs[i + 1],
                                  maxiter=100, xtol=1e-12)
                    critical_points.append(float(root))
                except (ValueError, RuntimeError):
                    pass

        abs_vals = np.abs(vals)
        for i in range(1, len(xs) - 1):
            if not (math.isfinite(abs_vals[i - 1]) and
                    math.isfinite(abs_vals[i]) and
                    math.isfinite(abs_vals[i + 1])):
                continue

            if abs_vals[i] <= abs_vals[i - 1] and abs_vals[i] <= abs_vals[i + 1]:
                if abs_vals[i] < 1e-3:  # подозрение на корень
                    try:
                        res = minimize_scalar(lambda t: abs(float(f_second(t))),
                            bounds=(xs[i - 1], xs[i + 1]),
                            method="bounded",
                            options={"xatol": 1e-12},
                        )
                        if res.success and res.fun < 1e-10:
                            critical_points.append(float(res.x))
                    except Exception:
                        pass


    critical_points.sort()
    deduped = []
    for p in critical_points:
        if not deduped or abs(p - deduped[-1]) > 1e-10:
            deduped.append(p)
    critical_points = deduped

    candidates = [a, b] + critical_points

    eps_side = 1e-9 * max(1.0, abs(a), abs(b))
    derivative_values = []
    for point in candidates:
        try:
            v_center = abs(float(f_prime(point)))
            if math.isfinite(v_center):
                derivative_values.append(v_center)
            # если есть разрыв f' — проверим пределы слева/справа
            if has_nonsmooth:
                for side in (-eps_side, eps_side):
                    t = point + side
                    if a <= t <= b:
                        v_side = abs(float(f_prime(t)))
                        if math.isfinite(v_side):
                            derivative_values.append(v_side)
        except Exception:
            continue

    if not derivative_values:
        raise ValueError("Не удалось вычислить |f'| ни в одной точке-кандидате.")

    L_exact = max(derivative_values)

    x_coarse = np.linspace(a, b, 10000)
    x_fine = np.linspace(a, b, 20000)
    slopes_coarse = np.abs(np.asarray(f_prime(x_coarse), dtype=float))
    slopes_fine = np.abs(np.asarray(f_prime(x_fine), dtype=float))

    if not np.all(np.isfinite(slopes_coarse)) or not np.all(np.isfinite(slopes_fine)):
        raise ValueError("Не удалось получить конечные значения производной на сетке.")

    max_coarse = float(np.max(slopes_coarse))
    max_fine = float(np.max(slopes_fine))

    denominator = max(max_coarse, max_fine, L_exact, 1e-12)
    relative_change = abs(max_fine - max_coarse) / denominator
    if relative_change > 0.05:
        warnings.append("Оценка максимума производной заметно изменилась при измельчении сетки.")

    if max(max_coarse, max_fine) > L_exact * (1 + 1e-6):
        warnings.append(
            "Сеточный максимум |f'| превышает значение в найденных критических точках: "
            "возможно, пропущен корень f''. L увеличена по сетке."
        )
        L_exact = max(L_exact, max_coarse, max_fine)

    L = 1.05 * max(L_exact, 1e-10)

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
    validate_expression(expression)

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
    L, warnings, coarse_slope, fine_slope = estimate_lipschitz(evaluate, a, b, expression)
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
    print("\nКритерий остановки выполнен.")
elif status == "max_iter":
    print("\nДостигнут лимит итераций. Требуемая точность не подтверждена.")
elif status == "stalled":
    print("\nАлгоритм остановлен из-за повторного выбора точки. Требуемая точность не подтверждена.")

print("Гарантия глобальной точности зависит от достаточности оценки L:\n\tчисленная оценка и проверки сетки не доказывают, что L является верхней границей истинной константы Липшица.")

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
