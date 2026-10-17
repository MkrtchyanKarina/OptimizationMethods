
import sympy as sp

TOL = 1e-8


def input_problem():
    """Ввод целевой функции, переменных и ограничений."""
    names = input(
        "Введите переменные через пробел (например, x y z): "
    ).split()

    if not names or len(set(names)) != len(names):
        raise ValueError("Переменные должны быть непустыми и уникальными.")

    if any(not name.isidentifier() for name in names):
        raise ValueError("Некорректное имя переменной.")

    variables = list(sp.symbols(" ".join(names), real=True))
    if not isinstance(variables, list):
        variables = [variables]

    local_dict = dict(zip(names, variables))
    local_dict.update({
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "pi": sp.pi,
        "E": sp.E,
    })

    print("\nВведите целевую функцию f.")
    print("Пример: x**2 + y**2 + 2*z**2")
    f = sp.sympify(input("f = "), locals=local_dict)

    if f.free_symbols - set(variables):
        raise ValueError("В функции есть неизвестные переменные.")

    count = int(input("\nКоличество ограничений-равенств: "))
    if count < 1:
        raise ValueError("Нужно ввести хотя бы одно ограничение.")

    constraints = []

    print(
        "Введите каждое ограничение в форме h = 0.\n"
        "Например, для x + y + z = 4 введите x + y + z - 4."
    )

    for i in range(count):
        h = sp.sympify(input(f"h{i + 1} = "), locals=local_dict)

        if h.free_symbols - set(variables):
            raise ValueError("В ограничении есть неизвестные переменные.")

        constraints.append(h)

    return variables, f, constraints


def is_real_solution(value):
    """Проверка действительности найденного значения."""
    value = sp.simplify(value)

    if value.is_real is True:
        return True

    if value.is_real is False:
        return False

    try:
        number = complex(sp.N(value))
        return (
            abs(number.imag) <= TOL
            and abs(number.real) < float("inf")
        )
    except (TypeError, ValueError):
        return False


def check_solution(solution, variables, multipliers,
                   constraints, f):
    """Проверка кандидата и вычисление характеристик."""
    point = {
        v: sp.simplify(solution[v])
        for v in variables
    }

    lambda_values = {
        a: sp.simplify(solution[a])
        for a in multipliers
    }

    residuals = [
        sp.simplify(h.subs(point))
        for h in constraints
    ]

    feasible = True

    for residual in residuals:
        try:
            if abs(float(sp.N(residual))) > TOL:
                feasible = False
        except (TypeError, ValueError):
            feasible = False

    value = sp.simplify(f.subs(point))

    return point, lambda_values, residuals, feasible, value


def classify_problem(variables, f, constraints):
    """
    Строгая выпуклость/вогнутость для квадратичных функций
    с линейными ограничениями.
    """
    n = len(variables)
    m = len(constraints)

    jacobian = sp.Matrix([
        [sp.diff(h, v) for v in variables]
        for h in constraints
    ])

    rank = jacobian.rank()

    affine_constraints = all(
        sp.diff(h, a, b) == 0
        for h in constraints
        for a in variables
        for b in variables
    )

    hessian = sp.hessian(f, variables)
    constant_hessian = all(
        sp.diff(hessian[i, j], v) == 0
        for i in range(n)
        for j in range(n)
        for v in variables
    )

    positive_definite = False
    negative_definite = False

    if constant_hessian:
        eigenvalues = list(hessian.eigenvals().keys())

        positive_definite = all(
            sp.ask(sp.Q.positive(ev)) is True
            for ev in eigenvalues
        )
        negative_definite = all(
            sp.ask(sp.Q.negative(ev)) is True
            for ev in eigenvalues
        )

    print("\nАНАЛИЗ УСЛОВИЙ")
    print("Матрица Якоби ограничений:")
    sp.pprint(jacobian)
    print("Ранг матрицы Якоби:", rank)
    print("Число переменных:", n)
    print("Число ограничений:", m)
    print("Ограничения аффинные:", affine_constraints)
    print("Гессиан целевой функции:")
    sp.pprint(hessian)

    return {
        "rank": rank,
        "affine": affine_constraints,
        "constant_hessian": constant_hessian,
        "positive_definite": positive_definite,
        "negative_definite": negative_definite,
    }


def main():
    variables, f, constraints = input_problem()

    multipliers = sp.symbols(
        f"lambda1:{len(constraints) + 1}",
        real=True
    )

    print("\n" + "=" * 55)
    print("ПОСТАНОВКА ЗАДАЧИ")
    print("=" * 55)

    print("Целевая функция:")
    sp.pprint(f)

    print("Ограничения:")
    for h in constraints:
        sp.pprint(sp.Eq(h, 0))

    # Построение функции Лагранжа
    L = f + sum(a * h for a, h in zip(multipliers, constraints))

    print("\nФункция Лагранжа:")
    sp.pprint(L)

    # Формирование системы стационарности
    equations = [ sp.diff(L, v) for v in variables ] + constraints

    unknowns = list(variables) + list(multipliers)

    print("\nУРАВНЕНИЯ СТАЦИОНАРНОСТИ")
    for equation in equations:
        sp.pprint(sp.Eq(equation, 0))

    # Исследование регулярности и выпуклости
    properties = classify_problem(variables, f, constraints)

    # Решение системы Лагранжа
    print("\nРЕШЕНИЕ СИСТЕМЫ")

    try:
        solutions = sp.solve(
            equations, unknowns, dict=True
        )
    except (NotImplementedError, ValueError) as error:
        print("Символьный решатель не смог решить систему:", error)
        print("Автоматическая классификация невозможна.")
        return

    if not solutions:
        print("Решатель не вернул решений.")
        print(
            "Это не всегда доказывает отсутствие кандидатов: "
            "система может быть слишком сложной для solve."
        )
        return

    candidates = []

    for solution in solutions:
        # Пропускаем параметрические решения:
        # для них нужна отдельная обработка.
        if any(u not in solution for u in unknowns):
            continue

        if not all(
            is_real_solution(solution[u])
            for u in unknowns
        ):
            continue

        result = check_solution(
            solution, variables, multipliers,
            constraints, f
        )

        point, lambdas, residuals, feasible, value = result

        print(f"\nКАНДИДАТ {len(candidates) + 1}")
        print("Точка:")
        sp.pprint(tuple(point[v] for v in variables))
        print("Множители:")
        sp.pprint(lambdas)
        print("Невязки ограничений:")
        sp.pprint(residuals)
        print("Допустимость:", "да" if feasible else "нет")
        print("Значение функции:")
        sp.pprint(value)

        if feasible:
            candidates.append(result)

    print("\n" + "=" * 55)
    print("ИТОГ")
    print("=" * 55)

    if not candidates:
        print("Допустимые действительные кандидаты не найдены.")
        print(
            "Нельзя заключать, что экстремумов нет: "
            "нужно дополнительно исследовать систему."
        )
        return

    # Проверяем, не является ли множество допустимых точек
    # единственной точкой. Это верно для линейных ограничений
    # полного ранга при числе ограничений, равном числу переменных.
    singleton = (
        properties["affine"]
        and properties["rank"] == len(variables)
    )

    if singleton and len(candidates) == 1:
        point, lambdas, residuals, feasible, value = candidates[0]
        print("Допустимое множество состоит из одной точки.")
        print("Эта точка является глобальным минимумом и максимумом.")
        print("Значение функции:", value)
        return

    if properties["affine"] and properties["positive_definite"]:
        best = min(candidates, key=lambda item: float(sp.N(item[4])))
        print("Глобальный минимум:")
        sp.pprint(tuple(best[0][v] for v in variables))
        print("Минимальное значение:", best[4])

        if properties["rank"] < len(variables):
            print(
                "Глобального максимума нет: допустимое множество "
                "неограниченно, а строго выпуклая квадратичная "
                "функция неограниченно возрастает на нём."
            )
        else:
            print(
                "Допустимое множество имеет единственную точку; "
                "она также является глобальным максимумом."
            )

    elif properties["affine"] and properties["negative_definite"]:
        best = max(candidates, key=lambda item: float(sp.N(item[4])))
        print("Глобальный максимум:")
        sp.pprint(tuple(best[0][v] for v in variables))
        print("Максимальное значение:", best[4])

        if properties["rank"] < len(variables):
            print(
                "Глобального минимума нет: на неограниченном "
                "допустимом множестве функция неограниченна снизу."
            )

    else:
        print(
            "Глобальные экстремумы автоматически не классифицированы."
        )
        print(
            "Найденные кандидаты получены из системы Лагранжа, "
            "но необходимо отдельно исследовать вторые производные, "
            "особые точки и поведение функции на допустимом множестве."
        )

    if properties["rank"] < len(constraints):
        print(
            "\nВНИМАНИЕ: ограничения могут быть зависимыми. "
            "Необходимо отдельно исследовать регулярность "
            "и особые допустимые точки."
        )

    print(
        "\nОграничение программы: вывод solve не гарантирует "
        "нахождение всех решений произвольной нелинейной системы. "
        "Параметрические семейства и особые точки требуют "
        "дополнительного исследования."
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, TypeError, SyntaxError) as error:
        print("Ошибка ввода:", error)
