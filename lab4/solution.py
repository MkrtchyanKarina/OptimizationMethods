from functools import lru_cache
import time

# Исходные данные (в копейках)
# Начальные стоимости: ЦБ1=100, ЦБ2=800, Депозит=400, Свободные средства=600 д.е.
INIT = (10000, 80000, 40000, 60000)  # a1, a2, dep, cash
INITIAL_TOTAL = sum(INIT)            # 190000 коп. = 1900 д.е.

# Минимально допустимые объёмы после сделок
MIN = (3000, 15000, 10000)           # 30, 150, 100 д.е.

# Размер пакета: четверть начальной стоимости актива
PKG = (2500, 20000, 10000)           # 25, 200, 100 д.е.

# Комиссии: 4%, 7%, 5%
BUY_COST = (
    PKG[0] + round(PKG[0] * 0.04),   # 2500 + 100 = 2600
    PKG[1] + round(PKG[1] * 0.07),   # 20000 + 1400 = 21400
    PKG[2] + round(PKG[2] * 0.05),   # 10000 + 500 = 10500
)
SELL_GAIN = (
    PKG[0] - round(PKG[0] * 0.04),   # 2500 - 100 = 2400
    PKG[1] - round(PKG[1] * 0.07),   # 20000 - 1400 = 18600
    PKG[2] - round(PKG[2] * 0.05),   # 10000 - 500 = 9500
)

# Сценарии: (вероятность, коэффициент ЦБ1, ЦБ2, Депозит) в сотых
SCEN = [
    # Этап 1
    [
        (0.60, 120, 110, 107),
        (0.30, 105, 102, 103),
        (0.10,  80,  95, 100),
    ],
    # Этап 2
    [
        (0.30, 140, 115, 101),
        (0.20, 105, 100, 100),
        (0.50,  60,  90, 100),
    ],
    # Этап 3
    [
        (0.40, 115, 112, 105),
        (0.40, 105, 101, 101),
        (0.20,  70,  94, 100),
    ],
]

def apply_coef(amount: int, coef100: int) -> int:
    """Применить коэффициент, заданный в сотых, с округлением до копейки."""
    return (amount * coef100 + 50) // 100

# Ожидаемые коэффициенты 3-го этапа (для аналитического V3)
Q = [0.0, 0.0, 0.0]
for p, m1, m2, m3 in SCEN[2]:
    Q[0] += p * m1 / 100.0
    Q[1] += p * m2 / 100.0
    Q[2] += p * m3 / 100.0
# Q ≈ [1.02, 1.04, 1.024]

def delta_cash(trade, cash):
    """Изменение свободных средств при сделке trade=(d1,d2,d3)."""
    c = cash
    for i, d in enumerate(trade):
        if d > 0:
            c -= d * BUY_COST[i]
        elif d < 0:
            c += (-d) * SELL_GAIN[i]
    return c

def gen_trades(a1, a2, a3, cash):
    """
    Перебор допустимых сделок на этапе.
    d_i > 0 — покупка d_i пакетов,
    d_i < 0 — продажа |d_i| пакетов,
    d_i = 0 — без сделки.
    """
    amounts = (a1, a2, a3)
    max_sell = tuple(
        max(0, (amounts[i] - MIN[i]) // PKG[i])
        for i in range(3)
    )
    proceeds = tuple(max_sell[i] * SELL_GAIN[i] for i in range(3))

    ranges = []
    for i in range(3):
        if amounts[i] >= MIN[i]:
            dmin = -max_sell[i]
        else:
            dmin = (MIN[i] - amounts[i] + PKG[i] - 1) // PKG[i]

        max_cash_for_buy = cash + proceeds[(i + 1) % 3] + proceeds[(i + 2) % 3]
        dmax = max_cash_for_buy // BUY_COST[i]
        if dmax < dmin:
            return
        ranges.append((dmin, dmax))

    # Денежные изменения заранее вычисляются один раз на каждое значение сделки.
    cash_changes = []
    for i in range(3):
        lo, hi = ranges[i]
        changes = {}
        for d in range(lo, hi + 1):
            changes[d] = (-d * BUY_COST[i] if d > 0
                          else (-d) * SELL_GAIN[i] if d < 0 else 0)
        cash_changes.append(changes)

    d3_min, d3_max = ranges[2]
    # Максимальный денежный приток от третьего актива достигается при максимальной продаже.
    best_cash3 = cash_changes[2][d3_min]

    for d1 in range(ranges[0][0], ranges[0][1] + 1):
        a1n = a1 + d1 * PKG[0]
        c1 = cash + cash_changes[0][d1]
        if c1 + cash_changes[1][ranges[1][0]] + best_cash3 < 0:
            continue

        for d2 in range(ranges[1][0], ranges[1][1] + 1):
            a2n = a2 + d2 * PKG[1]
            c12 = c1 + cash_changes[1][d2]
            if c12 + best_cash3 < 0:
                continue

            for d3 in range(d3_min, d3_max + 1):
                cashn = c12 + cash_changes[2][d3]
                if cashn < 0:
                    continue
                a3n = a3 + d3 * PKG[2]
                yield d1, d2, d3, a1n, a2n, a3n, cashn

def validate_input(init=INIT, scenarios=SCEN):
    """Проверка некорректных исходных данных до запуска расчёта."""
    if len(init) != 4 or any(not isinstance(x, int) or x < 0 for x in init):
        raise ValueError("Начальное состояние должно содержать 4 неотрицательных целых значения в копейках")
    if len(scenarios) != 3:
        raise ValueError("Должно быть ровно три этапа")
    for stage in scenarios:
        if len(stage) != 3:
            raise ValueError("На каждом этапе должны быть ровно три сценария")
        if any(len(row) != 4 for row in stage):
            raise ValueError("Каждый сценарий должен содержать вероятность и три коэффициента")
        if any(row[0] < 0 or any(k <= 0 for k in row[1:]) for row in stage):
            raise ValueError("Некорректные вероятности или коэффициенты")
        if abs(sum(row[0] for row in stage) - 1.0) > 1e-9:
            raise ValueError("Сумма вероятностей сценариев каждого этапа должна быть равна 1")


@lru_cache(maxsize=8)
def _check_v3_assumption_cached(q):
    # Кэш зависит от значений Q, поэтому корректно учитывает их изменение в тестах.
    for i in range(3):
        expected_package = q[i] * PKG[i]
        if not (SELL_GAIN[i] < expected_package < BUY_COST[i]):
            raise RuntimeError("Условия V3_fast не выполнены: используйте полный перебор этапа 3")


def check_v3_assumption():
    # Q в тестах временно изменяется, поэтому ключом служит неизменяемый снимок.
    _check_v3_assumption_cached(tuple(Q))

def V3_fast(a1, a2, a3, cash):
    """
    Аналитическое решение для 3-го этапа.
    На последнем шаге все маргинальные доходности отрицательны,
    поэтому добровольные сделки невыгодны. Сделки нужны только:
      1) чтобы добрать активы до минимума;
      2) чтобы профинансировать такие покупки продажей других активов.
    Возвращает (ожидаемая_стоимость, trade).
    """
    check_v3_assumption()
    orig = [a1, a2, a3]
    a = [a1, a2, a3]
    req = [0, 0, 0]
    cost = 0

    for i in range(3):
        if a[i] < MIN[i]:
            need_pkgs = (MIN[i] - a[i] + PKG[i] - 1) // PKG[i]
            req[i] = need_pkgs
            a[i] += need_pkgs * PKG[i]
            cost += need_pkgs * BUY_COST[i]

    if cash >= cost:
        cash_final = cash - cost
        val = Q[0] * a[0] + Q[1] * a[1] + Q[2] * a[2] + cash_final
        return val, (req[0], req[1], req[2])

    # Не хватает cash на обязательные покупки — продаём излишки других активов
    deficit = cost - cash
    max_sell = [0, 0, 0]
    for j in range(3):
        if orig[j] >= MIN[j]:
            max_sell[j] = (orig[j] - MIN[j]) // PKG[j]

    loss_pkg = [Q[i] * PKG[i] - SELL_GAIN[i] for i in range(3)]

    best_loss = float("inf")
    best_s = None

    def ceildiv(x, y):
        return (x + y - 1) // y

    for s1 in range(max_sell[0] + 1):
        gain1 = s1 * SELL_GAIN[0]
        loss1 = s1 * loss_pkg[0]

        if gain1 >= deficit:
            if loss1 < best_loss:
                best_loss = loss1
                best_s = (s1, 0, 0)
            continue

        for s2 in range(max_sell[1] + 1):
            gain12 = gain1 + s2 * SELL_GAIN[1]
            loss12 = loss1 + s2 * loss_pkg[1]

            if gain12 >= deficit:
                if loss12 < best_loss:
                    best_loss = loss12
                    best_s = (s1, s2, 0)
                continue

            need = deficit - gain12
            s3 = ceildiv(need, SELL_GAIN[2])
            if s3 <= max_sell[2]:
                loss = loss12 + s3 * loss_pkg[2]
                if loss < best_loss:
                    best_loss = loss
                    best_s = (s1, s2, s3)

    if best_s is None:
        return float("-inf"), None

    for j in range(3):
        a[j] -= best_s[j] * PKG[j]

    cash_final = cash - cost + sum(best_s[j] * SELL_GAIN[j] for j in range(3))
    val = Q[0] * a[0] + Q[1] * a[1] + Q[2] * a[2] + cash_final
    trade = tuple(req[i] - best_s[i] for i in range(3))
    return val, trade

@lru_cache(maxsize=None)
def V2(a1, a2, a3, cash):
    """
    Функция Беллмана для 2-го этапа.
    Мемоизация по (a1, a2, a3, cash).
    """
    best_val = float("-inf")
    best_trade = None

    for d1, d2, d3, a1n, a2n, a3n, cashn in gen_trades(a1, a2, a3, cash):
        exp_val = 0.0
        feasible = True

        for prob, m1, m2, m3 in SCEN[1]:
            na1 = apply_coef(a1n, m1)
            na2 = apply_coef(a2n, m2)
            na3 = apply_coef(a3n, m3)

            v3, _ = V3_fast(na1, na2, na3, cashn)
            if v3 == float("-inf"):
                feasible = False
                break
            exp_val += prob * v3

        if feasible and exp_val > best_val:
            best_val = exp_val
            best_trade = (d1, d2, d3)

    return best_val, best_trade

def solve():
    a1, a2, a3, cash = INIT
    best_val = float("-inf")
    best_trade = None

    for d1, d2, d3, a1n, a2n, a3n, cashn in gen_trades(a1, a2, a3, cash):
        exp_val = 0.0
        feasible = True

        for prob, m1, m2, m3 in SCEN[0]:
            na1 = apply_coef(a1n, m1)
            na2 = apply_coef(a2n, m2)
            na3 = apply_coef(a3n, m3)

            v2, _ = V2(na1, na2, na3, cashn)
            if v2 == float("-inf"):
                feasible = False
                break
            exp_val += prob * v2

        if feasible and exp_val > best_val:
            best_val = exp_val
            best_trade = (d1, d2, d3)

    return best_val, best_trade

# ---------- Сбор всех полных ветвей ----------
def best_action(a1, a2, a3, cash, stage):
    """
    Универсальная обёртка: для данного состояния и этапа
    возвращает (лучшее значение, лучшая сделка).
    stage: 0, 1 или 2.
    """
    if stage == 0:
        scenarios = SCEN[0]
        continuation = V2          # продолжение — этапы 2-3
    elif stage == 1:
        scenarios = SCEN[1]
        continuation = V2          # сама V2 — это функция 2-го этапа
        # ВНИМАНИЕ: V2 уже включает перебор SCEN[1] и вызов V3_fast,
        # поэтому для stage=1 просто вызываем V2 напрямую:
        return V2(a1, a2, a3, cash)
    else:  # stage == 2
        return V3_fast(a1, a2, a3, cash)

    best_val = float("-inf")
    best_trade = None

    for d1, d2, d3, a1n, a2n, a3n, cashn in gen_trades(a1, a2, a3, cash):
        exp_val = 0.0
        feasible = True
        for prob, m1, m2, m3 in scenarios:
            na1 = apply_coef(a1n, m1)
            na2 = apply_coef(a2n, m2)
            na3 = apply_coef(a3n, m3)
            cont, _ = continuation(na1, na2, na3, cashn)
            if cont == float("-inf"):
                feasible = False
                break
            exp_val += prob * cont
        if feasible and exp_val > best_val:
            best_val = exp_val
            best_trade = (d1, d2, d3)

    return best_val, best_trade

def all_branches(a1, a2, a3, cash, stage, path, out):
    """Рекурсивно собирает все полные сценарные ветви в out."""
    if stage == 3:
        out.append(list(path))
        return

    if stage == 0:
        _, trade = best_action(a1, a2, a3, cash, stage=0)
    elif stage == 1:
        _, trade = V2(a1, a2, a3, cash)
    else:
        _, trade = V3_fast(a1, a2, a3, cash)

    a1n = a1 + trade[0] * PKG[0]
    a2n = a2 + trade[1] * PKG[1]
    a3n = a3 + trade[2] * PKG[2]
    cashn = delta_cash(trade, cash)

    for prob, m1, m2, m3 in SCEN[stage]:
        na1 = apply_coef(a1n, m1)
        na2 = apply_coef(a2n, m2)
        na3 = apply_coef(a3n, m3)
        path.append({
            "stage": stage,
            "state_before": (a1, a2, a3, cash),
            "trade": trade,
            "scen": (prob, m1, m2, m3),
            "state_after": (na1, na2, na3, cashn),
        })
        all_branches(na1, na2, na3, cashn, stage + 1, path, out)
        path.pop()


def branch_probability(branch):
    p = 1.0
    for step in branch:
        p *= step["scen"][0]
    return p


def branch_terminal(branch):
    a1, a2, a3, cash = branch[-1]["state_after"]
    return a1 + a2 + a3 + cash


def branch_contribution(branch):
    """Вклад ветви в математическое ожидание: p1*p2*p3 * терминальная стоимость."""
    return branch_probability(branch) * branch_terminal(branch)


def print_branch(branch, title):

    print(title)
    for step in branch:
        st = step["stage"]
        prob, m1, m2, m3 = step["scen"]
        a1s, a2s, a3s, cashs = step["state_before"]
        print(f"\nЭтап {st+1}: состояние до сделки "
              f"(ЦБ1={a1s/100:.2f}, ЦБ2={a2s/100:.2f}, "
              f"Деп={a3s/100:.2f}, cash={cashs/100:.2f})")
        print(f"  Сделка: {step['trade']}")
        a1n, a2n, a3n, cashn = step["state_after"]
        print(f"  Сценарий p={prob:.2f} "
              f"(×{m1/100:.2f}, ×{m2/100:.2f}, ×{m3/100:.2f})")
        print(f"  После этапа: ЦБ1={a1n/100:.2f}, ЦБ2={a2n/100:.2f}, "
              f"Деп={a3n/100:.2f}, cash={cashn/100:.2f}")

    a1t, a2t, a3t, casht = branch[-1]["state_after"]
    terminal = a1t + a2t + a3t + casht
    p = branch_probability(branch)
    contrib = p * terminal
    print(f"\nТерминальная стоимость: {terminal/100:.2f} д.е.")
    print(f"Вероятность ветви:      {p:.4f}")
    print(f"Вклад в ожидание:       {contrib/100:.2f} д.е.")



def show_best_branch_by_contribution():
    branches = []
    all_branches(*INIT, 0, [], branches)

    best = max(branches, key=branch_contribution)
    print(f"Всего полных ветвей: {len(branches)}")
    print_branch(best, "\nВетка с наибольшим вкладом в математическое ожидание")

    # Проверка: сумма вкладов всех ветвей должна совпасть с ожиданием
    total = sum(branch_contribution(b) for b in branches) / 100
    print(f"Сумма вкладов всех {len(branches)} ветвей: {total:.2f} д.е.")



def print_all_branches():
    """Печатает все 27 полных сценарных ветвей оптимальной стратегии."""
    labels = ("Б", "Н", "П")
    branches = []
    all_branches(*INIT, 0, [], branches)
    print("\nДЕРЕВО ИСХОДОВ: ВСЕ 27 ПОЛНЫХ ВЕТВЕЙ")
    print("№ | История | Управления по этапам | Состояния после этапов | P(ветви) | Wтерм | Вклад")
    for number, branch in enumerate(branches, 1):
        history, actions, states = [], [], []
        for step in branch:
            stage = step["stage"]
            history.append(labels[SCEN[stage].index(step["scen"])])
            actions.append(str(step["trade"]))
            states.append("(" + ";".join(f"{x/100:.2f}" for x in step["state_after"]) + ")")
        probability = branch_probability(branch)
        terminal = branch_terminal(branch)
        contribution = branch_contribution(branch)
        print(f"{number:02d} | {''.join(history)} | U1={actions[0]}, U2={actions[1]}, U3={actions[2]} | "
              f"X1={states[0]}, X2={states[1]}, X3={states[2]} | "
              f"{probability:.4f} | {terminal/100:.2f} | {contribution/100:.4f}")


def print_full_strategy():
    """Печать всех 13 узлов оптимальной условной стратегии."""
    labels = ("Б", "Н", "П")
    root_value, root_trade = solve()
    print("\nОПТИМАЛЬНАЯ СТРАТЕГИЯ: 13 УЗЛОВ")
    print("Этап | История сценариев | Состояние до сделки (д.е.) | Управление")
    print(f"1 | начало | {tuple(round(x/100,2) for x in INIT)} | {root_trade}")
    for i, sc1 in enumerate(SCEN[0]):
        state2 = tuple(apply_coef(INIT[j] + root_trade[j]*PKG[j], sc1[j+1]) for j in range(3)) + (delta_cash(root_trade, INIT[3]),)
        trade2 = V2(*state2)[1]
        print(f"2 | после {labels[i]} | {tuple(round(x/100,2) for x in state2)} | {trade2}")
        for j, sc2 in enumerate(SCEN[1]):
            state3 = tuple(apply_coef(state2[k] + trade2[k]*PKG[k], sc2[k+1]) for k in range(3)) + (delta_cash(trade2, state2[3]),)
            value3, trade3 = V3_fast(*state3)
            print(f"3 | после {labels[i]}{labels[j]} | {tuple(round(x/100,2) for x in state3)} | {trade3}; V3={value3/100:.5f}")


def run_tests():
    import unittest
    class Tests(unittest.TestCase):
        def test_zero_portfolio(self):
            self.assertEqual(V3_fast(0,0,0,0), (float("-inf"), None))
        def test_zero_market_return(self):
            old_q = Q[:]
            Q[:] = [1.0, 1.0, 1.0]
            val, trade = V3_fast(3000,15000,10000,5000)
            self.assertEqual(trade, (0,0,0))
            self.assertAlmostEqual(val, 33000.0)
            Q[:] = old_q
        def test_multiple_optima(self):
            # Контрольный малый случай с Q=(1.04,1,1): пакет ЦБ1 даёт 26 д.е.
            # ожидаемой стоимости и стоит 26 д.е.; несколько чисел покупаемых пакетов равнозначны.
            q = (1.04, 1.0, 1.0)
            candidates = []
            for d1,d2,d3,b1,b2,b3,c in gen_trades(10000,80000,40000,10000):
                candidates.append((q[0]*b1 + q[1]*b2 + q[2]*b3 + c, (d1,d2,d3)))
            optimum = max(v for v,_ in candidates)
            ties = [u for v,u in candidates if abs(v-optimum)<1e-9]
            self.assertIn((0,0,0), ties)
            self.assertIn((1,0,0), ties)
            self.assertGreaterEqual(len(ties), 2)
        def test_all_budget_one_asset(self):
            # Проверяется фактическое решение всей задачи, а не только арифметика покупки.
            _, trade = solve()
            self.assertEqual(trade, (23,0,0))
            self.assertEqual(delta_cash(trade, INIT[3]), 200)
        def test_invalid_data(self):
            with self.assertRaises(ValueError): validate_input(init=(-1,0,0,0))
            with self.assertRaises(ValueError): validate_input(scenarios=[[(.2,100,100,100)]*3]*2)
            with self.assertRaises(ValueError): validate_input(scenarios=[[(.2,100,100)]*3, SCEN[1], SCEN[2]])
        def test_independent_full_three_stage_enumeration(self):
            """Полный трёхэтапный контрольный пример со своим генератором сделок."""
            import itertools
            import math
            from functools import lru_cache

            minimum = (1.0, 1.0, 1.0)
            package = (1.0, 1.0, 1.0)
            buy_cost = (1.1, 1.1, 1.1)
            sell_gain = (0.9, 0.9, 0.9)
            initial = (2.0, 2.0, 1.0, 1.5)
            scenarios = [
                [(0.5, 90, 100, 100), (0.5, 100, 100, 100)],
                [(0.5, 150, 100, 100), (0.5, 100, 100, 100)],
                [(0.5, 130, 100, 100), (0.5, 100, 100, 100)],
            ]

            # Независимая генерация: перебираются целые векторы сделок,
            # затем напрямую проверяются минимумы активов и остаток денег.
            def independent_deals(state):
                assets, cash = state[:3], state[3]
                max_sales = [int(math.floor((assets[i] - minimum[i]) / package[i] + 1e-9))
                             for i in range(3)]
                possible_cash = cash + sum(max_sales[i] * sell_gain[i] for i in range(3))
                max_buys = int(math.floor(possible_cash / min(buy_cost) + 1e-9))
                ranges = [range(-max_sales[i], max_buys + 1) for i in range(3)]
                for deals in itertools.product(*ranges):
                    new_assets = tuple(assets[i] + deals[i] * package[i] for i in range(3))
                    new_cash = cash
                    for i, count in enumerate(deals):
                        if count > 0:
                            new_cash -= count * buy_cost[i]
                        elif count < 0:
                            new_cash += (-count) * sell_gain[i]
                    if new_cash >= -1e-9 and all(new_assets[i] >= minimum[i] - 1e-9
                                                 for i in range(3)):
                        yield deals, new_assets, max(0.0, new_cash)

            def market(assets, coeffs):
                return tuple(round(assets[i] * coeffs[i] / 100.0, 2) for i in range(3))

            @lru_cache(maxsize=None)
            def memo_recursion(stage, state):
                if stage == 3:
                    return sum(state), None
                best_value, best_deal = float("-inf"), None
                for deals, assets, cash in independent_deals(state):
                    expected = 0.0
                    for probability, k1, k2, k3 in scenarios[stage]:
                        next_state = market(assets, (k1, k2, k3)) + (cash,)
                        expected += probability * memo_recursion(stage + 1, next_state)[0]
                    if expected > best_value + 1e-10:
                        best_value, best_deal = expected, deals
                return best_value, best_deal

            # Контрольный полный перебор без мемоизации.
            def exhaustive_recursion(stage, state):
                if stage == 3:
                    return sum(state), None
                best_value, best_deal = float("-inf"), None
                for deals, assets, cash in independent_deals(state):
                    expected = 0.0
                    for probability, k1, k2, k3 in scenarios[stage]:
                        next_state = market(assets, (k1, k2, k3)) + (cash,)
                        expected += probability * exhaustive_recursion(stage + 1, next_state)[0]
                    if expected > best_value + 1e-10:
                        best_value, best_deal = expected, deals
                return best_value, best_deal

            recursive_value, recursive_deal = memo_recursion(0, initial)
            brute_value, brute_deal = exhaustive_recursion(0, initial)
            self.assertAlmostEqual(recursive_value, brute_value, places=9)
            self.assertEqual(recursive_deal, brute_deal)
            self.assertAlmostEqual(recursive_value, 7.80625, places=8)
            self.assertEqual(recursive_deal, (0, -1, 0))

            # Дополнительно сравниваем значения и управления в каждом достижимом узле.
            def apply_deal(state, deal):
                assets = tuple(state[i] + deal[i] * package[i] for i in range(3))
                cash = state[3]
                for i, count in enumerate(deal):
                    if count > 0:
                        cash -= count * buy_cost[i]
                    elif count < 0:
                        cash += (-count) * sell_gain[i]
                return assets + (max(0.0, cash),)

            first_state = apply_deal(initial, recursive_deal)
            for _, k1, k2, k3 in scenarios[0]:
                state2 = market(first_state[:3], (k1, k2, k3)) + (first_state[3],)
                self.assertEqual(memo_recursion(1, state2), exhaustive_recursion(1, state2))
                deal2 = memo_recursion(1, state2)[1]
                second_state = apply_deal(state2, deal2)
                for _, j1, j2, j3 in scenarios[1]:
                    state3 = market(second_state[:3], (j1, j2, j3)) + (second_state[3],)
                    self.assertEqual(memo_recursion(2, state3), exhaustive_recursion(2, state3))

        def test_full_enumeration_small_stage3(self):
            # Независимый полный перебор для малого состояния; сравниваем с аналитическим V3_fast.
            state = (2500,15000,10000,10000)
            fast, _ = V3_fast(*state)
            best = float("-inf")
            for d1,d2,d3,b1,b2,b3,c in gen_trades(*state):
                exp = 0.0
                for p,k1,k2,k3 in SCEN[2]:
                    exp += p*(apply_coef(b1,k1)+apply_coef(b2,k2)+apply_coef(b3,k3)+c)
                best = max(best, exp)
            self.assertAlmostEqual(fast, best, delta=0.03)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return result.wasSuccessful()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        raise SystemExit(0 if run_tests() else 1)

    # Проверка исходных данных выполняется и при обычном запуске.
    validate_input()
    check_v3_assumption()

    # resource доступен в Unix-подобных системах; Windows использует tracemalloc.
    try:
        import resource
    except ImportError:
        resource = None
        import tracemalloc
        tracemalloc.start()

    t0 = time.perf_counter()
    best_val, best_trade = solve()
    elapsed = time.perf_counter() - t0

    print(f"Максимальная ожидаемая конечная стоимость: {best_val/100:.2f} д.е.")
    print(f"Ожидаемый доход относительно начальной стоимости: {(best_val - INITIAL_TOTAL)/100:.2f} д.е.")
    print(f"Начальная стоимость портфеля: {INITIAL_TOTAL/100:.2f} д.е.")
    print(f"Оптимальная сделка на 1-м этапе: {best_trade}")
    print(f"Количество вычисленных состояний V2: {V2.cache_info().currsize}")
    print(f"Время работы: {elapsed:.4f} с")

    show_best_branch_by_contribution()
    print_full_strategy()
    print_all_branches()
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import platform
        # Linux сообщает ru_maxrss в КиБ, macOS — в байтах.
        peak_mib = peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024
        print(f"Пиковая память процесса (resource.ru_maxrss): {peak_mib:.2f} МиБ")
    else:
        _, peak = tracemalloc.get_traced_memory()
        print(f"Пиковая память Python (tracemalloc): {peak / (1024 * 1024):.2f} МиБ")
        tracemalloc.stop()