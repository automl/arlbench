from smac.intensifier.successive_halving import SuccessiveHalving
import numpy as np

total_budget = 10_000_000
min_budget = 100000
max_budget = 1000000  # R
eta = 2

_s_max = SuccessiveHalving._get_max_iterations(eta, max_budget, min_budget)

_max_iterations: dict[int, int] = {}
_n_configs_in_stage: dict[int, list] = {}
_budgets_in_stage: dict[int, list] = {}

for i in range(_s_max + 1):
    max_iter = _s_max - i

    _budgets_in_stage[i], _n_configs_in_stage[i] = SuccessiveHalving._compute_configs_and_budgets_for_stages(
        eta, max_budget, max_iter, _s_max
    )
    _max_iterations[i] = max_iter + 1


total_trials = np.sum([np.sum(v) for v in _n_configs_in_stage.values()])
total_budget = np.sum([np.sum(v) for v in _budgets_in_stage.values()])

print("n_brackets", _s_max)
print("budgets per stage", _budgets_in_stage)
print("n configs per stage", _n_configs_in_stage)
print("total number of trials", total_trials)
print("total budget",  total_budget)

