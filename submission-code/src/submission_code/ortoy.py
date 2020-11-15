from typing import Set, Optional, Tuple, List

from ortools.linear_solver import pywraplp
import itertools
import numpy as np
#from ortools.sat.python import cp_model
from ortools.sat.python import cp_model
import time


def optimize_selections(costs, probs, picks: int) -> Tuple[Optional[List[int]], float, Optional[float]]:
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    scale = 100
    conf_range = 20
    scale_sq = scale * scale * conf_range

    size = len(costs)
    costs = (scale * costs).astype(int)
    probs /= sum(probs)
    probs = (scale * probs).astype(int)
    start_time = time.time()
    is_chosen_vars = []
    confidence_vars = []
    for i in range(len(costs)):
        is_chosen_vars.append(model.NewBoolVar(f"choose{i}"))
        confidence_vars.append(model.NewIntVar(0, conf_range, f"confidence{i}"))
    model.Add(sum(is_chosen_vars) == picks)
    val_if_gt = []
    for i, conf_var in enumerate(confidence_vars):
        model.Add(conf_var == 0).OnlyEnforceIf(is_chosen_vars[i].Not())
    for i, prob in enumerate(probs):
        max_var = model.NewIntVar(-scale_sq, scale_sq, f"maxv{i}")
        scores = [model.NewIntVar(-scale_sq, scale_sq, "score{j}") for j in range(len(probs))]
        for j, score in enumerate(scores):
            model.Add(score == (costs[i, j]) * confidence_vars[i])
        model.AddMaxEquality(max_var, scores)
        is_positive = model.NewBoolVar(f'is_positive{i}')
        model.Add(max_var >= 0).OnlyEnforceIf(is_positive)
        model.Add(max_var < 0).OnlyEnforceIf(is_positive.Not())
        agg_score = model.NewIntVar(-scale_sq, scale_sq, f"aggscore{i}")
        model.Add(agg_score == agg_score).OnlyEnforceIf(is_positive)
        sum_var = model.NewIntVar(-scale_sq*len(scores), scale_sq*len(scores), f"maxv{i}")
        model.Add(sum_var == sum(scores)).OnlyEnforceIf(is_positive.Not())
        model.AddDivisionEquality(agg_score, sum_var, len(scores)).OnlyEnforceIf(is_positive.Not())
        #model.Add(agg_score == sum_var / len(scores)).OnlyEnforceIf(is_positive.Not())
        val_if_gt.append(prob * agg_score)
    model.Maximize(
        sum(val_if_gt)
    )

    #solver.parameters.max_time_in_seconds = 0.1
    solver.Solve(model)
    status = solver.Solve(model)
    #print("TIME", time.time() - start_time)

    #if status == cp_model.OPTIMAL:
    #print("optimal", status == cp_model.OPTIMAL)
    #print("infeasible", status == cp_model.INFEASIBLE)
    #print("feasible", status == cp_model.FEASIBLE)
    expected_score = (solver.ObjectiveValue() / scale_sq)
    #print(f"Maximum of objective function: {expected_score}")
    #print()
    if status == cp_model.INFEASIBLE:
        return None, None, None
    else:
        outs = []
        out_confs = []
        for i_v, var in enumerate(is_chosen_vars):
            print(var)
            val = solver.Value(var)
            if val:
                outs.add(i_v)
                out_confs.add(solver.Value(confidence_vars[i_v]) / conf_range)
        return outs, out_confs, expected_score


def brute_force_explore():
    combos = list(itertools.combinations(range(size), 5))

    print(len(combos))
    a = np.ones((len(combos), size)).astype(bool)
    #print(a.shape)
    #print(costs.shape)
    #print((a[:, :, None] * costs[None, :, :]).shape)
    start_time = time.time()
    M3 = np.max(a[:, :, None] * costs[None, :, :], axis=1)
    M3 = M3.sum(axis=1)
    a = np.max(M3)
    print("TIME", time.time() - start_time)
    #print((M3).shape)


def main():
    costs = np.array([
        [1.0, 0.6, 0.0],
        [0.9, 1.0, 0.2],
        [0.0, 0.2, 1.0]
    ])  # an i, j matrix. If item i is the gt, then the score if j is chosen
    probs = np.array([
        0.3,
        0.6,
        0.1
    ])
    #size = 25
    #r = np.random.rand(size, size)
    #e = np.eye(size)
    #costs = np.maximum(r, e)
    #print("costs", costs)
    #probs = np.random.rand(size)
    picks = 1
    optimize_selections(costs, probs, picks)


if __name__ == "__main__":
    main()