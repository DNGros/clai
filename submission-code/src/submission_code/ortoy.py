from ortools.linear_solver import pywraplp
import itertools
import numpy as np
#from ortools.sat.python import cp_model
from ortools.sat.python import cp_model
import time


def main():
    #solver = pywraplp.Solver.CreatSolver("SCIP")
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    scale = 1000
    scale_sq = scale * scale

    costs = np.array([
        [1.0, 0.6, 0.0],
        [0.9, 1.0, 0.2],
        [0.0, 0.2, 1.0]
    ])  # an i, j matrix. If item i is the gt, then the score if j is chosen
    size = 25
    r = np.random.rand(size, size)
    e = np.eye(size)
    costs = np.maximum(r, e)
    print("costs", costs)
    costs = (scale * costs).astype(int)
    probs = np.array([
        0.3,
        0.6,
        0.1
    ])
    probs = np.random.rand(size)
    probs /= sum(probs)
    print("probs", probs)
    probs = (scale * probs).astype(int)
    start_time = time.time()
    picks = 5
    vars = []
    for i in range(len(costs)):
        vars.append(model.NewBoolVar(f"choose{i}"))
    model.Add(sum(vars) == picks)
    val_if_gt = []
    for i, prob in enumerate(probs):
        max_var = model.NewIntVar(0, scale * scale, f"maxv{i}")
        scores = [model.NewIntVar(0, scale * scale, "score{j}") for j in range(len(probs))]
        for j, score in enumerate(scores):
            model.Add(score == (costs[i, j]) * vars[i])
        #scores = [4 * vars[i] for j in range(len(probs))]
        model.AddMaxEquality(max_var, scores)
        val_if_gt.append(prob * max_var)
    model.Maximize(
        sum(val_if_gt)
    )

    #solver.parameters.max_time_in_seconds = 0.1
    solver.Solve(model)
    status = solver.Solve(model)
    print("TIME", time.time() - start_time)

    #if status == cp_model.OPTIMAL:
    print("optimal", status == cp_model.OPTIMAL)
    print("infeasible", status == cp_model.INFEASIBLE)
    print("feasible", status == cp_model.FEASIBLE)
    print(f"Maximum of objective function: {(solver.ObjectiveValue() / scale_sq)}")
    print()
    for var in vars:
        print(var, solver.Value(var))

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


if __name__ == "__main__":
    main()