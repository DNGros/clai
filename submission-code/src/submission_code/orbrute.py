from typing import Tuple
import time

import numpy as np
from numba import jit


@jit(nopython=True)
def find_best_combo(costs):
    best_cost = -999999.9
    total_options = len(costs)
    num_vars = 5
    best_choice = list(range(num_vars))
    best_confidences = [1.0] * 5
    p0, p1, p2, p3, p4 = 1.0, 1.0, 1.0, 1.0, 1.0
    # Suuuper ugly nested loop. But gotta have that speeed (idk, numba/llvm probably figure out
    #   a recursive version or an iterator version, but not sure. This is
    #   definately over optimization)
    for i0 in range(min(3, total_options - num_vars + 1)):
        for i1 in range(i0 + 1, min(8, total_options - num_vars + 2)):
            for i2 in range(i1 + 1, min(10, total_options - num_vars + 3)):
                for i3 in range(i2 + 1, min(13, total_options - num_vars + 4)):
                    for i4 in range(i3 + 1, min(16, total_options - num_vars + 5)):
                        for p2 in (0.0, 1.0):
                            for p3 in (0.0, 1.0):
                                for p4 in (0.0, 1.0):
                                    # all the indents (this is disgusting...)
                                    expected_value = 0.0
                                    for gt in range(total_options):
                                        max_cost = max(p0*costs[gt, i0], p1*costs[gt, i1],
                                                       p2*costs[gt, i2], p3*costs[gt, i3],
                                                       p4*costs[gt, i4])
                                        if max_cost >= 0:
                                            expected_value += max_cost
                                        else:
                                            expected_value += (
                                                costs[gt, i0] * p0
                                                + costs[gt, i1] * p1
                                                + costs[gt, i2] * p2
                                                + costs[gt, i3] * p3
                                                + costs[gt, i4] * p4
                                                              ) / num_vars
                                    if expected_value > best_cost:
                                        best_choice = [i0, i1, i2, i3, i4]
                                        best_confidences = [p0, p1, p2, p3, p4]
                                        best_cost = expected_value
    return best_choice, best_confidences, best_cost


find_best_combo(np.array([
    [1.0, 0.6, 0.0, 0.0, 0.0],
    [0.9, 1.0, 0.2, 0.0, 0.0],
    [0.0, 0.2, 1.0, 0.0, 0.0],
    [0.0, 0.2, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]))



def main():
    #costs = np.array([
    #    [1.0, 0.6, 0.0, 0.0, 0.0],
    #    [0.9, 1.0, 0.2, 0.0, 0.0],
    #    [0.0, 0.2, 1.0, 0.0, 0.0],
    #    [0.0, 0.2, 0.0, 1.0, 0.0],
    #    [0.0, 0.0, 0.0, 0.0, 1.0],
    #])  # an i, j matrix. If item i is the gt, then the score if j is chosen
    #probs = np.array([
    #    0.3,
    #    0.6,
    #    0.08,
    #    0,
    #    0.02
    #])
    size = 20
    r = np.random.rand(size, size)
    e = np.eye(size)
    costs = np.maximum(r, e)
    print("costs", costs)
    probs = np.random.rand(size)
    costs = costs * probs[:, None]
    print(costs)

    start_time = time.time()
    best_option, expectedval = find_best_combo(costs)
    print(time.time() - start_time)


if __name__ == '__main__':
    main()
