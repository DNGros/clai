import numpy as np

from ortoy import optimize_selections


def test_simple():
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
    picks = 1
    selections, confs, expected_value = optimize_selections(costs, probs, picks)
    assert selections == [1]
    print(confs)
    assert expected_value == .6
