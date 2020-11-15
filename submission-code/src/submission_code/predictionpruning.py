from typing import List
from scipy.special import softmax
from pprint import pprint
import innereval.utils.metric_utils
from innereval.evaluate import compute_metric_cache

from modeling import Prediction
import numpy as np

#from ortoy import optimize_selections
from orbrute import find_best_combo


def compute_metric_grid(preds: List[str]) -> np.ndarray:
    grid = np.ndarray((len(preds), len(preds)))
    for pi, p in enumerate(preds):
        for gi, gt in enumerate(preds[pi:], pi):
            s = compute_metric_cache(p, 1.0, gt)
            grid[pi, gi] = s
            grid[gi, pi] = s
    return grid


def prune_predictions(predictions: List[Prediction], max_cnt=5) -> List[Prediction]:
    predictions.sort(key=lambda pred: pred.score, reverse=True)
    #return _prune_duplicates(predictions, max_cnt=max_cnt)
    #return _prune_duplicate_strs(predictions, max_cnt=max_cnt)
    predictions = prune_optimized(_prune_duplicates(predictions, len(predictions)), max_cnt=max_cnt)
    return pad_predictions(predictions)


def pad_predictions(predictions, max_cnt=5):
    if len(predictions) < max_cnt:
        return predictions + ([Prediction("padcmd", 0.0, 0.0, "pad")] * (len(predictions) - max_cnt))
    return predictions
    pass


def prune_optimized(predictions: List[Prediction], max_cnt=5) -> List[Prediction]:
    #print(f"Predictions:")
    #print("\n".join([f"{i}: {pred.dump_str()}" for i, pred in enumerate(predictions)]))
    if len(predictions) <= max_cnt:
        return predictions
    grid = compute_metric_grid([pred.cmd for pred in predictions])
    pred_scores = np.array([pred.score for pred in predictions])
    #print(grid)
    #print(list(probs))
    #probs /= probs.sum()
    scale_prob = pred_scores**4
    probs = scale_prob / scale_prob.sum()
    #print("norm prob", probs)
    grid_withprob = grid * probs[:, None]

    other_prob = (1 - max(pred_scores))**2

    optimal_picks, optimal_confs, expected_val = find_best_combo(grid_withprob, other_prob)
    if optimal_picks is None:
        print("FAIL TO OPTIMIZE")
        return predictions[:min(max_cnt, len(predictions))]
    print(optimal_picks, optimal_confs, expected_val)
    return [
        #predictions[i]
        Prediction(
            predictions[pick_i].cmd,
            score=predictions[pick_i].score,
            eval_prob=conf,
            debug=predictions[pick_i].debug,
        )
        for pick_i, conf in zip(optimal_picks, optimal_confs)
    ]


def _prune_duplicate_strs(predictions: List[Prediction], max_cnt=5) -> List[Prediction]:
    seen = set()
    out = []
    for pred in predictions:
        if pred.cmd not in seen:
            out.append(pred)
        if len(out) == max_cnt:
            return out
    return out


def _prune_duplicates(predictions: List[Prediction], max_cnt=5):
    out = []
    for pred in predictions:
        havent_seen_before = all(
            o.cmd != pred.cmd and compute_metric_cache(pred.cmd, 1, o.cmd) < 1
            for o in out
        )
        if havent_seen_before:
            out.append(pred)
        if len(out) == max_cnt:
            return out
    return out


if __name__ == "__main__":
    print(compute_metric_cache("ls -la", 1.0, "ls -S"))
    print(compute_metric_cache("ls -S", 1.0, "ls -la"))
    print(compute_metric_grid(["ls -la", "wc -l", "ls -Sa", "cat foo.txt | wc -l"]))
