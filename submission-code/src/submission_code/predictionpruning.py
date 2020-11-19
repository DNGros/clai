from typing import List
from scipy.special import softmax
from pprint import pprint
import innereval.utils.metric_utils
from innereval.evaluate import compute_metric_cache

from modeling import Prediction
import numpy as np
from scipy.special import expit

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
    predictions = _prune_duplicates(predictions, max_cnt=max_cnt)
    #return _prune_duplicate_strs(predictions, max_cnt=max_cnt)
    #predictions = prune_optimized(_prune_duplicates(predictions, len(predictions)), max_cnt=max_cnt)
    #return predictions
    #return predictions
    #predictions = recalibrate(predictions)
    return pad_predictions(predictions)


def recalibrate(predictions: List[Prediction]) -> List[Prediction]:
    scores = [pred.score for pred in predictions]
    score_sum = sum(scores)
    score_max = max(scores)
    prob_any_is_positive = expit(score_max * 3.22 + score_sum * 0.2 - 1.93)
    prob_indv_is_positive = [
        expit(score*3.88-3.34)
        for score in scores
    ]
    prob_indv_is_neg = [
        (1-prob_pos) if prob_pos < 0.1 else 0  # Square for extra benifit of the doubt
        for prob_pos in prob_indv_is_positive
    ]
    prob_all_neg = (1 - prob_any_is_positive)
    if prob_all_neg < 0.9:
        prob_all_neg = 0
    new_confs = [
        1 - (prob_neg * prob_all_neg)
        for prob_neg in prob_indv_is_neg
    ]
    #print("scores", scores, "new_congs", new_confs)
    return [
        Prediction(
            pred.cmd,
            score=pred.score,
            eval_prob=new_conf,
            debug=pred.debug,
        )
        for pred, new_conf in zip(predictions, new_confs)
    ]



def pad_predictions(predictions, max_cnt=5):
    if len(predictions) < max_cnt:
        return predictions + ([Prediction("find pad", 0.0, 0.005, "pad")] * (max_cnt - len(predictions)))
    return predictions


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
    scale_prob = pred_scores**5
    probs = scale_prob / scale_prob.sum()
    #print("norm prob", probs)
    grid_withprob = grid * probs[:, None]

    other_prob = (1 - max(pred_scores))**2

    optimal_picks, optimal_confs, expected_val = find_best_combo(grid_withprob, other_prob)
    if optimal_picks is None:
        print("FAIL TO OPTIMIZE")
        return predictions[:min(max_cnt, len(predictions))]
    #print(optimal_picks, optimal_confs, expected_val)
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
    a = compute_metric_grid([
        "ls -l",
        "ls -a",
        "ls -la",
        "ls -S",
        "ls -Sa",
        "find .",
        "find . -type f"
    ])
