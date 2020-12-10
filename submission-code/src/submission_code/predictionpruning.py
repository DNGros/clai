from functools import reduce
from typing import List, Union
from scipy.special import softmax
from pprint import pprint
import innereval.utils.metric_utils
from datacleaning import normalize_nl
from innereval.evaluate import compute_metric_cache, make_preparse_cached

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
    predictions = prune_duplicates(predictions, max_cnt=max_cnt)
    #return _prune_duplicate_strs(predictions, max_cnt=max_cnt)
    #predictions = prune_optimized(_prune_duplicates(predictions, len(predictions)), max_cnt=max_cnt)
    #return predictions
    #return predictions
    #predictions = recalibrate(predictions)
    predictions = new_recalibrate(predictions)
    return pad_predictions(predictions)


def new_recalibrate(predictions: List[Prediction]) -> List[Prediction]:
    scores = [pred.score for pred in predictions]
    score_sum = sum(scores)
    score_max = max(scores)
    #prob_any_is_positive = expit(score_max * 3.22 + score_sum * 0.2 - 1.93)
    all_util_count = [len(make_preparse_cached(pred.cmd).utility_names) for pred in predictions]
    all_flag_counts = [
        sum(len(flag_set) for flag_set in make_preparse_cached(pred.cmd).flag_nodes)
        for pred in predictions
    ]
    all_return_words = [set(normalize_nl(pred.debug_ref_nl).split()) for pred in predictions]
    ex_words = [
        len(set(all_return_words))
        for all_return_words in all_return_words
    ]
    prob_indv_is_positive = [
        expit(score*4.28+util_count*(-.81)+flag_count*(0.087)+word_count*(0.02)+(-2.82)) if not pred.is_pad else 0
        for score, util_count, flag_count, word_count, pred in
        zip(scores, all_util_count, all_flag_counts, ex_words, predictions)
    ]
    #print("prob possitive", prob_indv_is_positive)
    prob_indv_is_neg = [
        (1-prob_pos) if prob_pos < 0.15 else 0
        for prob_pos in prob_indv_is_positive
    ]
    prob_all_neg = reduce(lambda x, y: x * y, [1-p for p in prob_indv_is_positive])
    new_confs = [
        1 - (prob_neg * prob_all_neg)
        for prob_neg in prob_indv_is_neg
    ]
    #print("new cofs", new_confs, prob_all_neg)
    return [
        Prediction(
            pred.cmd,
            score=pred.score,
            eval_prob=new_conf,
            debug=pred.debug,
            debug_ref_nl=pred.debug_ref_nl,
        )
        for pred, new_conf in zip(predictions, new_confs)
    ]
    pass


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
            debug_ref_nl=pred.debug_ref_nl,
        )
        for pred, new_conf in zip(predictions, new_confs)
    ]


def pad_predictions(predictions, max_cnt=5):
    if len(predictions) < max_cnt:
        return predictions + (
                [Prediction("find pad", 0.0, 0.005, "pad", is_pad=True)] * (max_cnt - len(predictions)))
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
            debug_ref_nl=predictions[pick_i].debug_ref_nl
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


def prune_duplicates(predictions: List[Union[Prediction, str]], max_cnt=5):
    def extract(v: Union[str, Prediction]) -> str:
        if isinstance(v, str):
            return v
        return v.cmd
    out = []
    for pred in predictions:
        havent_seen_before = all(
            extract(o) != extract(pred) and compute_metric_cache(extract(pred), 1, extract(o)) < 1
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
