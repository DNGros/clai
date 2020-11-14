from typing import List
import innereval.utils.metric_utils

from modeling import Prediction
import numpy as np


def compute_metric(pred: str, gt: str) -> float:
    if pred == gt:
        return 1
    return innereval.utils.metric_utils.compute_metric(
        predicted_cmd=pred,
        predicted_confidence=1.0,
        ground_truth_cmd=gt,
        metric_params={"u1": 1, "u2": 1}
    )


def compute_metric_grid(preds: List[str]) -> np.ndarray:
    grid = np.ndarray((len(preds), len(preds)))
    for pi, p in enumerate(preds):
        for gi, gt in enumerate(preds):
            grid[pi, gi] = compute_metric(p, gt)
    return grid


def prune_predictions(predictions: List[Prediction], max_cnt=5) -> List[Prediction]:
    predictions.sort(key=lambda pred: pred.prob, reverse=True)
    return _prune_duplicates(predictions, max_cnt=max_cnt)
    #return _prune_duplicate_strs(predictions, max_cnt=max_cnt)


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
        havent_seen_before = all(o.cmd != pred.cmd and compute_metric(pred.cmd, o.cmd) < 1 for o in out)
        if havent_seen_before:
            out.append(pred)
        if len(out) == max_cnt:
            return out
    return out


if __name__ == "__main__":
    print(compute_metric("ls -la", "ls -S"))
    print(compute_metric("ls -S", "ls -la"))
    print(compute_metric_grid(["ls -la", "wc -l", "ls -Sa", "cat foo.txt | wc -l"]))
