from typing import List
import innereval.utils.metric_utils

from modeling import Prediction

def compute_metric(pred, gt):
    return innereval.utils.metric_utils.compute_metric(
        predicted_cmd=pred,
        predicted_confidence=1.0,
        ground_truth_cmd=gt,
        metric_params={"u1": 1, "u2": 1}
    )


def prune_predictions(predictions: List[Prediction], max_cnt=5):
    predictions.sort(key=lambda pred: pred.prob, reverse=True)
    return _prune_duplicates(predictions, max_cnt=max_cnt)


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
    print(compute_metric("ls -l", "ls -S"))
    print(compute_metric("ls -lS", "ls -S"))
