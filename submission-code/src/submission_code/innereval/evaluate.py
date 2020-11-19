import argparse
import attr
import numpy as np
import time
import os
import json
from datetime import datetime
import tempfile
import traceback

from innereval.utils.metric_utils import compute_metric, get_utility_nodes, get_utility_flags, pad_arrays
from innereval.utils.dataset import Nlc2CmdDS
from innereval.utils.dataloaders import Nlc2CmdDL

from typing import List, Set

from bashlint import data_tools, nast
from bashlint.data_tools import bash_parser


def get_score(prediction_scores):
    score = -1.0
    if len(prediction_scores) == 0:
        return score

    has_positive_score = True in [x > 0 for x in prediction_scores]

    if has_positive_score:
        score = max(prediction_scores)
    else:
        score = sum(prediction_scores) / float(len(prediction_scores))

    return score


@attr.s(auto_attribs=True)
class Preparse:
    utility_names: List[str]
    flag_nodes: List[Set[str]]


def make_preparse(string: str) -> Preparse:
    ast = bash_parser(string)

    def get_node_value(node):
        if isinstance(node, nast.Node):
            return node.value.lower()
        return None

    utilites = get_utility_nodes(ast)
    flags = [
        set([node.value for node in get_utility_flags(util)])
        for util in utilites
    ]
    util_names = [get_node_value(n) for n in utilites]
    return Preparse(util_names, flags)


def flag_score_preparse(gt_flags, pred_flags):
    if len(gt_flags) == len(gt_flags) == 0:
        # return a score of 1.0 when there are no flags to predict
        return 1.0
    intersection_len = len(gt_flags.intersection(pred_flags))
    union_len = len(gt_flags.union(pred_flags))
    Z = max(1, len(pred_flags), len(gt_flags))
    score = (2 * intersection_len - union_len) / float(Z)
    return score


def score_preparses(
    pred: Preparse,
    predicted_confidence: float,
    gt: Preparse,
    u1: float = 1.0,
    u2: float = 1.0,
):
    ground_truth_utilities, predicted_utilities = pad_arrays(
        gt.utility_names, pred.utility_names)
    ground_truth_flagnames, predicted_flagnames = pad_arrays(
        gt.flag_nodes, pred.flag_nodes)

    score = []
    for ground_truth_utility, predicted_utility, gt_flags, pred_flags in zip(
            ground_truth_utilities, predicted_utilities, ground_truth_flagnames,
            predicted_flagnames):
        utility_score = float(ground_truth_utility == predicted_utility)
        flag_score = flag_score_preparse(gt_flags or set(), pred_flags or set())

        flag_score_normed = (u1 + u2 * flag_score) / (u1 + u2)
        prediction_score = predicted_confidence * (
                (utility_score * flag_score_normed) -
                (1 - utility_score)
        )
        score.append(prediction_score)

    score_mean = 0.0 if len(score) == 0 else np.mean(score)
    return score_mean


class PreparseCache:
    _cache = {}

    def __init__(self, max_size: int = 1e6):
        self._max_size = max_size

    def parse(self, string: str) -> Preparse:
        cache_result = self._cache.get(string, None)
        if cache_result is not None:
            return cache_result
        val = make_preparse(string)
        if len(self._cache) < self._max_size:
            self._cache[string] = val
        return val


parse_cache = PreparseCache()


def compute_metric_cache(predicted_cmd: str, predicted_confidence: float, ground_truth_cmd) -> float:
    return score_preparses(
        parse_cache.parse(predicted_cmd),
        predicted_confidence,
        parse_cache.parse(ground_truth_cmd)
    )


if __name__ == '__main__':
    print(make_preparse("find . -type f -exec chmod +x;"))
    pass


