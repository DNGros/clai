import argparse
import time
import os
import json
from datetime import datetime
import tempfile
import traceback

from innereval.utils.metric_utils import compute_metric
from innereval.utils.dataset import Nlc2CmdDS
from innereval.utils.dataloaders import Nlc2CmdDL


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


