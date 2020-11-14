from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.absolute()))
sys.path.append(str(Path(__file__).parent.absolute() / "lib"))
print(sys.path)

from predictionpruning import prune_predictions
from irtoy import load_model

model = load_model()


def predict_one(nl, model, result_cnt=5):
    predictions = prune_predictions(model.predict(nl, n=result_cnt * 2), max_cnt=result_cnt)
    cmds = [
        pred.cmd for pred in predictions
    ]
    confs = [1.0] * result_cnt
    return cmds, confs


def predict(invocations, result_cnt=5):
    """ 
    Function called by the evaluation script to interface the participants model
    `predict` function accepts the natural language invocations as input, and returns
    the predicted commands along with confidences as output. For each invocation, 
    `result_cnt` number of predicted commands are expected to be returned.
    
    Args:
        1. invocations : `list (str)` : list of `n_batch` (default 16) natural language invocations
        2. result_cnt : `int` : number of predicted commands to return for each invocation

    Returns:
        1. commands : `list [ list (str) ]` : a list of list of strings of shape (n_batch, result_cnt)
        2. confidences: `list[ list (float) ]` : confidences corresponding to the predicted commands
                                                 confidence values should be between 0.0 and 1.0. 
                                                 Shape: (n_batch, result_cnt)
    """

    n_batch = len(invocations)
    
    # `commands` and `confidences` have shape (n_batch, result_cnt)
    #commands = [
    #    [''] * result_cnt
    #    for _ in range(n_batch)
    #]
    #confidences = [
    #    [1.0] * result_cnt
    #    for _ in range(n_batch)
    #]

    ################################################################################################
    #     Participants should add their codes to fill predict `commands` and `confidences` here    #
    ################################################################################################

    commands = []
    confidences = []
    for nl in invocations:
        cmds, confs = predict_one(nl, model, result_cnt=result_cnt)
        commands.append(cmds)
        confidences.append(confs)

    ################################################################################################
    #                               Participant code block ends                                    #
    ################################################################################################

    return commands, confidences
