from innereval import evaluate
from innereval.evaluate import compute_metric_cache
from irtoy import build_model
from tqdm import tqdm
from main import predict_one
from pred_suc_toy import run_toy_class
from readdata import ACDataset
from scipy import stats

pos_highest = []
neg_highests = []
pos_sums = []
neg_sums = []
all_pred_scores = []
all_eval_scores = []

def predict_subset(training: ACDataset, test: ACDataset, print_exs: bool = True):
    model = build_model(training)
    all_cmds, all_cons = [], []

    for example in tqdm(test.examples, mininterval=10):
        if print_exs:
            print('-'*50)
            print("Example:")
            print(f"NL: {example.nl}")
            print(f"GT: {example.cmd}")
        cmds, confs, predictions = predict_one(example.nl, model, 5)
        all_cmds.append(cmds)
        all_cons.append(confs)
        if print_exs:
            print("Predictions", cmds)
            scores = [compute_metric_cache(cmd, conf, example.cmd) for cmd, conf in zip(cmds, confs)]
            print('Scores:', scores)
            print('Score:', evaluate.get_score(scores))
        ###
        scores = [compute_metric_cache(cmd, conf, example.cmd) for cmd, conf in zip(cmds, confs)]
        fin_score = evaluate.get_score(scores)
        highest_pred_score = max(pred.score for pred in predictions)
        score_sum = sum(pred.score for pred in predictions)
        (pos_highest if fin_score >= 0 else neg_highests).append(highest_pred_score)
        (pos_sums if fin_score >= 0 else neg_sums).append(score_sum)
        all_pred_scores.append([pred.score for pred in predictions])
        all_eval_scores.append(scores)
        #print(example)
        #print(cmds)
    print("POS HIGHEST", stats.describe(pos_highest))
    print("NEG HIGHEST", stats.describe(neg_highests))
    print("POS sum", stats.describe(pos_sums))
    print("NEG sum", stats.describe(neg_sums))
    run_toy_class(pos_highest, neg_highests, pos_sums, neg_sums, all_pred_scores, all_eval_scores)
    return all_cmds, all_cons


