from datacleaning import normalize_nl
from innereval import evaluate
from innereval.evaluate import compute_metric_cache
from irtoy import build_model
from tqdm import tqdm
from main import predict_one
from pred_suc_toy import run_toy_class_highest, run_class_indv
from readdata import ACDataset
from scipy import stats

highests = []
vals_sums = []
to_pred_word_counts = []
is_pos_labels = []
all_pred_scores = []
all_queries = []
all_pred = []
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
        to_pred_word_counts.append(len(normalize_nl(example.nl).split(" ")))
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
        #(pos_highest if fin_score >= 0 else neg_highests).append(highest_pred_score)
        #(pos_sums if fin_score >= 0 else neg_sums).append(score_sum)
        highests.append(highest_pred_score)
        vals_sums.append(score_sum)
        is_pos_labels.append(1 if fin_score >= 0 else 0)

        predictions = [pred for pred in predictions if not pred.is_pad]
        all_pred_scores.extend([pred.score for pred in predictions])
        all_pred.extend(predictions)
        scores = [compute_metric_cache(pred.cmd, pred.eval_prob, example.cmd) for pred in predictions]
        all_eval_scores.extend(scores)
        all_queries.extend([example.nl for _ in predictions])
        #print(example)
        #print(cmds)
    #print("POS HIGHEST", stats.describe(pos_highest))
    #print("NEG HIGHEST", stats.describe(neg_highests))
    #print("POS sum", stats.describe(pos_sums))
    #print("NEG sum", stats.describe(neg_sums))
    #run_toy_class_highest(highests, vals_sums, to_pred_word_counts, is_pos_labels)
    run_class_indv(all_pred_scores, all_pred, all_queries, all_eval_scores)
    return all_cmds, all_cons


def find_best_matches(training: ACDataset, compare_cmd: str, return_n: int = 5):
    all_scores = [
        compute_metric_cache(ex.cmd, 1.0, compare_cmd)
        for ex in training.examples
    ]
    best_scores = sorted(zip(all_scores, training.examples), key=lambda t: t[0], reverse=True)
    best_scores = best_scores[:min(len(best_scores), return_n)]
    return best_scores


def explore_worst_examples(training: ACDataset, test: ACDataset):
    model = build_model(training)
    all_cmds, all_cons, all_preds = [], [], []

    all_scores_indv = []
    all_fin_scores = []
    for example in tqdm(test.examples, mininterval=10):
        cmds, confs, predictions = predict_one(example.nl, model, 9)
        all_cmds.append(cmds)
        all_cons.append(confs)
        assert len(predictions) >= 5
        all_preds.append(predictions)
        scores_indv = [compute_metric_cache(cmd, conf, example.cmd) for cmd, conf in zip(cmds, confs)]
        all_scores_indv.append(scores_indv)
        fin_score = evaluate.get_score(scores_indv[:5])
        all_fin_scores.append(fin_score)

    combo = list(zip(all_fin_scores, test.examples, all_preds, all_scores_indv))
    #combo.sort(key=lambda v: v[0])
    for fin_score, example, preds, indv_scores in combo[:10]:
        print("Fin Score", fin_score)
        print("Example", example)
        print("Predictions")
        print("\n".join([
            f"{i} ({score}): {pred.dump_str()}"
            for i, (score, pred) in enumerate(zip(indv_scores, preds))
        ]))
        print("Best", find_best_matches(training, example.cmd, 5))
        print(model.predict(example.nl))

    return all_cmds, all_cons



