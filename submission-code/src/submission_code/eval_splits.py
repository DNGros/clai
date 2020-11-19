import sklearn
import statistics

from fake_main import predict_subset
from innereval.evaluate import compute_metric_cache
from readdata import get_all_data, ACDataset
import evaluate
from tqdm import tqdm


DEBUG_PRINT = False


def compute_score(ground_truths, predicted_cmds, predicted_confds, metric_params):
    prediction_scores = []

    for grnd_truth_cmd in ground_truths:
        for i, predicted_cmd in enumerate(predicted_cmds):

            if predicted_cmd is None or len(predicted_cmd) == 0:
                continue

            predicted_confidence = predicted_confds[i]
            #pair_score = evaluate.compute_metric(predicted_cmd, predicted_confidence, grnd_truth_cmd, metric_params)
            pair_score = compute_metric_cache(predicted_cmd, predicted_confidence, grnd_truth_cmd)
            #pair_score = max(0, pair_score)
            prediction_scores.append(pair_score)

    score = evaluate.get_score(prediction_scores)
    #print('-' * 50)
    #print(f'Ground truth: {ground_truths}')
    #print(f'Predictions: {predicted_cmds}')
    #print(f'Score: {score}')

    return score


def predict_on(data, seed=42):
    train, test = split_dataset(data, seed=seed, train_size=.95)

    cmds, confs = predict_subset(train, test, print_exs=DEBUG_PRINT)
    assert len(cmds) == len(confs) == len(test.examples)
    scores = [
        compute_score([test.examples[i].cmd], cmds[i], confs[i], {"u1": 1.0, "u2": 1.0})
        for i in tqdm(range(len(cmds)))
    ]
    mean_score = sum(scores) / len(cmds)
    print(mean_score)
    return mean_score


def main():
    data = get_all_data()
    split_results = [
        predict_on(data, seed) for
        #seed in range(10)
        #seed in range(5)
        #seed in range(20)
        seed in range(20)
    ]
    print(split_results)
    print("mean all splits", statistics.mean(split_results))


def split_dataset(data: ACDataset, train_size=0.95, seed=42):
    train_ex, test_ex = sklearn.model_selection.train_test_split(
        data.examples, train_size=train_size, random_state=seed)
    return ACDataset(train_ex), ACDataset(test_ex)


if __name__ == '__main__':
    main()