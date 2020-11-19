from sklearn.datasets import load_iris
from scipy import stats
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


def run_toy_class(
    pos_highest: List[float], neg_highest: List[float],
    pos_sums: List[float], neg_sums: List[float],
    all_pred_scores: List[List[float]], all_eval_scores: List[List[float]]
):
    labels = np.array([1] * len(pos_highest) + [0] * len(neg_highest))
    feats = np.array([pos_highest + neg_highest, pos_sums + neg_sums]).transpose()
    clf = LogisticRegression(random_state=0).fit(feats, labels)
    print(clf.predict(feats[:2, :]))
    print(clf.predict_proba(feats[:2, :]))
    print("score", clf.score(feats, labels))
    print(clf.get_params())
    print("intercept", clf.intercept_)
    print("coeff", clf.coef_)

    print("-- Individual Scores")
    feats = np.array(all_pred_scores).flatten()[:, None]
    labels = np.array(all_eval_scores).flatten() > 0
    clf = LogisticRegression(random_state=0).fit(feats, labels)
    print(clf.predict(feats[:2, :]))
    print(clf.predict_proba(feats[:2, :]))
    print("score", clf.score(feats, labels))
    print(clf.get_params())
    print("intercept", clf.intercept_)
    print("coeff", clf.coef_)


def class_toy():
    #X, y = load_iris(return_X_y=True)
    m = 0.2
    s = 0.1*5
    print(expit(m*3.22+s*.2-1.93))

    print(expit(1*3.88-3.34))


if __name__ == '__main__':
    class_toy()

