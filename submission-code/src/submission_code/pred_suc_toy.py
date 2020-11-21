from typing import List, Iterable, TypeVar
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, train_test_split

from datacleaning import normalize_nl, filter_stop_words, stem_words
from innereval.evaluate import make_preparse_cached
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from modeling import Prediction
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def run_toy_class_highest(
    highests: List[float], vals_sums: List[float], to_pred_word_counts: List[int],
    is_pos_labels: List[int],
):
    labels = np.array(is_pos_labels)
    feats = np.array([highests, vals_sums, to_pred_word_counts]).transpose()
    clf = LogisticRegression(random_state=0).fit(feats, labels)
    print(clf.predict(feats[:2, :]))
    print(clf.predict_proba(feats[:2, :]))
    print("score", clf.score(feats, labels))
    print(clf.get_params())
    print("intercept", clf.intercept_)
    print("coeff", clf.coef_)


def run_class_indv(
    all_pred_scores: List[float],
    all_pred: List[Prediction],
    all_queries: List[str],
    all_eval_scores: List[float]
):
    all_word_counts = [sum(1 for _ in filter_stop_words(q.split())) for q in all_queries]
    all_util_count = [len(make_preparse_cached(pred.cmd).utility_names) for pred in all_pred]
    all_flag_counts = [
        sum(len(flag_set) for flag_set in make_preparse_cached(pred.cmd).flag_nodes)
        for pred in all_pred
    ]

    def proc_words(words):
        return stem_words(filter_stop_words(words))
    all_query_words = [set(proc_words(query)) for query in all_queries]
    all_return_words = [set(proc_words(normalize_nl(pred.debug_ref_nl))) for pred in all_pred]
    words_iou = [
        len(query_words & return_words) / len(query_words | return_words)
        for query_words, return_words in zip(all_query_words, all_return_words)
    ]
    print("-- Individual Scores")
    feats = np.array([
        all_pred_scores,
        all_word_counts,
        all_util_count,
        all_flag_counts,
        words_iou
    ]).transpose()
    labels = np.array(all_eval_scores) > 0

    #from interpret.glassbox import ExplainableBoostingClassifier
    clf = GaussianProcessClassifier(1.0 * RBF(1.0))
    #clf = SVC(gamma=2, C=1, probability=True)
    #clf = LogisticRegression()
    #clf = RandomForestClassifier(n_estimators=10)
    #clf = LogisticRegression(random_state=0).fit(feats, labels)
    print("Mean Label", labels.mean())

    all_scores = cross_val_score(clf, feats, labels, cv=10, n_jobs=10)
    print(all_scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (all_scores.mean(), all_scores.std() * 2))
    all_f1 = cross_val_score(clf, feats, labels, cv=10, n_jobs=10, scoring='f1_macro')
    print("F1: %0.2f (+/- %0.2f)" % (all_f1.mean(), all_f1.std() * 2))
    recall = cross_val_score(clf, feats, labels, cv=10, n_jobs=10, scoring='recall')
    print("Recall: %0.2f (+/- %0.2f)" % (recall.mean(), recall.std() * 2))

    X_train, X_test, y_train, y_test = train_test_split(feats, labels,
                                                        test_size=.8,
                                                        random_state=1)
    clf.fit(X_train, y_train)
    disp = plot_precision_recall_curve(clf, X_test, y_test)
    y_score = clf.decision_function(X_test)
    average_precision = average_precision_score(y_test, y_score)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))
    plt.show()
    #clf.fit(feats, labels)
    #print(clf.predict(feats[:2, :]))
    #print(clf.predict_proba(feats[:2, :]))
    #print("score", clf.score(feats, labels))

    #from interpret import show, init_show_server, show_link
    ##init_show_server(clf.explain_global())
    #show_link(clf.explain_global())

    #print(clf.get_params())
    #print("intercept", clf.intercept_)
    #print("coeff", clf.coef_)



def class_toy():
    #X, y = load_iris(return_X_y=True)
    m = 0.2
    s = 0.1*5
    print(expit(m*3.22+s*.2-1.93))

    print(expit(1*3.88-3.34))


if __name__ == '__main__':
    class_toy()


T = TypeVar('T')

def flatten_list(l: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in l for item in sublist]
