from eval_splits import compute_score
from innereval.evaluate import make_preparse, score_preparses
from innereval.utils.metric_utils import compute_metric


def is_same(gt: str, pred: str, conf: float = 1.0):
    gtp = make_preparse(gt)
    predp = make_preparse(pred)
    assert compute_metric(pred, conf, gt, {'u1': 1.0, 'u2': 1.0}) == score_preparses(predp, conf, gtp)


def test_parse_simple():
    is_same("ls -l", "ls -la")


def test_parse_simple_conf():
    is_same("ls -l", "ls -la", 0.5)


def test_parse_simple_conf2():
    is_same("ls -l", "ls -la", 0)


def test_parse_simple_pipe():
    is_same("ls -l", "ls -la | echo -n", 1.0)


def test_parse_slack():
    """The example Parth Thar mentioned on slack"""
    is_same("find -type f -name $(echo -n 'foo')", "find -type f -name $(echo 'foo')")


def test_parse_slack2():
    is_same("find -type f -name $(echo -n 'foo')", "find -type f -name foo | echo bar")
    
    
def test_parse_other():
    is_same("find $HOME -name '*.ogg' -type f -exec du -h '{}' \;", "find $HOME -iname '*.ogg' -type f -size -100M")

