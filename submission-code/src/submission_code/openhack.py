from collections import Counter
from pathlib import Path

import sklearn

from datacleaning import stem_words

cur_file = Path(__file__).parent.absolute()
translate_root = cur_file / "translate"

def split_stuff(seed, train_size):
    nls = (translate_root / "base.nl.txt").read_text().split("\n")
    nls_stemmed = [" ".join(stem_words(nl.split())) for nl in nls]
    cmds = (translate_root / "base.cmd.txt").read_text().split("\n")
    new_root = translate_root / "split1"
    new_root.mkdir(exist_ok=True)
    data = list(zip(nls_stemmed, cmds))
    train_ex, test_ex = sklearn.model_selection.train_test_split(
        data, train_size=train_size, random_state=seed)
    train_x, train_y = zip(*train_ex)
    test_x, test_y = zip(*test_ex)
    (new_root / "train.nl.txt").write_text("\n".join(train_x))
    (new_root / "train.cmd.txt").write_text("\n".join(train_y))
    (new_root / "val.nl.txt").write_text("\n".join(test_x))
    (new_root / "val.cmd.txt").write_text("\n".join(test_y))

    
if __name__ == "__main__":
    #nls = (translate_root / "base.nl.txt").read_text().split("\n")
    ##nls = nls[:20]
    #nls_stemmed = [" ".join(stem_words(nl.split())) for nl in nls]
    #words = Counter()
    #s_words = Counter()
    #for nl, stemmed in zip(nls, nls_stemmed):
    #    #print("----")
    #    #print(nl)
    #    #print(list(stem_words(nl.split())))
    #    words.update(nl.split())
    #    s_words.update(stemmed.split())
    #print(words)
    #print(len(words))
    #print(s_words)
    #print(len(s_words))
    split_stuff(seed=42, train_size=95)


