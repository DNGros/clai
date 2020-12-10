from pathlib import Path
from pprint import pprint
import numpy as np
import sklearn
from scipy import stats
from typing import Iterable, Tuple, Dict

import torchtext
from pmap import pmap
from torch.utils.data import Dataset
from torchtext.datasets import TranslationDataset
from tqdm import tqdm
from transformers import T5Tokenizer

from exploregivein import cmd_to_seq_str_edit
from readdata import get_all_data, DataItem, ACDataset

cur_file = Path(__file__).parent.absolute()
translate_root = cur_file / "translate"

MAX_LEN = 64


def build_dataset(tokenizer):
    src_field = torchtext.data.Field(
        use_vocab=False,
        tokenize=tokenizer
    )
    tgt_field = torchtext.data.Field(
        use_vocab=False,
        tokenize=tokenizer,
    )
    data = TranslationDataset(
        str(translate_root / "base."),
        exts=("nl.txt", "cmd.txt"),
        fields=(src_field, tgt_field)
    )
    return data


def build_batch_from_tokens(src_tokens, tgt_tokens):
    source_ids = src_tokens["input_ids"].squeeze()
    if tgt_tokens:
        target_ids = tgt_tokens["input_ids"].squeeze()
    else:
        target_ids = None

    src_mask = src_tokens["attention_mask"].squeeze()  # might need to squeeze
    if tgt_tokens:
        target_mask = tgt_tokens["attention_mask"].squeeze()  # might need to squeeze
    else:
        target_mask = None

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids,
            "target_mask": target_mask}


class ProgSeq2SeqDataset(Dataset):
    def __init__(self, data: Iterable[Tuple[str, str]],
                 tokenizer, max_len_src=MAX_LEN,
                 max_len_tgt=MAX_LEN):
        self.max_len_src = max_len_src
        self.max_len_tgt = max_len_tgt
        self.tokenizer = tokenizer
        self.data = list(data)
        self.inputs = []
        self.targets = []
        self._build()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return build_batch_from_tokens(self.inputs[index], self.targets[index])

    def _build(self):
        #Jsrc_lens = []
        #Jtgt_lens = []
        for src, target in self.data:
            #src_lens.append(len(self.tokenizer(src).input_ids))
            #tgt_lens.append(len(self.tokenizer(target).input_ids))
            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [src], max_length=self.max_len_src, pad_to_max_length=True, return_tensors="pt",
                truncation=True
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=self.max_len_tgt, pad_to_max_length=True, return_tensors="pt",
                truncation=True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)
        #srcs, targets = zip(*self.data)
        #print(stats.describe(src_lens))
        #print(stats.describe(tgt_lens))
        #print(np.percentile(src_lens, (50, 75, 95, 99)))
        #print(np.percentile(tgt_lens, (50, 75, 95, 99)))
        #pprint(sorted(zip(src_lens, srcs), reverse=True)[:10])
        #pprint(sorted(zip(tgt_lens, targets), reverse=True)[:10])


def load_translation_dataset(tokenizer, type_path: str, seed, train_size, args: Dict) -> Dataset:
    nls = (translate_root / "base.nl.txt").read_text().split("\n")
    cmds = (translate_root / "base.cmd.txt").read_text().split("\n")
    assert len(nls) == len(cmds)
    data = list(zip(nls, cmds))
    if type_path == "train":
        train_ex, test_ex = sklearn.model_selection.train_test_split(
            data, train_size=train_size, random_state=seed)
        data = train_ex
    elif type_path == "val":
        train_ex, test_ex = sklearn.model_selection.train_test_split(
            data, train_size=train_size, random_state=seed)
        data = test_ex
    elif type_path == "all":
        pass
    else:
        raise ValueError(type_path)
    return ProgSeq2SeqDataset(data, tokenizer)


def build_translation_file():
    all_nl = []
    all_cmd = []
    def process(ex: DataItem):
        return ex.nl_norm, cmd_to_seq_str_edit(ex.cmd), ex
    all_examples = get_all_data(preparse=False).examples
    for nl, cmd, ex in tqdm(map(process, all_examples), total=len(all_examples)):
        if ex.pref_weight >= 0.25:
            all_nl.append(nl)
            all_cmd.append(cmd)
    translate_root.mkdir(exist_ok=True)
    (translate_root / "base.nl.txt").write_text("\n".join(all_nl))
    (translate_root / "base.cmd.txt").write_text("\n".join(all_cmd))



def main():
    build_translation_file()
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    toks = tokenizer.tokenize(
        "archive showing progress all files in wordglob CONSTANTDIRLIKE WASQUOTEDWORD to CONSTURL WASQUOTEDWORD excluding files matching".lower())
    print(toks)

    data = load_translation_dataset(tokenizer, type_path="all", args={})
    print(data)
    #for d in torchtext.data.BucketIterator(data, batch_size=1):
    #    print(d)


if __name__ == '__main__':
    main()