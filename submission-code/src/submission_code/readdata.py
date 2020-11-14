import json
from pathlib import Path
import sys
from typing import Set, List, Tuple
import attr

from datacleaning import normalize_nl, replace_cmd

cur_file = Path(__file__).parent.absolute()


@attr.s(auto_attribs=True, frozen=True, repr=False)
class DataItem:
    nl: str
    nl_norm: str
    cmd: str
    src: str
    pref_weight: float = 1.0

    def __str__(self):
        return f"[{self.cmd}\n\t{self.nl}\n\t{self.nl_norm}\n]"

    def __repr__(self):
        return str(self)


@attr.s(auto_attribs=True, frozen=True)
class ACDataset:
    examples: Tuple[DataItem, ...]


def get_nl2bash_data() -> List[DataItem]:
    with open(cur_file / "nl2bash-data.json", 'r') as f:
        data = [
            DataItem(item['invocation'], normalize_nl(item['invocation']), item['cmd'], "nl2bash")
            for id, item in json.load(f).items()
        ]
    return data


def get_ainix_data() -> List[DataItem]:
    data = []
    with open(cur_file / "ainix-kernal-dataset-archie.json", 'r') as f:
        for id, yset in json.load(f).items():
            for xval in yset['x']:
                for yval in yset['y']:
                    data.append(DataItem(
                        xval['x_text'],
                        normalize_nl(xval['x_text']),
                        replace_cmd(yval['y_text']),
                        "ainix",
                        yval['y_preference']
                    ))
    return data


def get_all_data() -> ACDataset:
    return ACDataset(examples=tuple(
        get_nl2bash_data() +
        get_ainix_data()
    ))


def main():
    print(list(get_all_data().examples)[:10])


if __name__ == "__main__":
    main()
