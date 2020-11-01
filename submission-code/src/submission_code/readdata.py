import json
from pathlib import Path
import sys
from typing import Set, List, Tuple
import attr

from datacleaning import normalize_nl


@attr.s(auto_attribs=True, frozen=True, repr=False)
class DataItem:
    nl: str
    nl_norm: str
    cmd: str
    src: str

    def __str__(self):
        return f"[{self.cmd}\n\t{self.nl}\n\t{self.nl_norm}\n]"

    def __repr__(self):
        return str(self)



@attr.s(auto_attribs=True, frozen=True)
class ACDataset:
    examples: Tuple[DataItem]


def get_all_data() -> ACDataset:
    cur_file = Path(__file__).parent.absolute()
    with open(cur_file / "nl2bash-data.json", 'r') as f:
        data = [
            DataItem(item['invocation'], normalize_nl(item['invocation']), item['cmd'], "nl2bash")
            for id, item in json.load(f).items()
        ]

    return ACDataset(examples = tuple(data))


def main():
    print(list(get_all_data().examples)[:10])


if __name__ == "__main__":
    main()
