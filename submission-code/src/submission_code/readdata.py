import json
from pathlib import Path
import sys
from typing import Set
import attr


@attr.s(auto_attribs=True, frozen=True)
class DataItem:
    nl: str
    cmd: str
    src: str


@attr.s(auto_attribs=True, frozen=True)
class ACDataset:
    examples: Set[DataItem]


def get_all_data() -> ACDataset:
    cur_file = Path(__file__).parent.absolute()
    with open(cur_file / "nl2bash-data.json", 'r') as f:
        data = [
            DataItem(item['invocation'], item['cmd'], "nl2bash")
            for id, item in json.load(f).items()
        ]

    return ACDataset(examples = set(data))


def main():
    print(list(get_all_data().examples)[:10])


if __name__ == "__main__":
    main()
