from abc import ABC, abstractmethod
import abc
from typing import List, Tuple
import attr


@attr.s(frozen=True, auto_attribs=True)
class Prediction:
    cmd: str
    score: float
    eval_prob: float
    debug: str

    def dump_str(self) -> str:
        return "Prediction(\n" + "\n".join([
            f"\tscore: {self.score:.4f}",
            f"\teval_prob: {self.eval_prob:.4f}",
            f"\tcmd: {self.cmd}",
            f"\tdebug: {self.debug}",
        ]) + ")"


class Model(ABC):
    @abstractmethod
    def predict(self, invocation: str, n=10) -> List[Prediction]:
        pass
