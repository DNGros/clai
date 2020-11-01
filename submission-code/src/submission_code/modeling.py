from abc import ABC, abstractmethod
import abc
from typing import List, Tuple
import attr


@attr.s(frozen=True, auto_attribs=True)
class Prediction:
    cmd: str
    prob: float
    debug: str


class Model(ABC):
    @abstractmethod
    def predict(self, invocation: str, n=10) -> List[Prediction]:
        pass
