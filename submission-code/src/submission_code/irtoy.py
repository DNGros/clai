from typing import List, Tuple

from vec4ir.base import Matching, Tfidf
from pprint import pprint
from vec4ir.core import Retrieval

from datacleaning import normalize_nl
from modeling import Model, Prediction
from readdata import get_all_data, ACDataset


class ToyIRModel(Model):
    def __init__(self, inner_model, data: ACDataset):
        self.inner_model = inner_model
        self.data = data

    def predict(self, invocation: str, n=10) -> List[Prediction]:
        query = normalize_nl(invocation)
        print(query)
        top_inds, scores = self.inner_model.query(
            query,
            k=n, return_scores=True)
        return [
            Prediction(self.data.examples[ind].cmd, score, str(self.data.examples[ind]))
            for ind, score in zip(top_inds, scores)
        ]
        pass


def load_model() -> Model:
    data = get_all_data()
    matching_op = Matching()
    tfidf = Tfidf()
    retrieval = Retrieval(retrieval_model=tfidf, matching=matching_op)
    retrieval.fit(
        X=[
            d.nl_norm for d in data.examples
        ],
        #y=[
        #    d.cmd for d in data.examples
        #]
    )
    return ToyIRModel(retrieval, data)


def main():
    model = load_model()
    pprint(model.predict("counts lines in foo.txt"))


if __name__ == "__main__":
    main()