from typing import List, Tuple
from pathlib import Path
import pickle

from vec4ir.base import Matching, Tfidf
from pprint import pprint
from vec4ir.core import Retrieval

from datacleaning import normalize_nl
from modeling import Model, Prediction
from readdata import get_all_data, ACDataset


save_file = Path(Path(__file__).parent / "modelsave.pkl").absolute()



class ToyIRModel(Model):
    def __init__(self, inner_model, data: ACDataset):
        self.inner_model = inner_model
        self.data = data

    def predict(self, invocation: str, n=10) -> List[Prediction]:
        query = normalize_nl(invocation)
        #print(query)
        top_inds, scores = self.inner_model.query(
            query,
            k=n,
            return_scores=True
        )
        return sorted([
            Prediction(
                self.data.examples[ind].cmd,
                score*self.data.examples[ind].pref_weight,
                eval_prob=1.0,
                debug=str(self.data.examples[ind])
            )
            for ind, score in zip(top_inds, scores)
        ], key=lambda pred: pred.score, reverse=True)[:n]


def build_model_all_data() -> Model:
    return build_ir_model(get_all_data())


def build_model(data: ACDataset) -> Model:
    return build_ir_model(data)


def build_ir_model(data) -> Model:
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


def cache_model(model: Model) -> None:
    with save_file.open("wb") as fp:
        pickle.dump(model, fp)


def load_model() -> Model:
    with save_file.open("rb") as fp:
        return pickle.load(fp)


def main():
    model = load_model()
    for pred in model.predict("counts lines in all shap files pid 3255"):
        print("---")
        print(pred.cmd)
        print(pred.prob)
        print(pred.debug)


if __name__ == "__main__":
    #build_ir_model()
    main()
    #main()