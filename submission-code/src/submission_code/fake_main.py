from irtoy import build_model
from tqdm import tqdm
from main import predict_one
from readdata import ACDataset


def predict_subset(training: ACDataset, test: ACDataset, print_exs: bool = True):
    model = build_model(training)
    all_cmds, all_cons = [], []
    for example in tqdm(test.examples, mininterval=10):
        if print_exs:
            print("Example:")
            print(f"NL: {example.nl}")
            print(f"GT: {example.cmd}")
        cmds, confs = predict_one(example.nl, model, 5)
        all_cmds.append(cmds)
        all_cons.append(confs)
        #print(example)
        #print(cmds)
    return all_cmds, all_cons
