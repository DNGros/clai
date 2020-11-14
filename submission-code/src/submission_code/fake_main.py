from irtoy import build_model
from tqdm import tqdm
from main import predict_one
from readdata import ACDataset


def predict_subset(training: ACDataset, test: ACDataset):
    model = build_model(training)
    all_cmds, all_cons = [], []
    for example in tqdm(test.examples):
        cmds, confs = predict_one(example.nl, model, 5)
        all_cmds.append(cmds)
        all_cons.append(confs)
        #print(example)
        #print(cmds)
    return all_cmds, all_cons
