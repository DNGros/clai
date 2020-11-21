import os
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from combustable.assert_on_assign import eq

from eval_splits import split_dataset
from exploregivein import SetExamplesVectorizer
from readdata import get_all_data, DataItem, ACDataset


class SimpleGiveInModel(pl.LightningModule):
    def __init__(
        self,
        vectorizer: SetExamplesVectorizer,
        hidden_size: int = 64
    ):
        super().__init__()
        self.vectorizer = vectorizer
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=vectorizer.nl_vocab_size(),
            embedding_dim=hidden_size,
            padding_idx=vectorizer.nl_pad_ind(),
        )
        self.pred_util = nn.Linear(hidden_size, vectorizer.num_utils())
        self.valid_acc = pl.metrics.Accuracy()
        self.train_acc = pl.metrics.Accuracy()

    def forward(self, cmd):
        # in lightning, forward defines the prediction/inference actions
        #nl = self.vectorizer.nl_to_inds(cmd)
        nl, _ = cmd
        return self.predict(nl)


    def validation_step(self, batch, batch_idx):
        nl, tgt = batch
        tgt = tgt.squeeze(0)
        #print(tgt)
        #print(self.predict(nl))
        self.valid_acc(self.predict(nl), tgt)
        #print("valid acc", self.valid_acc)
        self.log("valid_acc", self.valid_acc)

    def training_step(self, example, batch_idx):
        #nl = torch.Tensor(self.vectorizer.nl_to_inds(example.nl_norm.split())).long().unsqueeze(0)
        nl, tgt = example
        tgt = tgt.squeeze(0)
        #y = self.vectorizer.convert_pred(example.cmd)
        preds = self.predict(nl)
        self.train_acc(preds, tgt)
        loss = F.cross_entropy(preds, tgt)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        self.log("train_acc_step", self.train_acc)
        return loss

    def predict(self, nl_inds):
        return self.pred_util(self.sumerize_encoding(self.encode(nl_inds)))

    def encode(self, nl_inds):
        return self.embedding(nl_inds)

    def sumerize_encoding(self, encoding):
        batch_size, seq_len, eq[self.hidden_size] = encoding.shape
        out = encoding.mean(axis=1)
        batch_size, eq[self.hidden_size] = out.shape
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())#, lr=1e-3)
        return optimizer


class SetPredDataModule(pl.LightningDataModule):
    def __init__(self, data: ACDataset):
        super().__init__()
        self.train, self.test = split_dataset(data, seed=1, train_size=.95)
        self.vectorizer = SetExamplesVectorizer(self.train.examples)
        self.train = self.torchize(self.train.examples)
        self.test = self.torchize(self.test.examples)

    def torchize(self, data: Iterable[DataItem]):
        return [
            (
                torch.Tensor(self.vectorizer.nl_to_inds(ex.nl_norm)).long(),
                torch.Tensor([self.vectorizer.convert_pred(ex.cmd).utils[0]]).long(),
            )
            for ex in data
        ]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=1)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=1)


def main():
    data = get_all_data(preparse=False)
    data_module = SetPredDataModule(data)
    trainer = pl.Trainer(max_epochs=50)
    model = SimpleGiveInModel(data_module.vectorizer)
    trainer.fit(
        model, datamodule=data_module
    )


if __name__ == "__main__":
    main()
