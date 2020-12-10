import argparse
import itertools
import statistics
from pathlib import Path
from typing import List

import more_itertools

from build_translation_file import load_translation_dataset, build_batch_from_tokens, MAX_LEN
from exploregivein import cmd_to_seq
from innereval.evaluate import compute_metric_cache, get_score
from predictionpruning import prune_duplicates
from readdata import ACDataset

cur_file = Path(__file__).parent.absolute()
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


def main():
    # from transformers import AutoTokenizer, AutoModelWithLMHead

    # tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # model = AutoModelWithLMHead.from_pretrained("t5-small")

    # input_ids = tokenizer('translate English to German: The house is wonderful.',
    #                      return_tensors='pt').input_ids
    # labels = tokenizer('Das Haus ist wunderbar.', return_tensors='pt').input_ids
    ## the forward function automatically creates the correct decoder_input_ids
    # loss = model(input_ids=input_ids, labels=labels, return_dict=True).loss
    # print(loss)
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
    print(model.device)

    input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
    labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>',
                       return_tensors='pt').input_ids
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    logits = outputs.logits

    input_ids = tokenizer("how many <extra_id_0> are in foo.txt that have less than 4 <extra_id_1>?",
                          return_tensors="pt").input_ids  # Batch size 1
    print(input_ids)
    outputs = model.generate(input_ids, num_return_sequences=5,
                             do_sample=False, num_beams=5)
    print(outputs)
    print(tokenizer.batch_decode(outputs))

## Adapted from https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        if not isinstance(hparams, argparse.Namespace):
            hparams = argparse.Namespace(**hparams)
        self.hparams = hparams

        print("hparams", hparams)
        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path)
        self.greedy_scores = []
        self.beam_scores: List[float] = []

    #def is_logger(self):
    #    return True
    #    #return self.trainer.proc_rank <= 0

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
            lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        #lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        #print("LOSSS", loss)

        self.log('train_loss', float(loss))
        #tensorboard_logs = {"train_loss": loss}
        return loss
        #return {"loss": loss, "log": tensorboard_logs}

    #def training_epoch_end(self, outputs):
    #    avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #    tensorboard_logs = {"avg_train_loss": avg_train_loss}
    #    return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        inputs = self.tokenizer.batch_decode(batch['source_ids'])
        preds_greedy = self.predict_greedy(batch)
        gts = self.tokenizer.batch_decode(batch['target_ids'])
        beam_preds = self.predict_beam(batch)
        #gts = ["" for _ in preds]
        assert len(inputs) == len(preds_greedy) == len(gts)
        for src, pred, gt, beam_pred in zip(inputs, preds_greedy, gts, beam_preds):
            beam_trim = prune_duplicates(beam_pred, max_cnt=5)
            assert len(beam_trim) <= 5
            score = compute_metric_cache(pred, 1.0, gt)
            beam_scores = [compute_metric_cache(p, 1.0, gt) for p in beam_trim]
            beam_scores += [-0.0001] * (5 - len(beam_scores))
            beam_score_agg = get_score(beam_scores)
            if batch_idx <= 3:
                print("-------")
                print(f"src: {src}")
                print(f"gt: {gt}")
                print(f"pred_greedy: {pred}")
                print(f"score_greedy: {score}")
                print(f"beam: {list(zip(beam_trim, beam_scores))}")
                print(f"beam_score_agg: {beam_score_agg}")
            self.greedy_scores.append(score)
            self.beam_scores.append(beam_score_agg)
        self.log("val_loss", loss)
        #return {"val_loss": loss}

    def on_validation_epoch_start(self) -> None:
        self.greedy_scores = []
        self.beam_scores = []

    def on_validation_epoch_end(self) -> None:
        mean_greedy_score = statistics.mean(self.greedy_scores)
        print(f"mean_greedy_score {mean_greedy_score}")
        greedy_positive_frac = (np.array(self.greedy_scores) > 0).mean()
        print(f"greedy_positive_frac {greedy_positive_frac}")
        self.log("val_epoch_greedy_score", mean_greedy_score)
        self.log("val_greedy_positive_frac", greedy_positive_frac)
        mean_beam_score = statistics.mean(self.beam_scores)
        beam_positive_frac = (np.array(self.beam_scores) > 0).mean()
        self.log("val_epoch_beam_score", mean_beam_score)
        self.log("val_beam_pos_frac", beam_positive_frac)
        print(f"mean_beam_score {mean_beam_score}")
        print(f"beam_pos_frac {beam_positive_frac}")

    def predict_greedy(self, batch):
        source_ids = batch['source_ids']
        mask = batch['source_mask']
        if len(source_ids.shape) == 1:
            source_ids, mask = source_ids.unsqueeze(0), mask.unsqueeze(0)
        outputs = self.model.generate(
            source_ids, num_return_sequences=1, do_sample=False,
            attention_mask=mask, bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id
        )
        return self.tokenizer.batch_decode(outputs)

    def predict_beam(self, batch, num_return_sequences=10, num_beams=10):
        source_ids_all = batch['source_ids']
        mask_all = batch['source_mask']
        if len(source_ids_all.shape) == 1:
            source_ids_all, mask_all = source_ids_all.unsqueeze(0), mask_all.unsqueeze(0)
        outputs = self.model.generate(
            source_ids_all, num_return_sequences=num_return_sequences, do_sample=False,
            attention_mask=mask_all, num_beams=num_beams
        )
        out_strs = self.tokenizer.batch_decode(outputs)
        return list(more_itertools.chunked(out_strs, n=10, strict=True))



    #def validation_epoch_end(self, outputs):
    #    avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #    tensorboard_logs = {"val_loss": avg_loss}
    #    return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        #return [optimizer]
        data_size = 10e3  # hardcode for now
        t_total = (
                (data_size // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=t_total)
        return [optimizer], [self.lr_scheduler]

    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None,
    #                   on_tpu=False, using_native_amp=False, using_lbfgs=False, **kwargs):
    #    if self.trainer.use_tpu:
    #        raise NotImplemented
    #        #xm.optimizer_step(optimizer)
    #    else:
    #        optimizer.step()
    #    optimizer.zero_grad()
    #    self.lr_scheduler.step()

    #def get_tqdm_dict(self):
    #    tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss),
    #                 "lr": self.lr_scheduler.get_last_lr()[-1]}

    #    return tqdm_dict

    def train_dataloader(self):
        train_dataset = load_translation_dataset(
            tokenizer=self.tokenizer, type_path="train", args=self.hparams, seed=self.hparams.seed,
            train_size=self.hparams.train_size)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True,
                                shuffle=True, num_workers=4)
        #t_total = (
        #        (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
        #        // self.hparams.gradient_accumulation_steps
        #        * float(self.hparams.num_train_epochs)
        #)
        #scheduler = get_linear_schedule_with_warmup(
        #    self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        #)
        #self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = load_translation_dataset(
            tokenizer=self.tokenizer, type_path="val", args=self.hparams, seed=self.hparams.seed,
            train_size=self.hparams.train_size)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


#logger = logging.getLogger(__name__)


#class LoggingCallback(pl.Callback):
#    def on_validation_end(self, trainer, pl_module):
#        logger.info("***** Validation results *****")
#        if pl_module.is_logger():
#            metrics = trainer.callback_metrics
#            # Log results
#            for key in sorted(metrics):
#                if key not in ["log", "progress_bar"]:
#                    logger.info("{} = {}\n".format(key, str(metrics[key])))
#
#    def on_test_end(self, trainer, pl_module):
#        logger.info("***** Test results *****")
#
#        if pl_module.is_logger():
#            metrics = trainer.callback_metrics
#
#            # Log and save results to file
#            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
#            with open(output_test_results_file, "w") as writer:
#                for key in sorted(metrics):
#                    if key not in ["log", "progress_bar"]:
#                        logger.info("{} = {}\n".format(key, str(metrics[key])))
#                        writer.write("{} = {}\n".format(key, str(metrics[key])))
                        
                        
args_dict = dict(
    #data_dir="", # path for data files
    output_dir=cur_file / "t5toy", # path to save the checkpoints
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_seq_length=64,
    learning_rate=0.001 / 2,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=200,
    train_batch_size=48,
    eval_batch_size=4,
    num_train_epochs=35,
    gradient_accumulation_steps=1,
    n_gpu=1,
    #early_stop_callback=False,
    fp_16=True, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=44,
    train_size=0.95,
)

#checkpoint_callback = pl.callbacks.ModelCheckpoint(
#    filepath=args_dict['output_dir'], prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
#)


def train_t5():
    args = argparse.Namespace(**args_dict)
    model = T5FineTuner(args)
    trainer = pl.Trainer(
        automatic_optimization=True,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        max_epochs=args.num_train_epochs,
        # early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        #limit_train_batches=0.1,
        #limit_val_batches=0.1,
    )
    trainer.fit(model)


def try_restore():
    model = T5FineTuner.load_from_checkpoint(
        str(cur_file / "lightning_logs/version_3/checkpoints/epoch=14.ckpt"),
        #hparams_file=str(cur_file / "lightning_logs/version_3/hparams.yaml"),
    )
    model.eval()
    model.model.eval()
    batch = build_batch_from_tokens(model.tokenizer.batch_encode_plus(
        ["print all user names and terminals of users who are logged in", "wc -l"],
        max_length=MAX_LEN, truncation=True, return_tensors='pt',
        padding=True
    ), None)
    p = model.predict_greedy(batch)
    print(p)
    print(batch)
    print("UNSQUUEZE", batch['source_ids'].unsqueeze(0))
    #g = model.predict_greedy(batch)
    g = model.predict_beam(batch)
    #g = model.tokenizer.batch_decode(model.model.generate(
    #    input_ids=batch['source_ids'].unsqueeze(0),
    #    attention_mask=batch['source_mask'].unsqueeze(0),
    #    #max_length=64,
    #    #num_beams=20,
    #    do_sample=True,
    #    top_p=0.8,
    #    num_return_sequences=20
    #))
    print(g)


if __name__ == "__main__":
    #main()
    train_t5()
    #try_restore()
