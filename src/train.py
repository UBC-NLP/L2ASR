import numpy as np
import re
import os
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import csv
import shutil

import soundfile as sf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import transformers
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from transformers import get_constant_schedule_with_warmup, get_constant_schedule, get_linear_schedule_with_warmup
from transformers import AdamW
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets
from dataclasses import dataclass, field

# cache
import pyarrow

# custom script
import sys
sys.path.append('/content/drive/MyDrive/w2v2_project/src')  # Colab
sys.path.append('/home/rosalin3/scratch/w2v2_project/src')  # Compute Canada
import asr4l2_utils as ut
import evaluate as ev


class ASRTrainer():
    """Capitalized variables are hyperparams specified by the caller."""
    def __init__(self):
        self.args = None
        self.STAGE_LIST = None
        self.LR_LIST = None
        self.MAX_STEPS_LIST = None
        self.WARMUP_STEPS_LIST = None
        self.SCHEDULER_LIST = None
        self.BATCH_SIZE = None
        self.MASKING = None

        # Output directories
        self.ckpt_path = "checkpoints"  # save all checkpoints
        self.best_ckpt_path = "best_checkpoint"

        # Tracking progress
        self.counter = 0
        self.scoreboard = []   # best WER from each stage
        self.candidates = []   # best checkpoint from each stage

        # processor, data_collator, wer_metric stay constant & shared among all stages
        self.processor = None
        self.data_collator = None
        self.wer_metric = None


    def setup(
        self, args, 
        STAGE_LIST, LR_LIST, MAX_STEPS_LIST, WARMUP_STEPS_LIST, SCHEDULER_LIST,
        BATCH_SIZE, MASKING
        ):

        assert len(STAGE_LIST) == len(LR_LIST)
        assert len(STAGE_LIST) == len(MAX_STEPS_LIST)
        assert len(STAGE_LIST) == len(WARMUP_STEPS_LIST)
        assert len(STAGE_LIST) == len(SCHEDULER_LIST)

        self.args = args
        self.STAGE_LIST = STAGE_LIST
        self.LR_LIST = LR_LIST
        self.MAX_STEPS_LIST = MAX_STEPS_LIST
        self.WARMUP_STEPS_LIST = WARMUP_STEPS_LIST
        self.SCHEDULER_LIST = SCHEDULER_LIST
        self.BATCH_SIZE = BATCH_SIZE
        self.MASKING = MASKING

        # Output directories
        self.ckpt_path = os.path.join(self.args.SAVE_PATH, self.ckpt_path)
        self.best_ckpt_path = os.path.join(self.args.SAVE_PATH, self.best_ckpt_path)

        # processor, data_collator, wer_metric stay constant & shared among all stages
        self.processor = Wav2Vec2Processor.from_pretrained(
#                           "facebook/wav2vec2-large-960h-lv60-self", 
                            self.args.PROCESSOR_PATH,
                            disable_tqdm=True
                            )
        self.data_collator = ut.DataCollatorCTCWithPadding(
                                processor=self.processor, padding=True
                                )
        self.wer_metric = load_metric("wer")


    def start(self):
        """Train a model over multiple stages."""

        ut.fix_seed(self.args.SEED)

        # Load datasets
        train, dev, _, _ = ut.load_ARCTIC(
            split_path=self.args.SPLIT_PATH, purposes=["train", "dev"], 
            L1=self.args.L1, removed_ids=self.args.REMOVED_IDS
            )

        # First stage
        model, training_args, trainer = self.instantiate(train, dev)
        trainer.train()
        self._report(trainer)
        self.counter += 1

        # All of the remaining stages
        for stage in self.STAGE_LIST:
            if self.counter == len(self.STAGE_LIST):
                break
            else: 
                # update 3 training objects and continue
                model, training_args, trainer = self.update(
                    model, training_args, trainer, train, dev)
                trainer.train()
                self._report(trainer)
                self.counter += 1

        ut.print_("\n------------ Save the best ------------")
        self.save_the_best()

        ut.print_("\n------------ Training trajectory ------------")
        self.print_config()
        self.save_logs()

        ut.print_("\n------------ Evaluate the best model ------------")
        # Load dev & test sets
        # (Reload dev because doing so caused an error saying "speech" column 
        #  was missing when it was not.)
        _, dev, test, test_unseen = ut.load_ARCTIC(
            split_path=self.args.SPLIT_PATH, purposes=["dev", "test"], 
            L1=self.args.L1, removed_ids=self.args.REMOVED_IDS
            )

        # fine-tuned model
        m_scores = self.test(self.best_ckpt_path, dev, test, test_unseen)
        # baseline model
        b_scores = self.test(self.args.PRETRAINED_MODEL, dev, test, test_unseen)
        self.write_summary(m_scores, b_scores)
        

    def test(self, ckpt_path, dev, test, test_unseen):

        test_model = Wav2Vec2ForCTC.from_pretrained(ckpt_path)
        test_model.freeze_feature_extractor()

        scores = []
        for split_name, ds in [("dev", dev), ("test", test), ("test_unseen", test_unseen)]:
#            ut.print_("Evaluating on", split_name, "...", len(ds))
#            if isinstance(ds, Dataset): ut.print_(ds.column_names)  ###
            if len(ds) == 0:
                scores.append(0)
                continue
            else:
                wer, sentIDs, preds, refs = ev.test_ARC(
                    test_model, self.processor, ds
                    )
                scores.append(wer)
                ut.print_(split_name, sentIDs[0], "\n\t", preds[0], "\n\t", refs[0])

        return scores


    def write_summary(self, m_scores, b_scores):
        """
        To control rounding error at computing % change,
        round up all scores before reporting/computing % changes.
        
        Args:
            m_scores (list of float) -- model scores
            b_scores (list of float) -- baseline scores
        """

        m_scores = list(map(lambda x: round(x, 4), m_scores))
        b_scores = list(map(lambda x: round(x, 4), b_scores))

        changes = []
        for m, b in zip(m_scores, b_scores):
            if m > 0:
                changes.append(round((m-b)/b, 4))
            else:
                changes.append(0)

        ut.print_("{0:>33}{1:>23}{2:>23}".format(
                    "dev", "test", "test_unseen"))
        ut.print_("-"*80)
        ut.print_("{0:<10}{1:>23.4f}{2:>23.4f}{3:>23.4f}".format(
                    "baseline", *b_scores))
        ut.print_("-"*80)
        ut.print_("{0:<10}{1:>23.4f}{2:>23.4f}{3:>23.4f}".format(
                    "fine-tuned", *m_scores))
        ut.print_("-"*80)
        ut.print_("{0:<10}{1:>23.2%}{2:>23.2%}{3:>23.2%}".format(
                    "change", *changes))


    def instantiate(self, train, dev):
        """Helper for `start`. Create 3 training objects and return them."""

        # Instantiate model
        model = Wav2Vec2ForCTC.from_pretrained(
#                    "facebook/wav2vec2-large-960h-lv60-self",
                      self.args.PRETRAINED_MODEL,
                      mask_feature_prob=self.MASKING.get("mask_feature_prob"), 
                      mask_feature_length=self.MASKING.get("mask_feature_length"), 
                      mask_time_prob=self.MASKING.get("mask_time_prob"),
                      layerdrop=0.1,
                      activation_dropout=0.1,
                      gradient_checkpointing=True,
                      ctc_zero_infinity = True,
                      ctc_loss_reduction="mean",  
                      pad_token_id=self.processor.tokenizer.pad_token_id,
                      vocab_size=len(self.processor.tokenizer),
                      )
        model.freeze_feature_extractor()

        # Instantiate training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.ckpt_path, self.STAGE_LIST[0]),  
#            output_dir=os.path.join(self.args.SAVE_PATH, self.STAGE_LIST[0]),  
            group_by_length=True,
            per_device_train_batch_size=self.BATCH_SIZE,
    #       gradient_accumulation_steps=2, 
            evaluation_strategy="steps",
            max_steps = self.MAX_STEPS_LIST[0],
            adam_beta1 = 0.9,
            adam_beta2 = 0.98,
            adam_epsilon = 1e-08,
            fp16=True,
            save_steps=500,  #4,  ###
            eval_steps=500,  #4,  ###
            logging_steps=500,  #4,  ###
            learning_rate=self.LR_LIST[0],
            weight_decay=0.005,
            warmup_steps=self.WARMUP_STEPS_LIST[0],
            save_total_limit=4,
            load_best_model_at_end=True, # for early stopping
            metric_for_best_model= 'wer', # for early stopping
            greater_is_better=False, # for early stopping,
            seed=self.args.SEED,
            disable_tqdm=True
            )

        # Instantiate trainer
        trainer = Trainer(
            model=model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train,
            eval_dataset=dev,
            tokenizer=self.processor.feature_extractor,
            optimizers=prepare_optimizers(model, training_args, self.SCHEDULER_LIST[0]),
        )
        trainer.add_callback(EarlyStoppingCallback(3))

        # Declare the start of new stage
        self._declare()

        return model, training_args, trainer


    def update(self, model, training_args, trainer, train, dev):
        """Helper for `start`. Update 3 objects and return them.""" 

        # Instantiate the best checkpoint so far
        best_wer_so_far = min(self.scoreboard)
        best_ckpt_so_far = self.candidates[self.scoreboard.index(best_wer_so_far)]
        if trainer.state.best_metric == best_wer_so_far:
            ut.print_(f"Best checkpoint was obtained from this stage. Keep training with the best checkpoint.")
        else:
            ut.print_(f"Failed to improve the model during this stage. Loading the best checkpoint from the previous stage(s)...")
        del model
        model = Wav2Vec2ForCTC.from_pretrained(best_ckpt_so_far)   # replace model
        model.freeze_feature_extractor()

        # Modify training args
#        training_args.output_dir=os.path.join(self.args.SAVE_PATH, self.STAGE_LIST[self.counter])
        training_args.output_dir=os.path.join(self.ckpt_path, self.STAGE_LIST[self.counter])
        training_args.max_steps=self.MAX_STEPS_LIST[self.counter]
        training_args.learning_rate=self.LR_LIST[self.counter]
        training_args.warmup_steps=self.WARMUP_STEPS_LIST[self.counter]
        training_args.seed=self.args.SEED
        training_args.disable_tqdm=True

        # Instantiate trainer (only difference is optimizers)
        trainer = Trainer(
            model=model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train,
            eval_dataset=dev,
            tokenizer=self.processor.feature_extractor,
            optimizers=prepare_optimizers(model, training_args, self.SCHEDULER_LIST[self.counter]),
        )
        trainer.add_callback(EarlyStoppingCallback(3))

        # Declare the start of new stage
        self._declare()

        return model, training_args, trainer

    def save_the_best(self):
        """Save the best checkpoint at the root directory and discard all the rest."""

        # Read the scoreboard and get the path to best checkpoint
        best_wer = min(self.scoreboard)
        best_ckpt_path = self.candidates[self.scoreboard.index(best_wer)]

        # Rename directory
#        dest = os.path.join(self.args.SAVE_PATH, self.best_ckpt_path)
#        os.rename(best_ckpt_path, dest)
#        ut.print_(f"The best checkpoint was saved at {dest}")
        os.rename(best_ckpt_path, self.best_ckpt_path)
        ut.print_(f"The best checkpoint was saved at {self.best_ckpt_path}")

        # Discard the rest
        """
        for stage in self.STAGE_LIST:
            d = os.path.join(self.ckpt_path, stage)
            try:
                ut.print_(f"Deleting {stage}")
                shutil.rmtree(d)
            except OSError as e:
                ut.print_(f"Error with {d}")
                ut.print_(e.strerror)
        """
        shutil.rmtree(self.ckpt_path)
        ut.print_("Suboptimal checkpoints were discarded.")


    def _declare(self):
        "Helper for `instantiate` and `update`. Declare the start of a new stage."
        ut.print_(f"------------{self.STAGE_LIST[self.counter]}------------")
        ut.print_(f"lr: {self.LR_LIST[self.counter]}")
        ut.print_(f"max_steps: {self.MAX_STEPS_LIST[self.counter]}")
        ut.print_(f"warmup steps: {self.WARMUP_STEPS_LIST[self.counter]}")
        ut.print_(f"lr scheduler: {self.SCHEDULER_LIST[self.counter]}")


    def _report(self, trainer):
        if not trainer.is_in_train:
            ut.print_(f"The {self.STAGE_LIST[self.counter]} stage is complete.")
            best_wer = trainer.state.best_metric
            best_ckpt = trainer.state.best_model_checkpoint
            ut.print_(f"Best WER {best_wer:.4f} at step {self._get_step_number(best_ckpt)}")
            self.scoreboard.append(best_wer)
            self.candidates.append(best_ckpt)
        else:
            ut.print_(f"ERORR: The {self.STAGE_LIST[self.counter]} stage is incomplete.")


    def _get_step_number(self, ckpt_path):
        "Given a full path for a checkpoint, return the step number."
        re_step = re.compile(r"\-(\d+)$")
        return int(re_step.search(ckpt_path).groups()[0])


    def compute_metrics(self, pred):
        """Helper for ASRTrainer."""

        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


    def save_logs(self):
        "Convert scoreboard into CSV and save."

        steps = [self._get_step_number(ckpt_path) for ckpt_path in self.candidates]
        csv_path = os.path.join(self.args.SAVE_PATH, f"training_log.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            # prep for csv
            writer = csv.writer(f)
            writer.writerow(["stage", "best_wer", "best_steps", "lr", "max_steps", "warmup"])
            # prep for out
            ut.print_("{0:<25}{1:>10}{2:>14}{3:>10}{4:>10}{5:>10}".format(
                "stage", "wer", "best step", "lr", "max step", "warmup" 
            ))
            ut.print_("-"*80)
            for stage, wer, step, lr, m_step, w_step in zip(
                    self.STAGE_LIST, self.scoreboard, steps, self.LR_LIST, 
                    self.MAX_STEPS_LIST, self.WARMUP_STEPS_LIST
                    ):
                # csv
                writer.writerow([stage, wer, step, lr, m_step, w_step])
                # out
                ut.print_("{0:<25}{1:>10.6f}{2:>14}{3:>10}{4:>10}{5:>10}".format(
                    stage, wer, step, lr, m_step, w_step))
                ut.print_("-"*80)


    def print_config(self):
        "Print training configurations."
        ut.print_("Pretrained model:", self.args.PRETRAINED_MODEL)
        ut.print_("Saved in:", os.path.join(os.path.basename(os.getcwd()), self.args.SAVE_PATH))
        ut.print_("Corpus:", os.path.basename(self.args.SPLIT_PATH))
        ut.print_("L1:", self.args.L1)
        ut.print_("Removed IDs:", *self.args.REMOVED_IDS)
        ut.print_("Seed:", self.args.SEED)
        ut.print_("Batch size:", self.BATCH_SIZE)
        ut.print_("Masking: feature_prob", self.MASKING["mask_feature_prob"], \
                  "feature_length", self.MASKING["mask_feature_length"], \
                  "mask_time_prob", self.MASKING["mask_time_prob"])


def prepare_optimizers(model, training_args, scheduler_type):
    "Given the model, training args and scheduler type, returns a tuple of optimizer and scheduler."
    assert isinstance(model, Wav2Vec2ForCTC)
    assert isinstance(training_args, TrainingArguments)
    assert scheduler_type in ["warmup2constant", "constant", "decay"]

    # optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": training_args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate, 
        eps=training_args.adam_epsilon
        )

    # scheduler
    if scheduler_type == "warmup2constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=training_args.warmup_steps, 
            )
    elif scheduler_type == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif scheduler_type == "decay":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,  # hard code zero
            num_training_steps=training_args.max_steps
        )

    return (optimizer, scheduler)

