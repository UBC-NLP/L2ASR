import pyarrow
import argparse
import pandas as pd
import numpy as np
import random
import re
import os
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import json
from collections import Counter

import soundfile as sf
import librosa

import torch
#from transformers import Wav2Vec2ForCTC, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, HubertForCTC
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from datasets import Dataset, concatenate_datasets, load_dataset, load_metric
from datasets import ClassLabel
from dataclasses import dataclass, field
from pyctcdecode import build_ctcdecoder
#import kenlm

# custom script
import sys
#sys.path.append('/content/drive/MyDrive/w2v2_project/src')  # Colab
sys.path.append('/home/prsull/scratch/l2asr/src')  # Compute Canada
import asr4l2_utils as ut

class Evaluator():
    """The pipeline for testing one or more models on the specified corpus.
        Copora: ARC, LS
    """
    def __init__(self):
        self.args = None
        self.processor = None
        self.wer_metric = None
        self.test_sets = []
        self.decoder = None
#        self.cur_e_name = None  # Experiment name currently being evaluated
#        self.cur_model = None  # Model currently being evaluated

    def setup(self, args):

        assert args.CORPUS in {"ARC", "LS"}

        self.args = args

        self.processor = Wav2Vec2Processor.from_pretrained(
                            self.args.PROCESSOR_PATH,
                            disable_tqdm=True
                            )
        self.wer_metric = load_metric("wer")

        # Load datasets
        if args.CORPUS == "ARC":
            _, dev, test, test_unseen = ut.load_ARCTIC(
                args.SPLIT_PATH, purposes=["dev", "test"], L1=args.L1,
                removed_ids=args.REMOVED_IDS
                )
            if self.args.DEV_ONLY:
                self.test_sets = [
                    ("dev", dev), ("test", []), ("test_unseen", [])
                ]
            else:
                self.test_sets = [
                    ("dev", dev), ("test", test), ("test_unseen", test_unseen)
                ]

        
        elif args.CORPUS == "LS":
            dev_clean, dev_other, test_clean, test_other = ut.load_LibriSpeech(args.SPLIT_PATH)
            if self.args.DEV_ONLY:
                self.test_sets = [
                    ("dev_clean", dev_clean), ("dev_other", dev_other),
                    ("test_clean", []), ("test_other", [])
                ]
            else:    
                self.test_sets = [
                    ("dev_clean", dev_clean), ("dev_other", dev_other),
                    ("test_clean", test_clean), ("test_other", test_other)
                ]

        # Decoder
        if args.LM_PATH:
            vocab_dict = self.processor.tokenizer.get_vocab()
            sorted_dict = {k: v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
#           kenlm_model = kenlm.Model(args.LM_PATH)
            with open(args.VOCAB_PATH) as f:
                unigram_list = [t for t in f.read().strip().split("\n")]
            self.decoder = build_ctcdecoder(
                                      labels=list(sorted_dict.keys()),
                                      kenlm_model_path=args.LM_PATH,  #kenlm_model,
                                      unigrams=unigram_list,
                                      alpha=self.args.ALPHA,
                                      beta=self.args.BETA,                
            )


    def start(self):

        scoreboard = []
        for m_path, e_name in zip(self.args.MODEL_PATHS, self.args.EXPERIMENT_NAMES):
            torch.cuda.empty_cache()

            # Load model
#            if self.args.LM_PATH:
#                cur_model = AutoModelForCTC.from_pretrained(m_path)
#                cur_model = Wav2Vec2ForCTC.from_pretrained(m_path)
            m_path_components = m_path.split("/")
            developer, model_name = m_path_components[0], m_path_components[1]
            if developer == "facebook" and model_name.startswith("hubert"):
                cur_model = HubertForCTC.from_pretrained(m_path)
            else:
                cur_model = Wav2Vec2ForCTC.from_pretrained(m_path)
            cur_model.freeze_feature_extractor()
            # Get scores
            scores = self.test(e_name, cur_model)
            # Add the results to the scoreboard
            scoreboard.append((e_name[:30], scores))
        self.write_scoreboard(scoreboard)


    def test(self, e_name, cur_model):

        scores = []
        for split, ds in self.test_sets:
            if len(ds) == 0:
                scores.append(0)
                continue
            else:
                if self.args.CORPUS == "ARC":
                    if self.decoder:
#                        wer, sentIDs, preds, refs = self._test_ARC_withLM(cur_model, ds)  # with LM
                        wer, IDs, preds, refs = self._test_ARC_withLM(cur_model, ds)  # with LM
                    else:
#                        wer, sentIDs, preds, refs = self._test_ARC(cur_model, ds)  # without LM
                        wer, IDs, preds, refs = self._test_ARC(cur_model, ds)  # without LM
                elif self.args.CORPUS == "LS":
                    if self.decoder:
#                        wer, sentIDs, preds, refs = self._test_LS_withLM(cur_model, ds)  # with LM
                        wer, IDs, preds, refs = self._test_LS_withLM(cur_model, ds)  # with LM
                    else:
#                        wer, sentIDs, preds, refs = self._test_LS(cur_model, ds)  # without LM
                        wer, IDs, preds, refs = self._test_LS(cur_model, ds)  # without LM
#                self.write_predictions(e_name, split, sentIDs, preds, refs)
                self.write_predictions(e_name, split, IDs, preds, refs)
                scores.append(wer)

        return scores


    def write_scoreboard(self, scoreboard):

        ut.print_(f"EVALUATION ON {self.args.CORPUS}")
        ut.print_(
          "Prediction-Reference pairs saved in:", self.args.SAVE_PATH
          )
        if self.args.CORPUS == "ARC":
            ut.print_("Split type:", os.path.basename(self.args.SPLIT_PATH))
            # Header
            ut.print_(
                "{0:>50}{1:>15}{2:>15}".format(
                  "dev", "test", "test_unseen"
                )
            )
            ut.print_("-" * 80)
            # Scores
            for e_name, scores in scoreboard:
                    ut.print_(
                        "{0:<40}{1:>10.4f}{2:>15.4f}{3:>15.4f}".format(
                          e_name, *scores
                        )
                    )
        elif self.args.CORPUS == "LS":
            ut.print_("Split type: N/A")
            # Header
            ut.print_(
                "{0:>50}{1:>10}{2:>10}{3:>10}".format(
                    "dev_clean", "dev_other", "test_clean", "test_other"
                )
            )
            ut.print_("-" * 80)
            # Scores
            for e_name, scores in scoreboard:
                ut.print_(
                    "{0:<32}{1:>12.4f}{2:>12.4f}{3:>12.4f}{4:>12.4f}".format(
                      e_name, *scores
                    )
                )


    def _test_LS(self, cur_model, ds):

        def map_to_result(batch):
            cur_model.to("cuda")
            input_values = self.processor(
                batch["speech"],
                sampling_rate=batch["sampling_rate"],
                return_tensors="pt"
                ).input_values.to("cuda")

            with torch.no_grad():
                logits = cur_model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = self.processor.batch_decode(pred_ids)[0]  #.lower()  ## lowercase the prediction
            return batch

        results = ds.map(map_to_result)

        wer = self.wer_metric.compute(
            predictions=results["pred_str"], references=results["sentence"]
            )
        wer = round(wer, 4)

        return wer, results["id"], results["pred_str"], results["sentence"]


    def _test_LS_withLM(self, cur_model, ds):

        def map_to_result(batch):
            cur_model.to("cuda")
            input_values = self.processor(
                batch["speech"],
                sampling_rate=batch["sampling_rate"],
                return_tensors="pt"
                ).input_values.to("cuda")

            with torch.no_grad():
                logits = cur_model(input_values).logits.cpu().detach().numpy()[0]  # pyctcdecoder supports CPU only
#            batch["pred_str"] = self.decoder.batch_decode(logits)[0]
            batch["pred_str"] = self.decoder.decode(logits,beam_width=self.args.BEAM)
            return batch

        results = ds.map(map_to_result)

        # https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
    #    results = df_split.map(_map_to_pred, batched=True, batch_size=16, remove_columns=["speech"])

        wer = self.wer_metric.compute(
            predictions=results["pred_str"], references=results["sentence"]
        )
        wer = round(wer, 4)

        return wer, results["id"], results["pred_str"], results["sentence"]


    def _test_ARC(self, cur_model, ds):
        """The ds (dataset) needs to have fields "speech", "sampling_rate",
        "sentence"."""

        def map_to_result(batch):
            """For "sampling_rate", 16k is hard-coded because L1-ARC doesn't have
            this column."""

            cur_model.to("cuda")
            input_values = self.processor(
                batch["speech"],
                sampling_rate=16_000,  # batch["sampling_rate"], <- see the signature for the reason
                return_tensors="pt"
                ).input_values.to("cuda")

            with torch.no_grad():
                logits = cur_model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = self.processor.batch_decode(pred_ids)[0]  #.lower()  ## lowercase the prediction
            return batch

    #    ut.print_("inside of _test_ARC", df_split.column_names)
        results = ds.map(map_to_result)

        # https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
    #    results = df_split.map(_map_to_pred, batched=True, batch_size=16, remove_columns=["speech"])

        wer = self.wer_metric.compute(
            predictions=results["pred_str"], references=results["sentence"]
        )
        wer = round(wer, 4)

        # Create identifiers
        id_list = [speakerID + "-" + sentID for speakerID, sentID in zip(results["speaker_id"], results["sentence_id"])]

#        return wer, results["sentence_id"], results["pred_str"], results["sentence"]
        return wer, id_list, results["pred_str"], results["sentence"]


    def _test_ARC_withLM(self, cur_model, ds):
        """The ds (dataset) needs to have fields "speech", "sampling_rate", "sentence" """

        def map_to_result(batch):
            """For "sampling_rate", 16k is hard-coded because L1-ARC doesn't have
            this column."""

            cur_model.to("cuda")
            input_values = self.processor(
                batch["speech"],
                sampling_rate=16_000,  # batch["sampling_rate"], <- see the signature for the reason
                return_tensors="pt"
                ).input_values.to("cuda")

            with torch.no_grad():
                logits = cur_model(input_values).logits.cpu().detach().numpy()[0]    # pyctcdecoder supports CPU only
#            batch["pred_str"] = self.decoder.batch_decode(logits)[0]
            batch["pred_str"] = self.decoder.decode(logits,beam_width=self.args.BEAM)
            return batch

    #    ut.print_("inside of _test_ARC", df_split.column_names)
        results = ds.map(map_to_result)

        # https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self
    #    results = df_split.map(_map_to_pred, batched=True, batch_size=16, remove_columns=["speech"])

        wer = self.wer_metric.compute(
            predictions=results["pred_str"], references=results["sentence"]
        )
        wer = round(wer, 4)

        # Create identifiers
        id_list = [speakerID + "-" + sentID for speakerID, sentID in zip(results["speaker_id"], results["sentence_id"])]

#        return wer, results["sentence_id"], results["pred_str"], results["sentence"], results["speaker_id"]
        return wer, id_list, results["pred_str"], results["sentence"]

    def write_predictions(self, e_name, split, IDs, predictions, references):
        """Save predictions in a tab-separated format.
        Each line consists of ID, pred, ref, separated by a tab."""

        if not os.path.exists(self.args.SAVE_PATH):
            os.mkdir(self.args.SAVE_PATH)

        filepath = os.path.join(
            self.args.SAVE_PATH, f"pred-{e_name}-{split}.tsv"
            )
        with open(filepath, "w", encoding="utf-8") as f:
            for i, pred, ref in zip(IDs, predictions, references):
                f.write(i + "\t" + pred + "\t" + ref + "\n")

