#!/usr/bin/env python
# coding=utf-8

import pyarrow
import argparse
import pandas as pd
import numpy as np
import random
import sys
import re
import os
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import json
from collections import Counter

import soundfile as sf
import librosa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets
from dataclasses import dataclass, field
from datasets import ClassLabel

import pyarrow

def print_(*string):
    print(*string)
    sys.stdout.flush()


def fix_seed(seed):
    # Random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)  ## what is attention_mask.ne(1)?

        batch["labels"] = labels

        return batch



# ------ DATA LOADER --------

def load_LibriSpeech(data_path):

    dev_clean = Dataset.load_from_disk(os.path.join(data_path, "dev-clean"))
    dev_other = Dataset.load_from_disk(os.path.join(data_path, "dev-other"))
    test_clean = Dataset.load_from_disk(os.path.join(data_path, "test-clean"))
    test_other = Dataset.load_from_disk(os.path.join(data_path, "test-other"))

    return dev_clean, dev_other, test_clean, test_other


def load_ARCTIC(split_path, purposes=["train", "dev", "test"], L1="all", \
      removed_ids=[]):
    """Given a list of removed_ids, load the corresponsing train, dev and test
    set where test set has two types: `test small` only contains speakers in
    removed_ids while `test big` contains all speakers of L1 type.

    Args:
        L1 (str) -- L1 type of speakers. Default: "all", means load all speakers
                    except those to be removed
        removed_ids (list) -- A list of speakers to be removed. Default: [],
                    means nothing to be removed and no small test set
    """
    train_set, dev_set, test_set, test_unseen_set = [], [], [], []
    train, dev, test, test_unseen = [], [], [], []

    # Collect file paths
    for d in os.listdir(split_path):
        if d.startswith("."):
            continue
        else:
            speaker_path = os.path.join(split_path, d)
            cur_L1, cur_speaker = d.split("_")
            if L1 == "all":
                test_set.append(os.path.join(speaker_path, "test"))
                if cur_speaker in removed_ids:  # add to unseen; no train/dev
                    test_unseen_set.append(os.path.join(speaker_path, "test"))
                else:  # add to train/dev
                    train_set.append(os.path.join(speaker_path, "train"))
                    dev_set.append(os.path.join(speaker_path, "dev"))
            elif L1 == cur_L1:
                test_set.append(os.path.join(speaker_path, "test"))
                if cur_speaker in removed_ids:  # add to unseen; no train/dev
                    test_unseen_set.append(os.path.join(speaker_path, "test"))
                else:  # add to train/dev
                    train_set.append(os.path.join(speaker_path, "train"))
                    dev_set.append(os.path.join(speaker_path, "dev"))

    # Create dataset objects
    if "train" in purposes:
        print_(f"train loading... {[p.split('/')[-2] for p in train_set]}")
        train = concatenate_datasets(list(map(Dataset.load_from_disk, train_set)))
    if "dev" in purposes:
        print_(f"dev loading... {[p.split('/')[-2] for p in dev_set]}")
        dev = concatenate_datasets(list(map(Dataset.load_from_disk, dev_set)))
    if "test" in purposes:
        print_(f"test loading... {[p.split('/')[-2] for p in test_set]}")
        test = concatenate_datasets(list(map(Dataset.load_from_disk, test_set)))
        if len(test_unseen_set):
            print_(f"test_unseen loading... {[p.split('/')[-2] for p in test_unseen_set]}")
            test_unseen = concatenate_datasets(
                list(map(Dataset.load_from_disk, test_unseen_set))
                )

    # Verify L1
    if not L1 == "all":
        if len(train): assert train[0]["L1"] == L1
        if len(dev): assert dev[0]["L1"] == L1
    print_(f"train {len(train)} dev {len(dev)} test {len(test)} test_unseen {len(test_unseen)}")

    # Remove columns that are not necessary
#    removed_cols = ['L1', 'gender', 'path', 'speaker_id']
    removed_cols = ['path']
    if len(train):
        train = train.remove_columns(removed_cols+['sentence_id'])
    if len(dev):
        dev = dev.remove_columns(removed_cols)
    if len(test):
        test = test.remove_columns(removed_cols)
    if len(test_unseen):
        test_unseen = test_unseen.remove_columns(removed_cols)

#    return {"train": train, "dev": dev, "test": test, "test_unseen": test_unseen}
    return train, dev, test, test_unseen
