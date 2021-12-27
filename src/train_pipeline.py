import pyarrow
import argparse
import pandas as pd
import numpy as np
import random
import re
import json
import os
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
import json
from collections import Counter

import soundfile as sf
import librosa

import torch
import transformers
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback

from datasets import load_metric, Dataset, concatenate_datasets, load_dataset
from datasets import ClassLabel
from dataclasses import dataclass, field

# custom script
import sys
sys.path.append('/content/drive/MyDrive/w2v2_project/src')  # Colab
sys.path.append('/home/rosalin3/scratch/w2v2_project/src')  # Compute Canada
import asr4l2_utils as ut
import train as tr


def main(args):
    "The pipeline for training one model on the given dataset."

    STAGE_LIST, LR_LIST, MAX_STEPS_LIST, WARMUP_STEPS_LIST, SCHEDULER_LIST, \
        BATCH_SIZE, MASKING = _get_config(args)

    trainer = tr.ASRTrainer()
    trainer.setup(
        args, STAGE_LIST, LR_LIST, MAX_STEPS_LIST, 
        WARMUP_STEPS_LIST, SCHEDULER_LIST,
        BATCH_SIZE, MASKING
        )
    trainer.start()


def _get_config(args):
    """Helper for main. Returns 7 parameters.

    Notes:
      STAGE_LIST -- Used for subfolder names
      LR_LIST -- Make sure "constant" lr = "warmup2constant" lr; "decay" lr > "constant_after_decay" lr
      SCHEDULER_LIST -- PyTorch scheduler names. Different from stage names!
    """

    if args.CONFIG == 0:  # single-stage test for Colab
        STAGE_LIST = ["warmup2constant"]
        LR_LIST = [3e-5]
        MAX_STEPS_LIST = [12]
        WARMUP_STEPS_LIST = [4]
        SCHEDULER_LIST = ["warmup2constant"] #, "constant", "decay"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 1:  # tri-staged test for Colab
        STAGE_LIST = ["warmup2constant", "constant", "decay"]
        LR_LIST = [3e-5, 3e-5, 1e-5]
        MAX_STEPS_LIST = [24, 24, 24]
        WARMUP_STEPS_LIST = [4, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    if args.CONFIG == 3:  # single-stage test for Compute Canada
        STAGE_LIST = ["warmup2constant"]
        LR_LIST = [3e-5]
        MAX_STEPS_LIST = [2500]
        WARMUP_STEPS_LIST = [500]
        SCHEDULER_LIST = ["warmup2constant"] #, "constant", "decay"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 4:  # tri-staged test for Compute Canada
        STAGE_LIST = ["warmup2constant", "constant", "decay"]
        LR_LIST = [3e-5, 3e-5, 1e-5]
        MAX_STEPS_LIST = [2500, 2500, 2500]
        WARMUP_STEPS_LIST = [500, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 10:  # Baseline II, M3
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [1e-5, 1e-5, 1e-5, 1e-5, 5e-6]
        MAX_STEPS_LIST = [10000, 10000, 7000, 7000, 10000]
        WARMUP_STEPS_LIST = [2000, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.5, 
        "mask_feature_length": 15, 
        "mask_time_prob": 0.5
        }
    elif args.CONFIG == 11:  # M1
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [3e-5, 3e-5, 3e-5, 3e-5, 1e-5]
        MAX_STEPS_LIST = [12500, 12500, 12500, 12500, 12500]
        WARMUP_STEPS_LIST = [2500, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.5, 
        "mask_feature_length": 15, 
        "mask_time_prob": 0.5
        }
    elif args.CONFIG == 12:  # M2
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [3e-5, 3e-5, 3e-5, 5e-6, 2e-6]
        MAX_STEPS_LIST = [12500, 12500, 5000, 5000, 10000]
        WARMUP_STEPS_LIST = [5000, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 13:  # M4AR, M5AR
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [1e-5, 1e-5, 1e-5, 1e-5, 5e-6]
        MAX_STEPS_LIST = [10000, 10000, 5000, 7000, 10000]
        WARMUP_STEPS_LIST = [2000, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 14:  # M4ES, M4HI, M5ES, M5HI
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [1e-5, 1e-5, 1e-5, 5e-6, 2e-6]
        MAX_STEPS_LIST = [10000, 10000, 5000, 5000, 10000]
        WARMUP_STEPS_LIST = [2000, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 15:  # M4KO, M5KO, M4ZH, M5ZH
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [1e-5, 1e-5, 1e-5, 1e-5, 5e-6]
        MAX_STEPS_LIST = [10000, 20000, 7000, 7000, 10000]
        WARMUP_STEPS_LIST = [2000, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }
    elif args.CONFIG == 16:  # M4VI, M5VI
        STAGE_LIST = ["warmup2constant", "constant", "decay1", "decay2", "constant_after_decay"]
        LR_LIST = [1e-5, 1e-5, 1e-5, 5e-6, 2e-6]
        MAX_STEPS_LIST = [10000, 10000, 5000, 5000, 10000]
        WARMUP_STEPS_LIST = [3000, 0, 0, 0, 0]
        SCHEDULER_LIST = ["warmup2constant", "constant", "decay", "decay", "constant"]
        BATCH_SIZE = 16
        MASKING = {
        "mask_feature_prob": 0.25, 
        "mask_feature_length": 30, 
        "mask_time_prob": 0.75
        }

    return STAGE_LIST, LR_LIST, MAX_STEPS_LIST, WARMUP_STEPS_LIST, \
        SCHEDULER_LIST, BATCH_SIZE, MASKING


if __name__ == "__main__":

    if (
        int(pyarrow.__version__.split(".")[1]) < 16
        and int(pyarrow.__version__.split(".")[0]) == 0
    ):

        os.kill(os.getpid(), 9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--PRETRAINED_MODEL", required=True,
        help="The pretrained model to be finetuned",
    )
    parser.add_argument(
        "--PROCESSOR_PATH", required=True,
        help="The processor to be used",
    )
    parser.add_argument(
        "--SPLIT_PATH", required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--L1", required=True, 
        help="L1 to be trained & evaluated on"
    )
    parser.add_argument(
        "--REMOVED_IDS", required=False, default=[], nargs="*",
        help="Zero or more speaker IDs to be removed from dev set",
    )
    parser.add_argument(
        "--SAVE_PATH", required=False, default=".", 
        help="Path for saving training logs and predictions"
    )
    parser.add_argument(
        "--SEED", required=True, type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--CONFIG", required=True, type=int,
        help="Training configurations",
    )
    args = parser.parse_args()

    # ------ CALL MAIN ------
    main(args)