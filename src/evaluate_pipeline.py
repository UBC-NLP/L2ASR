import pyarrow
import argparse
import os
from typing import Any, Dict, List, Optional, Union

#import soundfile as sf
#import librosa

import torch
#from transformers import Wav2Vec2CTCTokenizer
#from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

#from datasets import load_metric, Dataset, concatenate_datasets, load_dataset
#from datasets import ClassLabel
#from dataclasses import dataclass, field

# custom script
import sys
#sys.path.append('/content/drive/MyDrive/w2v2_project/src')  # Colab
sys.path.append('/home/prsull/scratch/l2asr/src')  # Compute Canada
import asr4l2_utils as ut
import evaluate as ev


def main(args):
    "The pipeline for testing one or more models."

    evaluator = ev.Evaluator()
    evaluator.setup(args)
    evaluator.start()


if __name__ == "__main__":

    if (
        int(pyarrow.__version__.split(".")[1]) < 16
        and int(pyarrow.__version__.split(".")[0]) == 0
    ):

        os.kill(os.getpid(), 9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Parse input args")
    parser.add_argument(
        "--PROCESSOR_PATH", required=True,
        help="Path to the processor",
    )
    parser.add_argument(
        "--SPLIT_PATH", required=False,  # required for ARCTIC
        help="Path to the test dataset",
    )
    parser.add_argument(
        "--L1", required=False, 
        help="L1 to be evaluated on"
    )
    parser.add_argument(
        "--REMOVED_IDS", required=False, default=[], nargs="*",
        help="Zero or more speaker IDs to be removed from dev set",
    )
    parser.add_argument(
        "--MODEL_PATHS", required=True, default=[], nargs="+",
        help="One or more paths to the checkpoint or model name",
    )
    parser.add_argument(
        "--EXPERIMENT_NAMES", required=True, default=[], nargs="+",
        help="One or more experiment names (to be used for the pred-ref text file and the scoreboard)",
    )
    parser.add_argument(
        "--SAVE_PATH", required=True, 
        help="Path for saving the prediction-reference pair files"
    )
    parser.add_argument(
        "--CORPUS", required=True, 
        help="ARC=L1- or L2-ARCTIC, LS=LibriSpeech"
    )
    parser.add_argument(
        "--LM_PATH", required=False, default=None, 
        help="Path to KenLM arpa"
    )
    parser.add_argument(
        "--VOCAB_PATH", required=False, default=None, 
        help="Path to unigram_list from OpenSLR"
    )
    parser.add_argument(
        "--ALPHA", required=False, type=float, default=0.5,
        help="ALPHA value for pyctcdecode"
    )
    parser.add_argument(
        "--BETA", required=False, type=float, default=1.5,
        help="BETA value for pyctcdecode"
    )
    parser.add_argument(
        "--BEAM", required=False, type=int, default=100,
        help="BEAM value for pyctcdecode"
    )
    parser.add_argument(
        "--DEV_ONLY", required=False, type=bool, default=False,
        help="For doing hyperparameter tuning"
    )
    args = parser.parse_args()
    main(args)





