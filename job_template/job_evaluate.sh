#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:v100l:1
#SBATCH --mail-user=toshiko.shibano@gmail.com
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --job-name=evaluate_demo
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn
source ~/scratch/asr4l2/bin/activate

pwd

echo -e '\nL2ARC without LM'

python evaluate_pipeline.py --PROCESSOR_PATH processor_path --SPLIT_PATH path_to_l2arc --L1 all --MODEL_PATHS ckpt_path1 ckpt_path2 --EXPERIMENT_NAMES label1 label2 --SAVE_PATH path_to_save_preds --CORPUS ARC

echo -e '\nL2ARC with LM'

python evaluate_pipeline.py --PROCESSOR_PATH processor_path --SPLIT_PATH path_to_l2arc --L1 all --MODEL_PATHS ckpt_path1 ckpt_path2 --EXPERIMENT_NAMES label1 label2 --SAVE_PATH path_to_save_preds --CORPUS ARC --LM_PATH arpa_path --VOCAB_PATH vocab_path

echo -e '\nL1ARC with LM'

python evaluate_pipeline.py --PROCESSOR_PATH processor_path --SPLIT_PATH path_to_l1arc --L1 EN --MODEL_PATHS ckpt_path1 ckpt_path2 --EXPERIMENT_NAMES label1 label2 --SAVE_PATH path_to_save_preds --CORPUS ARC --LM_PATH arpa_path --VOCAB_PATH vocab_path

echo -e '\nLibriSpeech with LM'

python evaluate_pipeline.py --PROCESSOR_PATH processor_path --SPLIT_PATH path_to_librispeech --MODEL_PATHS ckpt_path1 ckpt_path2 --EXPERIMENT_NAMES label1 label2 --SAVE_PATH path_to_save_preds --CORPUS LS --LM_PATH arpa_path --VOCAB_PATH vocab_path

echo 'Done'
