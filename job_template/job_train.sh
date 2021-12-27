#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:v100l:1
#SBATCH --mail-user=toshiko.shibano@gmail.com
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --job-name=v5b_M4HI_sd3
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn
source ~/asr4l2/bin/activate

pwd
python /home/rosalin3/scratch/w2v2_project/src/train_pipeline.py --PRETRAINED_MODEL facebook/wav2vec2-large-960h-lv60-self --PROCESSOR_PATH facebook/wav2vec2-large-960h-lv60-self --SPLIT_PATH /home/rosalin3/scratch/w2v2_project/data/l2arc_splits_v5b --L1 HI --SAVE_PATH . --SEED 123 --CONFIG 10
echo 'Done'
