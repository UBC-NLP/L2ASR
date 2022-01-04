#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:v100l:1
#SBATCH --mail-user=toshiko.shibano@gmail.com
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --job-name=train_demo
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn
source ~/asr4l2/bin/activate

pwd

python train_pipeline.py --PRETRAINED_MODEL ckpt_path --PROCESSOR_PATH processor_path --SPLIT_PATH split_path --L1 all --SAVE_PATH path_to_save_best_ckpt --SEED seed --CONFIG hyperparam_config_no

echo 'Done'
