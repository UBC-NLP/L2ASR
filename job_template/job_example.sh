#!/bin/bash
#SBATCH --account=<---ACCOUNT NAME--->
#SBATCH --gres=gpu:1
#SBATCH --mail-user=<---EMAIL--->
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --job-name=example_script
#SBATCH --output=out_%x.out
#SBATCH --error=err_%x.err

module load python/3.8
module load scipy-stack
module load gcc arrow
module load cuda cudnn
source <---YOUR VIRTUAL ENV--->

pwd

echo -e '\n Example SLURM script for L2 ASR jobs'


#For hyperparameter tuning of decoder:
#Fill in list of parameters
#alphas=(  )
#betas=(  )
#beams=(  )  

#Set based on ranges of hyperparameters:
#For instance for set of ranges of axbxc for instance 2x3x4:
#  a_num=$ID/12    
#  b_num=$ID%3  
#  c_num=($ID%12)/3 
#let alphanum=$SLURM_ARRAY_TASK_ID
#let betanum=$SLURM_ARRAY_TASK_ID
#let beamnum=$SLURM_ARRAY_TASK_ID


ALPHA=${alphas[${alphanum}]}
BETA=${betas[${betanum}]}
BEAM=${beams[${beannum}]}


echo -e '\nHyperparameters are:'
echo -e '\nALPHA:' ${ALPHA}
echo -e '\nBETA:' ${BETA}
echo -e '\nBEAM:' ${BEAM}

PROJ_DIR=<---YOUR PROJECT DIRECTORY--->

python ./src/evaluate_pipeline.py \
       --PROCESSOR_PATH \
                     facebook/wav2vec2-large-960h-lv60-self \
       --SPLIT_PATH  \
                     <---DATA SPLIT PATH---> \
       --MODEL_PATHS \
         <---MODEL CHECKPOINT PATH--->
       --EXPERIMENT_NAMES \   #Example fill in with env vars
       example_A${ALPHA/./_}_B${BETA/./_}_BM${BEAM} \
       --SAVE_PATH \
                     ./example_A${ALPHA/./_}_B${BETA/./_}_BM${BEAM} \
       --CORPUS <---Corpus---> \   #e.g. LS 
       --LM_PATH ./KenLM/combined_4gram_fixed.arpa \
       --VOCAB_PATH ./KenLM/combined-vocab-final.txt \
       --ALPHA=${ALPHA} \
       --BETA=${BETA} \
       --BEAM=${BEAM}

deactivate

echo 'Done'
