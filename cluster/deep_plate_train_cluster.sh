#!/bin/bash

#SBATCH --job-name=test_JOB-gpu \n')
#SBATCH --time=06:00:00 \n')
#SBATCH --mem-per-cpu=8G \n')
#SBATCH --ntasks=1 \n')
#SBATCH --cpus-per-task=1 \n')
#SBATCH --qos=6hours \n')
#SBATCH --partition=k80 \n')
#SBATCH --gres=gpu:1 \n')
#SBATCH -e /scicore/home/nimwegen/witzg/DeepLearningData/Learn20170126/training_error.log
#SBATCH -o /scicore/home/nimwegen/witzg/DeepLearningData/Learn20170126/training_output.log

module load Python/3.5.2-goolf-1.7.20
module load cuDNN
source /scicore/home/nimwegen/witzg/DeepPlateSegmenter/venv-deeplate-gpu/bin/activate
python deep_plate_train.py "/scicore/home/nimwegen/witzg/DeepLearningData/Learn20170126/" 96 96 1

f.close()