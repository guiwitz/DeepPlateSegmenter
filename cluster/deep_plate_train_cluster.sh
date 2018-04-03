#!/bin/bash

#SBATCH --job-name=test_JOB-gpu
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=6hours
#SBATCH --partition=k80
#SBATCH --gres=gpu:1
#SBATCH -e /scicore/home/nimwegen/witzg/DeepLearningData/Learn20170126_manual/training_error.log
#SBATCH -o /scicore/home/nimwegen/witzg/DeepLearningData/Learn20170126_manual/training_output.log

module load Python/3.5.2-goolf-1.7.20
module load cuDNN

source /scicore/home/nimwegen/witzg/DeepPlateSegmenter/venv-deeplate-gpu/bin/activate
python /scicore/home/nimwegen/witzg/DeepPlateSegmenter/cluster/deep_plate_train.py "/scicore/home/nimwegen/witzg/DeepLearningData/Learn20170126_manual/" 128 128 1

f.close()