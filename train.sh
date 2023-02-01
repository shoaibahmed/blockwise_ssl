#!/bin/bash

srun -p RTXA6000 -K -N2 --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-gpu=8 --mem=400G \
                --kill-on-bad-exit --job-name "block_sim_expand_pool_fs1_noise_0.25_all" --nice=0 \
                --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.10-py3.sqsh \
                --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                /opt/conda/bin/python main.py /ds/images/imagenet/ \
                --epochs 300 --batch-size 2048 --learning-rate 0.2 --lambd 0.0051 --projector 8192-8192-8192 --scale-loss 0.024 \
                --checkpoint-dir ./checkpoint_block_sim_expand_pool_fs1_noise_0.25_all/ --filter-size 1 --noise-type "all" --noise-std 0.25
