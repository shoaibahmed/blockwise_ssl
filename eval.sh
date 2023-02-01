#!/bin/bash

for num_blocks in {1..4}; do
    srun -p RTXA6000 -K -N1 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-gpu=8 --mem=200G \
                    --kill-on-bad-exit --job-name "bt-dl_expand_pool_noise-blocks-"$num_blocks"-lincls" --nice=0 \
                    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_20.10-py3.sqsh \
                    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
                    /opt/conda/bin/python evaluate.py /ds/images/imagenet/ \
                    ./checkpoint_block_sim_expand_pool_fs1_noise_0.25_all/resnet50.pth --num-blocks $num_blocks --lr-classifier 0.1 --filter-size 1 \
                    --workers 8 --checkpoint-dir ./checkpoint_block_sim_expand_pool_fs1_noise_0.25_all/block-$num_blocks/lincls/ \
                    > ./checkpoint_block_sim_expand_pool_fs1_noise_0.25_all/block-$num_blocks-lincls.log 2>&1 &
done
