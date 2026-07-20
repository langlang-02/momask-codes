#!/usr/bin/env bash
set -euo pipefail

# ISAACLAB_DIR=/project/peilab/yanzhengyang/IsaacLab
# WORKSPACE_DIR=/project/peilab/yanzhengyang/workspace
# CACHE_DIR=/project/peilab/yanzhengyang/isaac-cache
# LOCAL_IMAGE=/project/peilab/yanzhengyang/containers/isaac-sim-5.1.0.sqsh
# SAVE_IMAGE=/project/peilab/yanzhengyang/containers/isaaclab-5.1.0-ready.sqsh

srun --account peilab \
    --partition preempt \
    --nodes 1 \
    --ntasks 1 \
    --cpus-per-task 8 \
    --gpus 1 \
    --time 08:00:00 \
    --mem 32G \
    --pty bash -l

    #--container-save "$SAVE_IMAGE" \
