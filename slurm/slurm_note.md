# Slurm 使用说明

本目录包含 BONES-SOMA 数据转换和 MoMask 三阶段训练的 sbatch 脚本。所有脚本默认在仓库根目录运行，并激活 `momask-h800` conda 环境。日志写入 `slurm/logs/%x-%j.out` 和 `slurm/logs/%x-%j.err`。

## 1. 数据转换：`convert_bones_soma.sbatch`

用途：将 BONES-SEED 的 SOMA BVH 转换为 HumanML3D 风格的 `new_joint_vecs/*.npy` 和 `texts/*.txt`，并生成 `Mean.npy`、`Std.npy`、`train.txt`、`val.txt`、`test.txt`。

常用命令：

```bash
sbatch --export=ALL,LIMIT=0,DROP_MIRRORS=1,OVERWRITE=0 slurm/convert_bones_soma.sbatch
```

从中断位置续跑并在结束后重建统计文件：

```bash
sbatch --export=ALL,LIMIT=0,START_INDEX=131000,DROP_MIRRORS=1,OVERWRITE=0,REBUILD_STATS=1 slurm/convert_bones_soma.sbatch
```

关键参数：

- `LIMIT`：转换 metadata 前多少条；`0` 表示全量。
- `START_INDEX`：从第几个 metadata row 开始转换；用于接着已中断的全量转换继续跑。
- `DROP_MIRRORS`：设为 `1` 时跳过 mirror 样本，减少重复数据。
- `OVERWRITE`：设为 `0` 时，如果 `.npy` 和 `.txt` 都已存在则复用；设为 `1` 时强制重写。
- `REBUILD_STATS`：设为 `1` 时，转换结束后扫描所有已有 `.npy/.txt`，重新生成 `Mean.npy`、`Std.npy` 和 split 文件。
- `OUTPUT_ROOT`：输出数据集目录，默认 `dataset/BONES-SOMA`。
- `VAL_RATIO` / `TEST_RATIO`：验证集和测试集比例，默认各 `0.05`。
- `USE_TEMPORAL`：设为 `1` 时使用 temporal label 做分段 caption。

注意：转换主要消耗 CPU 和文件 I/O，`--gpus=1` 主要用于满足集群 QOS 要求。全量转换必须等作业正常完成后，split 和 Mean/Std 才代表完整数据。

## 2. RVQ-VAE 训练：`train_bones_vq.sbatch`

用途：训练第一阶段动作 tokenizer / RVQ-VAE。后续两个 Transformer 都依赖这里生成的 checkpoint。

正式训练示例：

```bash
sbatch --export=ALL,NAME=bones_rvq_nq6,BATCH_SIZE=32,MAX_EPOCH=500,NUM_QUANTIZERS=6,WINDOW_SIZE=64 slurm/train_bones_vq.sbatch
```

--dependency=afterok:<jobid>
--dependency=afterok:
要写在export前面

关键参数：

- `NAME`：实验名，也是 checkpoint 目录名；输出到 `checkpoints/bones/$NAME/`。
- `BATCH_SIZE`：训练 batch size；H800 上可从 `32` 开始，确认稳定后再增大。
- `MAX_EPOCH`：训练 epoch 数。
- `NUM_QUANTIZERS`：RVQ 量化层数，当前默认 `6`。
- `WINDOW_SIZE`：训练窗口长度，当前默认 `64`。

重要输出：

```text
checkpoints/bones/<NAME>/model/net_best_fid.tar
checkpoints/bones/<NAME>/model/latest.tar
checkpoints/bones/<NAME>/meta/mean.npy
checkpoints/bones/<NAME>/meta/std.npy
```

## 3. Mask Transformer 训练：`train_bones_mtrans.sbatch`

用途：训练第二阶段文本到主 token 的 Mask Transformer。需要先完成 RVQ-VAE。

示例：

```bash
sbatch --export=ALL,NAME=bones_mtrans,VQ_NAME=bones_rvq_nq6,BATCH_SIZE=64,MAX_EPOCH=500 slurm/train_bones_mtrans.sbatch
```

关键参数：

- `NAME`：Mask Transformer 实验名；输出到 `checkpoints/bones/$NAME/`。
- `VQ_NAME`：要加载的 RVQ 实验名，必须对应 `checkpoints/bones/$VQ_NAME/`。
- `BATCH_SIZE`：训练 batch size。
- `MAX_EPOCH`：训练 epoch 数。

## 4. Residual Transformer 训练：`train_bones_res.sbatch`

用途：训练第三阶段 residual token 预测模型。需要先完成 RVQ-VAE。

示例：

```bash
sbatch --export=ALL,NAME=bones_res,VQ_NAME=bones_rvq_nq6,BATCH_SIZE=64,MAX_EPOCH=500,COND_DROP_PROB=0.2,SHARE_WEIGHT=1 slurm/train_bones_res.sbatch
```

关键参数：

- `NAME`：Residual Transformer 实验名；输出到 `checkpoints/bones/$NAME/`。
- `VQ_NAME`：要加载的 RVQ 实验名。
- `COND_DROP_PROB`：classifier-free guidance 的条件丢弃概率，默认 `0.2`。
- `SHARE_WEIGHT`：设为 `1` 时启用 `--share_weight`。

## 5. 查看作业和日志

提交后查看队列：

```bash
squeue -u fzhuah
```

实时查看日志：

```bash
tail -f slurm/logs/<job-name>-<jobid>.out
tail -f slurm/logs/<job-name>-<jobid>.err
```

TensorBoard 日志位置：

```text
log/vq/bones/<rvq_name>/
log/t2m/bones/<mtrans_name>/
log/res/bones/<res_name>/
```

启动 TensorBoard：

```bash
tensorboard --logdir log --host 0.0.0.0 --port 6006
```

## 6. G1 原生控制序列

G1 流程使用 `dataset_name=bones_g1` 和 `dataset/BONES-G1-rot6d/`。每帧是 38 维原生控制值：根平移 3 维（米）、连续 6D 根旋转和 29 个单自由度关节角（弧度）。它不会使用 HumanML3D 的 FID、R-precision 或人形 MP4 评估；RVQ、Mask Transformer 和 Residual Transformer 分别按验证重建损失、token 交叉熵保存 `net_best_val.tar`。

转换 G1 CSV：

```bash
sbatch --export=ALL,LIMIT=0,OVERWRITE=0,REBUILD_STATS=1 slurm/convert_bones_g1.sbatch
```

`convert_bones_g1.sbatch` 在 `cpu` 分区运行，不申请 GPU；它会在需要时解压 `bones-seed/g1.tar.gz`，按 120 FPS 输入每隔 6 帧取样为 20 FPS，并生成 `Mean.npy`、`Std.npy`、`Min.npy` 和 `Max.npy`。`REBUILD_STATS=1` 适合全量完成或续跑的最后一次作业；`START_INDEX`、`LIMIT`、`DROP_MIRRORS`、`OVERWRITE` 的含义与 SOMA 转换脚本相同。

三阶段训练：

```bash
sbatch --export=ALL,NAME=bones_g1_rvq_rot6d_nq6,IS_CONTINUE=0,BATCH_SIZE=256,MAX_EPOCH=500,NUM_QUANTIZERS=6 slurm/train_bones_g1_vq.sbatch
sbatch --dependency=afterok:<RVQ_JOBID> --export=ALL,NAME=bones_g1_mtrans_rot6d,VQ_NAME=bones_g1_rvq_rot6d_nq6,IS_CONTINUE=0,BATCH_SIZE=64,MAX_EPOCH=500 slurm/train_bones_g1_mtrans.sbatch
sbatch --dependency=afterok:<RVQ_JOBID> --export=ALL,NAME=bones_g1_res_rot6d,VQ_NAME=bones_g1_rvq_rot6d_nq6,IS_CONTINUE=0,BATCH_SIZE=64,MAX_EPOCH=500 slurm/train_bones_g1_res.sbatch
```

`VQ_NAME` 必须匹配 G1 RVQ 实验名。若作业超时，给相同脚本追加 `IS_CONTINUE=1` 会从 `latest.tar` 继续。Mask 与 Residual 可在 RVQ 完成后并行训练，因为 Residual 训练用真值 RVQ token，不依赖已训练的 Mask Transformer。

G1 RVQ 会每 10 个 epoch 在 `checkpoints/bones_g1/<name>/g1_visualization/E####/` 记录验证集 Ground Truth 与重建控制序列的 29 个 DoF 角度图、每关节 MAE/RMSE/最大误差 CSV。可在提交作业时加入 `G1_VIS_EVERY_E=...` 和 `G1_VIS_SAMPLES=...` 控制周期与样本数；`0` 会关闭可视化。

生成 G1 CSV：

```bash
TEXT_PROMPT="A person is walking forward." sbatch --export=ALL,MASK_NAME=bones_g1_mtrans_rot6d,RES_NAME=bones_g1_res_rot6d,MOTION_LENGTH=120 slurm/generate_bones_g1.sbatch
```

输出位于 `generation/<EXT>/motions/`（20 FPS、米/弧度 `.npy`）和 `generation/<EXT>/g1_csv/`（默认上采样为 120 FPS 的 MuJoCo-compatible CSV）。生成值默认截断到训练数据观察范围；这只能避免明显越界，部署到真实 G1 前仍须经过关节限位、速度/加速度限位、平衡与碰撞安全控制。



训练rvq
sbatch --export=ALL,NAME=bones_g1_rvq_rot6d_nq6,G1_VIS_EVERY_E=10,G1_VIS_SAMPLES=2 slurm/train_bones_g1_vq.sbatch

训练mtrans
sbatch --dependency=afterok:<RVQ_JOBID> --export=ALL,NAME=bones_g1_mtrans_rot6d,VQ_NAME=bones_g1_rvq_rot6d_nq6,IS_CONTINUE=0 slurm/train_bones_g1_mtrans.sbatch
第一次启动时设置is_continue=0,否则会找不到之前的训练结果而报错

--dependency=afterok:<RVQ_JOBID> --export  #一定要加在export前


训练res-trans
sbatch --dependency=afterok:<RVQ_JOBID> --export=ALL,NAME=bones_g1_res_rot6d,VQ_NAME=bones_g1_rvq_rot6d_nq6,IS_CONTINUE=0 slurm/train_bones_g1_res.sbatch



g1动作渲染:
sbatch --export=ALL,MOTION=dataset/BONES-G1-rot6d/new_joint_vecs/eating_apple_R_003__A235_M_seg001_win000.npy,OUTPUT_MP4=generation/g1_replay.mp4 g1_visualize/render_g1.sbatch

读取最新的rvq checkpoint，渲染重建效果
srun --account=peilab --partition=normal --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus=1 --time=00:20:00 \
  --export=ALL,MOTION=dataset/BONES-G1-rot6d/new_joint_vecs/eating_apple_R_003__A235_M_seg001_win000.npy,VQ_NAME=bones_g1_rvq_rot6d_nq6,CHECKPOINT=latest bash g1_visualize/reconstruct_g1_srun.sh


按原始csv重建
srun --account=peilab --partition=normal --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --gpus=1 --time=00:20:00 \
    --export=ALL,MOTION=bones-seed/g1/csv/230227/come_down_50cm_box_R_002__A219_M.csv,VQ_NAME=bones_g1_rvq_rot6d_nq6,CHECKPOINT=latest \
    bash g1_visualize/reconstruct_g1_srun.sh


bones-seed/g1/csv/230301/body_check_001__A230.csv




验证t2m生成效果


TEXT_PROMPT='A person brushes off dust from their right shoulder with the left arm.' \
  sbatch --export=ALL,MASK_NAME=bones_g1_mtrans_rot6d,RES_NAME=bones_g1_res_rot6d,MOTION_LENGTH=64 \
    slurm/generate_bones_g1.sbatch
    


可测试的、贴近验证集的 prompt：

A person is standing with the right leg positioned ahead, facing forward.
A person stands idle with a subtle tilt in the hips and the arms to the side, facing forward.
A person slowly walks backward and turns to the left side.
A person is walking backwards while looking in the right diagonal direction.
A person steps backward with the right leg and then turns over the left shoulder to face left.
A person makes a quick horizontal jump toward their left side.

也建议测训练分布内但动作更明显的例子：

A person stands in a place while holding a small, heavy object in their right hand.
A standing person is repeatedly pushing and pulling both hands forward and backward.

A person brushes off dust from their right shoulder with the left arm.

A person rotates the door knob with their right hand.
