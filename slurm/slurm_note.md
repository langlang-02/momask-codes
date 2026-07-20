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
