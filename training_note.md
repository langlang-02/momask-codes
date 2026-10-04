# MoMask 训练笔记

## 2026-09-21: 统一物理存储，保持旧 checkpoint 兼容

三个 A800 RVQ 任务均已由平台确认 `Completed`：rot6d68
`t-20260917232002-t4dh9`、sincos66 `t-20260917232035-7nlkf`、motion_ae70
`t-20260918222234-vt2c7`。

v1 根目录不变：`/EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1`。
任务内仍为 `/root/efs/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1`。

- `merged_joint_vecs/<sample>.npy`：统一的 `(T, 113)` float32 样本，前72维为
  原始特征库，后41维为 motion_ae 扩展，按列原样拼接，不重新计算速度或统计量。
  EFS 对大目录重命名返回 `EXDEV`，因此此目录使用相对符号链接发布，实际指向
  `.merged_joint_vecs.incomplete/`。后者虽然保留了临时名称，但发布后就是正式数据，
  **不是缓存，不要删除**；是否完成以 `merged_storage.json` 为准。相对链接兼容任务挂载路径。
- `merged_storage.json`：完整物理列布局、两份来源签名、样本/帧数及逐文件精确一致校验结果。
  只有全量检查通过才发布此文件；RVQ `MotionDataset` 检测到有效标记后优先读统一文件。
- 根目录 `dataset_info.json` 和 `Mean/Std/Min/Max.npy` 仍描述原72维逻辑基础格式；
  `extensions/motion_ae_v1` 中的小型元数据/统计量仍描述41维扩展。请勿把根目录统计量
  直接用于113维文件，应通过 `load_representation_stats` 获取选定表征的统计量。
- 原 `new_joint_vecs/` 和 `extensions/motion_ae_v1/features/` 暂保留为回退备份。
  新RVQ训练读取统一文件，不再逐样本加载旁路。直接传旧NPY路径的可视化仍兼容。
- 三个表征YAML、68/66/70维模型输入、归一化和逻辑数据签名不变，已有checkpoint
  可以继续使用。合并后的113维NPY也可直接传给重建脚本。
- 恢复旧存储读取只需将 `merged_storage.json` 改名停用（先停止使用该数据的任务）；
  不需要修改checkpoint。不要删除扩展元数据和统计量。

可复用的合并命令（已发布的目录拒绝重复覆盖；中断后的临时样本会重新校验后复用）：

```bash
.venv/bin/python dataset/bones-seed_prepare/merge_g1_feature_bank.py \
  --data-root /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 --workers 32
```

继续训练仍用原 `REPRESENTATION_CONFIG` 和 `G1_DATA_ROOT`。三个原任务目标是500轮，
若从 `latest.tar` 续训，设置 `IS_CONTINUE=1`，并将 `RVQ_MAX_EPOCH` 设为大于已完成轮数。
本次合并不自动提交新任务、不修改原checkpoint，也不改动旧训练loss/学习率策略。

本次完成验证：187,777条样本、15,925,244帧全部逐值一致；9项回归测试通过。
每种表征按train/val/test各抽32条，合并前后输入及归一化结果完全一致。
三个 `latest.tar` 均成功恢复（保存的epoch字段为499），并在本机L4完成前向、反向、
优化器更新、临时保存与重新加载；原checkpoint没有被写入。
数据根目录的 `merged_validation_report.json` 保存实际checkpoint步数和测试结果。

## 2026-09-18: BONES-G1 表征对照与旁路数据

以下是当前 G1 RVQ 实验配置；后文通用 MoMask/Slurm 示例不代表本轮云端训练参数。

| 拼接顺序/字段 | rot6d68 | sincos66 | motion_ae70 |
| --- | ---: | ---: | ---: |
| joint_angles | 29 | 29 | 29 |
| joint_vel (rad/frame) | 29 | 29 | - |
| joint_velocity (rad/s) | - | - | 29 |
| root_roll_pitch_rot6d (Euler yaw去除) | 6 | - | - |
| root_rotation_anchor6d (四元数w/z投影航向去除) | - | - | 6 |
| root_roll_pitch_sincos | - | 4 | - |
| root_yaw_vel (rad/frame) | 1 | 1 | - |
| root_xy_vel_body (m/frame) | 2 | 2 | - |
| root_height (m) | 1 | 1 | - |
| root_linear_velocity_body (m/s) | - | - | 3 |
| root_angular_velocity_body (rad/s) | - | - | 3 |
| 总维度 | 68 | 66 | 70 |

- 三种表征均为 20 FPS，原始 BONES CSV 为 120 FPS，当前每 6 帧取一帧。20 FPS 是本轮对照实验固定值，不代表 tracker 最优帧率。
- 速度转换使用当前帧的 yaw-only 航向系，不是完整倾斜机体系，也不是上一帧航向系。旧68D/66D用Euler yaw，motion_ae用四元数w/z投影；两者在同时有roll/pitch时不同，不能混用。
- 原68D/66D的 `*_vel` 是每帧差分；sincos 顺序为 `[sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]`。
- motion_ae70 按 `joint_angles29, joint_velocity29, root_rotation_anchor6d6, root_linear_velocity_body3, root_angular_velocity_body3` 拼接。rot6d 为旋转矩阵前两列按列拼接。
- motion_ae 原仓库直接读取提供的速度。本扩展从完整降采样轨迹估计：关节 unwrap 后差分乘 FPS；线速度为 XYZ 后向差分乘 FPS；世界角速度为 `rotvec(R[t] @ R[t-1].T) * FPS`，再转换到当前航向系。角速度不是 Euler 角直接差分。
- 完整源动作首帧速度为零；先计算再切窗口，不把每个窗口第一帧重新置零。保持与原数据相同的样本ID、窗口、文本及 train/val/test 划分。
- 新70D不含绝对高度；可视化使用输入首帧XYZ/yaw作为辅助锚点，XYZ通过速度积分，因此高度也可能漂移。辅助锚点不作为模型输入。

### 数据存放与正在运行任务的隔离

开发机基础目录：`/EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1`。
任务中通过 EFS 挂载访问 `/root/efs/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1`。

```text
BONES-G1-feature-bank-v1/
  new_joint_vecs/<sample>.npy          # 原72D，保持不变
  anchors/                           # 原XYZ/yaw，保持不变
  Mean.npy, Std.npy, train.txt, ...    # 原统计量/划分/元数据，保持不变
  extensions/motion_ae_v1/
    features/<sample>.npy             # 追加41D，非替换
    Mean.npy, Std.npy, Min.npy, Max.npy
    dataset_info.json                 # 字段单位、基础签名、完成状态
    conversion_report.json           # 覆盖数量、窗口一致性及错误
```

70D加载器从原72D取关节角29，从旁路取速度35和motion_ae专用去航向rot6d 6。最初拟只保存35D速度，交叉核对motion_ae代码后发现其航向定义不同，因此必须额外保存6D旋转，旁路实际41D，模型仍为70D。旁路统计量仅由训练集计算。
motion_ae初始航向由输入首帧旋转重新求四元数w/z投影，不直接使用旧anchors中的Euler yaw。严格倒置导致w/z投影退化的样本会报错，不会静默改用另一种航向。
扩展未完成、样本缺失、帧数不匹配或基础签名变化时拒绝训练。原68D/66D签名不包含扩展目录，旧checkpoint继续使用原统计量。
本轮不物理合并数据、不停止/重启训练、不自动提交任务；等当前训练结束后另行安排合并。转换默认2个CPU worker、无GPU，共享存储I/O仍可能有轻微竞争。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python dataset/bones-seed_prepare/g1_to_motion_ae_extension.py \
  --base-root /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 \
  --bones-root /EFS/users/zhaohui.yang/dataset/bones-seed-g1 \
  --workers 2
```

中断后可加 `--resume-incomplete` 重建未完成扩展；不会覆盖已完成扩展。
本轮全量转换及验收已完成：124,617个源动作、187,777个窗口（train168,895 / val9,294 / test9,588），训练统计使用14,321,261帧。转换报告无错误或缺失，原72D窗口逐值匹配，原元数据/统计量/划分hash未变。另完成全ID覆盖检查及96个分层抽查，详见扩展目录的 `validation_report.json`。新增及回归测试8项通过；原两个云端任务检查时仍为Running、重试0次、batch256。
`.conversion.lock` 防止同时运行两个转换器；进程被强制杀死后，确认转换器及其worker全部退出再人工清理残留锁。
小规模转换使用独立 `--output-root /tmp/... --limit 8`，产物标记 partial，禁止用于正式训练。

### 训练配置、batch与续训

| 表征 | 表征YAML (`configs/representations/`) | A800任务YAML (`volc/A800/`) | 实验名 |
| --- | --- | --- | --- |
| 68D | g1_local_rot6d68_v1.yaml | train_bones_g1_rvq_rot6d68.yaml | bones_g1_rvq_rot6d68_v1_nq6 |
| 66D | g1_local_sincos66_v1.yaml | train_bones_g1_rvq_sincos66.yaml | bones_g1_rvq_sincos66_v1_nq6 |
| 70D | g1_motion_ae70_v1.yaml | train_bones_g1_rvq_motion_ae70.yaml | bones_g1_rvq_motion_ae70_v1_nq6 |

原68D/66D正在运行的任务是 **单卡batch256，global batch256**，不会因本地YAML变动而改变。
三份YAML现为 **每卡2048**，只影响以后提交。当前没有梯度累积，`global batch = 每卡batch × world_size`；单卡2048，8卡16384。
默认 A800 1卡、500epoch、window64、6层RVQ、seed3407、学习率2e-4，重建为标准化特征SmoothL1，另加 `0.5 * 全特征相邻帧差分loss + 0.02 * commitment loss`。
学习率里程碑按iteration计算；batch从256升到2048后，每epoch的step数减少，不应认为与旧实验训练进度完全等价。未自动线性放大学习率。
L4已验证小batch小模型的数据读取、保存/恢复及重建流程；另用正式尺寸（width512、depth3、6层量化器全部启用、FP32、window64、batch2048）完成两次合成输入训练step，峰值allocated约6.62 GiB、reserved约6.87 GiB。该结果是显存/执行检查，不代表训练质量，也不是A800实测。报告保存在 `/EFS/users/zhaohui.yang/dataset/momask-motion-ae-validation-v1/batch2048_L4_report.json`。

数据完成后提交新70D任务：

```bash
mlp job create -f volc/A800/train_bones_g1_rvq_motion_ae70.yaml
```

代码使用vePFS挂载的当前工程，容器中为 `/root/code/momask-codes`，解释器为该目录 `.venv/bin/python`；没有复制代码到EFS。沿用A800队列 `q-20260706191152-scrkm`。
checkpoint保存到开发机 `/EFS/users/zhaohui.yang/dataset/momask-checkpoints/bones_g1/<实验名>/model/`。
续训时使用同一表征/数据签名与实验名，将YAML的 `IS_CONTINUE` 设为 `'1'`，确认 `latest.tar` 存在且目标epoch更大；不能跨68D/66D/70D续训。
若要用2048重新开始对照实验，应把 `VQ_NAME` 改成独立名字（如原名加 `_bs2048`），避免覆盖旧checkpoint；不要向正在运行的实验目录再次提交。
DDP需同步修改资源GPU数量、`DDP=1`、`NPROC_PER_NODE`及可见GPU配置，batch参数仍为每卡值。

本仓库的训练流程分为三个阶段：先训练运动 tokenizer（RVQ-VAE），再训练 masked text-to-motion transformer，最后训练 residual transformer。生成时会依次使用这三个模块：文本先生成粗粒度 motion token，再补全 residual token，最后由 VQ decoder 还原成连续动作特征。

## 1. 训练顺序

推荐按以下顺序训练：

```bash
python train_vq.py --name rvq_name --gpu_id 0 --dataset_name t2m --batch_size 256 --num_quantizers 6
python train_t2m_transformer.py --name mtrans_name --gpu_id 0 --dataset_name t2m --batch_size 64 --vq_name rvq_name
python train_res_transformer.py --name rtrans_name --gpu_id 0 --dataset_name t2m --batch_size 64 --vq_name rvq_name --cond_drop_prob 0.2 --share_weight
```

在 Slurm 作业中，如果只申请了一张 GPU，通常使用 `--gpu_id 0`。`CUDA_VISIBLE_DEVICES` 会把当前作业可见的 GPU 重新编号，因此第一张可见 GPU 是 `cuda:0`。

## 2. 第一阶段：RVQ-VAE / Motion Tokenizer

入口文件是 `train_vq.py`。

这一阶段训练 `RVQVAE`，目标是重建连续 motion feature，并学习一组 residual vector quantization codebook。主模型定义在 `models/vq/model.py`：

- `RVQVAE`：包含 encoder、residual quantizer 和 decoder。
- `LengthEstimator`：根据文本 embedding 预测动作长度，主要用于生成阶段。

相关模块：

- `models/vq/encdec.py`：定义 `Encoder` 和 `Decoder`。
- `models/vq/residual_vq.py`：定义 residual quantization 堆叠。
- `models/vq/quantizer.py`：定义 EMA codebook quantizer。
- `models/vq/vq_trainer.py`：定义 `RVQTokenizerTrainer` 训练循环。

前向流程是：

```text
motion -> Encoder -> ResidualVQ -> Decoder -> reconstructed motion
```

loss 在 `RVQTokenizerTrainer.forward()` 中计算：

```text
loss = reconstruction_loss + loss_vel * local_position_loss + commit * commitment_loss
```

模型和统计信息保存到：

```text
checkpoints/<dataset_name>/<name>/model/
checkpoints/<dataset_name>/<name>/meta/
```

其中 `meta/` 目录保存 `mean.npy` 和 `std.npy`，后续 transformer 阶段会继续使用这些归一化参数。

## 3. 第二阶段：Masked Text-to-Motion Transformer

入口文件是 `train_t2m_transformer.py`。

这一阶段会加载已经训练好的 VQ 模型并冻结它，只训练 `MaskTransformer`。它的任务不是直接预测连续动作，而是根据文本预测第一层 quantizer 的离散 motion token。

模型定义在：

```text
models/mask_transformer/transformer.py
```

训练流程：

```text
caption + motion
-> VQ encoder 将 motion 编码成 token indices
-> 使用 code_idx[..., 0] 作为粗粒度 token 目标
-> 随机 mask 一部分 motion token
-> CLIP 编码 caption
-> transformer 预测被 mask 的 token id
-> cross entropy loss
```

训练器是 `models/mask_transformer/transformer_trainer.py` 中的 `MaskTransformerTrainer`。核心代码是：

```python
code_idx, _ = self.vq_model.encode(motion)
loss, pred_ids, acc = self.t2m_transformer(code_idx[..., 0], conds, m_lens)
```

`MaskTransformer` 内部加载 `ViT-B/32` 版本的 CLIP，并冻结其参数。训练 checkpoint 中不会保存 CLIP 权重，之后构建模型时会重新加载 CLIP。

## 4. 第三阶段：Residual Transformer

入口文件是 `train_res_transformer.py`。

这一阶段同样加载训练好的 VQ 模型，但训练目标是预测后续 residual quantizer 层的 token。模型类是 `ResidualTransformer`，定义位置同样是：

```text
models/mask_transformer/transformer.py
```

训练器是：

```text
models/mask_transformer/transformer_trainer.py
```

训练流程：

```text
caption + motion
-> VQ encoder 得到所有 quantizer 层的 code_idx
-> 随机选择一个 residual quantizer 层
-> 将之前层的 code embedding 求和作为 history_sum
-> 输入文本条件和 quantizer-layer id
-> 预测当前 residual 层 token
```

核心调用：

```python
code_idx, all_codes = self.vq_model.encode(motion)
ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)
```

生成时，`ResidualTransformer` 会在 coarse token 之后逐层补齐 residual token。

## 5. 数据如何导入

主要数据集代码在：

```text
data/t2m_dataset.py
```

### VQ 训练数据

`train_vq.py` 使用 `MotionDataset`。这个 dataset 只读取 motion feature，不读取文本。

读取路径包括：

```text
dataset/HumanML3D/new_joint_vecs/<id>.npy
dataset/HumanML3D/train.txt
dataset/HumanML3D/val.txt
```

每个样本是一个固定长度 motion window：

```python
motion = self.data[motion_id][idx:idx + self.opt.window_size]
```

默认 `window_size=64`。motion 会用数据集的 `Mean.npy` 和 `Std.npy` 做归一化。

### Transformer 训练数据

`train_t2m_transformer.py` 和 `train_res_transformer.py` 都使用 `Text2MotionDataset`。它读取文本和动作配对数据。

读取路径包括：

```text
dataset/HumanML3D/new_joint_vecs/<id>.npy
dataset/HumanML3D/texts/<id>.txt
dataset/HumanML3D/train.txt
dataset/HumanML3D/val.txt
```

文本文件每行格式大致是：

```text
caption#token token token#start_time#end_time
```

如果 `start_time` 和 `end_time` 都是 0，则使用完整 motion；否则截取指定片段：

```python
n_motion = motion[int(f_tag * 20): int(to_tag * 20)]
```

`Text2MotionDataset.__getitem__()` 返回：

```python
caption, motion, m_length
```

motion 长度会被对齐到 `unit_length` 的倍数，通常是 4，并 padding 到 `max_motion_length`。

## 6. Evaluation Loader

评估 dataloader 由 `motion_loaders/dataset_motion_loader.py` 中的 `get_dataset_motion_loader()` 构建。它使用 `Text2MotionDatasetEval`，并从 `glove/` 加载词向量。

这个 dataloader 主要用于 evaluator 计算指标，例如：

- FID
- diversity
- top-k retrieval
- matching score

它不是主训练 dataloader，而是验证模型质量时使用。

## 7. 重要张量形状

HumanML3D：

```text
motion feature dim = 263
joints = 22
```

KIT：

```text
motion feature dim = 251
joints = 21
```

典型张量流：

```text
motion:      (batch, frames, feature_dim)
code_idx:    (batch, token_length, num_quantizers)
token_length = frames // 4
```

第一个 transformer 训练 `code_idx[..., 0]`，也就是第一层 coarse token。Residual transformer 训练完整的多层 quantizer token。

## 8. 关键文件索引

- `train_vq.py`：训练 RVQ-VAE tokenizer。
- `train_t2m_transformer.py`：训练 masked text-to-motion transformer。
- `train_res_transformer.py`：训练 residual transformer。
- `gen_t2m.py`：文本生成动作入口。
- `data/t2m_dataset.py`：定义 motion dataset 和 text-motion dataset。
- `models/vq/model.py`：定义 `RVQVAE`。
- `models/mask_transformer/transformer.py`：定义 `MaskTransformer` 和 `ResidualTransformer`。
- `models/vq/vq_trainer.py`：VQ 训练循环。
- `models/mask_transformer/transformer_trainer.py`：transformer 训练循环。
- `motion_loaders/dataset_motion_loader.py`：评估 dataloader。

## 9. 输出目录

训练 checkpoint：

```text
checkpoints/<dataset_name>/<run_name>/model/
```

训练日志：

```text
log/vq/<dataset_name>/<run_name>/
log/t2m/<dataset_name>/<run_name>/
log/res/<dataset_name>/<run_name>/
```

训练过程中的评估动画：

```text
checkpoints/<dataset_name>/<run_name>/animation/
```

生成结果：

```text
generation/<ext>/joints/
generation/<ext>/animations/
```

## 10. 简要总结

MoMask 的核心思想是先把连续人体动作压缩为离散 token，再用文本条件生成这些 token。`RVQVAE` 负责学习 motion token 空间；`MaskTransformer` 根据文本生成第一层粗 token；`ResidualTransformer` 根据文本和已有 token 补齐 residual token；最后通过 VQ decoder 还原成连续动作。




RAW_DIR=/DRoboticsAILab/zhaohui.yang/datasets/bones-seed/raw/hf_2026-08-13
mkdir -p "$RAW_DIR"

python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

raw_dir = Path("/DRoboticsAILab/zhaohui.yang/datasets/bones-seed/raw/hf_2026-08-13")

snapshot_download(
    repo_id="bones-studio/seed",
    repo_type="dataset",
    allow_patterns=[
        "g1.tar.gz",
        "metadata/seed_metadata_v004.csv",
        "metadata/seed_metadata_v002_temporal_labels.jsonl",
        "LICENSE.md",
        "README.md",
    ],
    local_dir=str(raw_dir),
    max_workers=4,
)
PY

cd /DRoboticsAILab/zhaohui.yang/momask-codes

  MOMASK_PYTHON=/DRoboticsAILab/zhaohui.yang/momask-codes/.venv/bin/python \
  RAW_CSV=/DRoboticsAILab/zhaohui.yang/EFS/users/zhaohui.yang/dataset/bones-seed-g1/g1/csv/230227/come_down_50cm_box_R_002__A219_M.csv \
  VQ_NAME=bones_g1_rvq_mixed_root_absxyz_nq6 \
  CHECKPOINTS_DIR=/DRoboticsAILab/zhaohui.yang/EFS/users/zhaohui.yang/dataset/momask-checkpoints \
  CHECKPOINT=net_best_val \
  MJCF=/DRoboticsAILab/zhaohui.yang/unitree_mujoco/unitree_robots/g1/scene_29dof.xml \
  OUTPUT_ROOT=/DRoboticsAILab/zhaohui.yang/EFS/users/zhaohui.yang/dataset/momask-recon-vis \
  DEVICE=cpu \
  bash g1_visualize/reconstruct_g1_local.sh


带root xyz+位移表征的重建可视化：
VQ_NAME=bones_g1_rvq_mixed_root_absxyz_nq6_l4_ddp8 \
CHECKPOINTS_DIR=/d-robotics-nas-dev/users/zhaohui.yang/train_jobs/momask-g1/momask-checkpoints \
CHECKPOINT=net_best_val \
DEVICE=cuda:0 \
OUTPUT_ROOT=/EFS/users/zhaohui.yang/dataset/momask-recon-vis \
bash g1_visualize/reconstruct_g1_local.sh \
/EFS/users/zhaohui.yang/dataset/bones-seed-g1/g1/csv/210531/jump_and_land_light_001__A001.csv
# rot6d68 Transformer tasks (2026-09-21 UTC)

- Mask: `t-20260922000914-pfhvm`, config `volc/A800/train_bones_g1_mask_rot6d68.yaml`.
- Residual: `t-20260922000915-x8vzf`, config `volc/A800/train_bones_g1_res_rot6d68.yaml`.
- Submitted with `mlp job create -f <config>`; both initially Queueing (possible resource fragmentation). A800 BS2048 runtime memory is not yet verified.
- Each task uses one A800, per-device/global batch 2048, 500 epochs, LR 2e-4, 5-epoch warmup, decay x0.1 after epochs 200 and 400.
- Feature view: `configs/representations/g1_local_rot6d68_v1.yaml`, loaded from merged feature-bank storage with original RVQ normalization.
- Frozen RVQ: `bones_g1_rvq_rot6d68_v1_nq6/model/frozen_latest_ep0499_it8998000.tar`; source latest has ep=499, total_it=8998000.
- RVQ SHA256: `51d776fa01876b27bc6f675196e6ee75b4881dac35c9f917836294b01689353c`.
- Checkpoint base on development host: `/EFS/users/zhaohui.yang/dataset/momask-checkpoints/bones_g1/`.
- Mask experiment: `bones_g1_mtrans_rot6d68_latest_ep499_bs2048_v1`.
- Residual experiment: `bones_g1_res_rot6d68_latest_ep499_bs2048_v1`.
- Each experiment saves `model/latest.tar`, best checkpoints, and `model/ep0050.tar` through `model/ep0500.tar` every 50 completed epochs. Resume restores optimizer, scheduler, best metrics and batch progress; RVQ binding and schedule must match.
- TensorBoard base: `/EFS/users/zhaohui.yang/dataset/momask-log/t2m/bones_g1/` (both stages use this base).
- Task mount mapping: `/root/code/momask-codes` = current vePFS checkout; `/root/efs` = development host `/EFS`.
- Verification: 11 unit tests passed; both full Transformer models passed L4 real-data batch-2 training, validation, checkpoint save and resume to epoch 2. Frozen RVQ tensors were unchanged. CLIP cache copied to shared storage and SHA256 verified.
# RVQ full-test reconstruction evaluation (2026-09-22 UTC)

- Entrypoint: `evaluation/eval_g1_rvq.py`; metric definitions: `evaluation/g1_metrics.py`; usage/protocol: `evaluation/README.md`.
- Output: `/EFS/users/zhaohui.yang/dataset/momask-rvq-eval/test_full_20260922_v1/`.
- Local L4, `.venv`, batch 32, float32 with TF32 disabled; no cloud job, rendering or physics rollout. Elapsed 301.72 seconds including shared GT loading.
- All six checkpoints evaluated on the identical 9,588 test clips / 811,003 frames per checkpoint. Full coverage, finite results, action means/P95 and per-joint pooled RMSE independently verified.
- Below: equal-weight mean of per-clip metrics, not frame-weighted values. Epochs are raw checkpoint `ep` values (latest ep499 = 500 completed RVQ epochs).

| Representation | Checkpoint | Epoch | Global MPJPE mm | Root-relative MPJPE mm | Root FDE m | Joint RMSE deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| rot6d68 | net_best_val | 14 | 107.1898 | 29.8309 | 0.1923 | 4.2980 |
| rot6d68 | latest | 499 | 101.6630 | 28.5225 | 0.1800 | 4.1921 |
| sincos66 | net_best_val | 12 | 108.6906 | 30.2715 | 0.1945 | 4.2888 |
| sincos66 | latest | 499 | 103.0308 | 29.2847 | 0.1821 | 4.2007 |
| motion_ae70 | net_best_val | 1 | 192.8650 | 46.3527 | 0.3731 | 5.5326 |
| motion_ae70 | latest | 499 | 100.9281 | 27.4346 | 0.1803 | 4.3576 |

- Latest beats its own best-validation checkpoint on these four metrics for all representations. motion_ae latest has the lowest action-mean global/root-relative MPJPE; rot6d latest has the lowest joint-angle RMSE and slightly lower mean FDE.
- Weighting matters: pooled global MPJPE is 112.4120 mm for rot6d latest versus 113.4729 mm for motion_ae latest. Do not claim one representation uniformly dominates. Training batch/schedule differences also confound representation-only conclusions.
- Metric reference is the common base-bank tilt/joints plus all GT anchors; predictions receive only the initial state. No joint clipping or per-frame alignment correction. FK uses 29 named hinge anchors; root-relative MPJPE removes translation, not rotation.
- `comparison.csv/json` contain the six-run action-mean table. Each run includes `identity.json`, `summary.json` (action mean, P95, pooled), `per_sample.csv` and `per_joint.csv`. Hashes and protocol are recorded in the manifest and identities.
- Tests: 16 passed. Representation roundtrip max error <= 6.56e-7; batch/single max feature difference <= 1.74e-6. FK versus existing mj_forward joint-position difference <= 5.28e-8 m.
- First smoke attempt exposed TF32 batch/single differences and is incomplete (`smoke_20260922_v1`). Corrected smoke (`smoke_20260922_v2`) and the full benchmark are complete. These short-window results do not measure long-horizon drift or tracker success.

```bash
.venv/bin/python evaluation/eval_g1_rvq.py \
  --data-root /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 \
  --checkpoints-dir /EFS/users/zhaohui.yang/dataset/momask-checkpoints \
  --device cuda:0 --batch-size 32 \
  --output-dir /EFS/users/zhaohui.yang/dataset/momask-rvq-eval/NEW_RUN_ID
```
# rot6d68 text-to-motion visualization (2026-09-29 UTC)

- Both submitted Transformer jobs are Completed: Mask `t-20260922000914-pfhvm` ended 2026-09-22 14:29:30 UTC; Residual `t-20260922000915-x8vzf` ended 2026-09-23 04:32:34 UTC. Both latest checkpoints record ep=500.
- This generation uses Mask `net_best_val` ep298 (best loss 2.5370273590), Residual `net_best_val` ep458 (best loss 4.2193017006), and the training-bound rot6d68 RVQ `frozen_latest_ep0499_it8998000.tar`, not RVQ best.
- Updated `gen_g1.py` to honor mounted checkpoint roots, enforce matching Transformer/RVQ bindings and normalization, decode local 68D/66D features, and save tokens plus a generation manifest. Legacy 38D/42D decoding is preserved.
- Output: `/EFS/users/zhaohui.yang/dataset/momask-t2m-vis/rot6d68_best_20260929/`. `README.md` links all 14 MP4s; `preview.jpg` shows first/middle/last frames for repeat 0. `motions/` contains model features, native38 and tokens; `g1_csv/` contains 120Hz exports.
- Seven test-set captions and their rounded frame lengths are in `configs/generation/g1_rot6d68_representative.txt`: forward walk, sideways crossover walk, turn, step down from box, squat, jump, moving/waving. Two stochastic repeats with seed10107, CFG4, 18 Mask steps; no feature-range or joint-limit clipping. Generation uses text/length only, not source motion poses or RVQ reconstruction.
- Caption source IDs in order: `Turn_Start_Walk_0270_001__A019_M_seg007_win000`, `Sideway_Walk_Left_001__A020_M_seg000_win000`, `Turn_Start_Walk_0270_001__A019_M_seg008_win000`, `come_down_50cm_box_R_001__A051_M_seg000_win000`, `squat_002__A360_M_seg000_win000`, `Jump_002__A017_M_seg016_win000`, `jog_ff_start_360_R_wave_right_hand_090_R_turn_walk_270_R_002__A478_M_seg001_win000`.
- Kinematic MuJoCo visualization only. The box is not modeled, contacts are not simulated, and initial height is model-predicted. Passing render checks does not establish semantic accuracy or physical trackability.
- Verification: 18 unit tests passed; 14 generated native trajectories finite; every MP4 frame count checked; preview visually inspected. No training job was changed.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python gen_g1.py \
  --dataset_name bones_g1 --gpu_id 0 \
  --checkpoints_dir /EFS/users/zhaohui.yang/dataset/momask-checkpoints \
  --g1_data_root /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 \
  --name bones_g1_mtrans_rot6d68_latest_ep499_bs2048_v1 \
  --res_name bones_g1_res_rot6d68_latest_ep499_bs2048_v1 \
  --which_epoch net_best_val --repeat_times 2 --seed 10107 \
  --text_path configs/generation/g1_rot6d68_representative.txt \
  --disable_data_range_clip \
  --ext /EFS/users/zhaohui.yang/dataset/momask-t2m-vis/NEW_RUN_ID
MUJOCO_GL=egl .venv/bin/python g1_visualize/render_g1_generation.py \
  --generation-dir /EFS/users/zhaohui.yang/dataset/momask-t2m-vis/NEW_RUN_ID
```
