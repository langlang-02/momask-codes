# BONES-SEED SOMA -> MoMask 数据准备

这个目录用于路线 A：把 BONES-SEED 的 `soma_uniform` BVH 转成 MoMask/HumanML3D 风格数据。输出目录默认是 `dataset/BONES-SOMA/`。

## 输出格式

脚本会生成：

```text
dataset/BONES-SOMA/
├── new_joint_vecs/   # 263 维 HumanML3D 风格 motion feature
├── texts/            # MoMask 文本描述文件
├── train.txt
├── val.txt
├── test.txt
├── Mean.npy
└── Std.npy
```

`texts/<id>.txt` 每行格式为：

```text
caption#token/POS token/POS#0.0#0.0
```

## 准备原始数据

先解压 SOMA Uniform：

```bash
cd bones-seed
tar -xzf soma_uniform.tar.gz
cd ..
```

## 小规模测试

建议先转换少量样本：

```bash
python dataset/bones-seed_prepare/soma_to_hml3d.py \
  --bones-root bones-seed \
  --metadata bones-seed/metadata/seed_metadata_v004.csv \
  --output-root dataset/BONES-SOMA \
  --limit 20 \
  --overwrite
```

如果要使用 temporal labels 做动作片段切分：

```bash
python dataset/bones-seed_prepare/soma_to_hml3d.py \
  --bones-root bones-seed \
  --metadata bones-seed/metadata/seed_metadata_v004.csv \
  --temporal-labels bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --use-temporal-labels \
  --output-root dataset/BONES-SOMA \
  --limit 20 \
  --overwrite
```

## 全量转换

```bash
python dataset/bones-seed_prepare/soma_to_hml3d.py \
  --bones-root bones-seed \
  --metadata bones-seed/metadata/seed_metadata_v004.csv \
  --temporal-labels bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --use-temporal-labels \
  --output-root dataset/BONES-SOMA \
  --drop-mirrors
```

`--drop-mirrors` 会跳过镜像动作，数据量更小；如果想保留镜像增强，去掉这个参数。

## 关节映射

SOMA 到 HumanML3D 22 joints 的映射在：

```text
soma_hml3d_mapping.py
```

如果报错提示缺少某些 SOMA joints，先打印一个 BVH 的关节名：

```bash
python dataset/bones-seed_prepare/soma_to_hml3d.py \
  --bones-root bones-seed \
  --metadata bones-seed/metadata/seed_metadata_v004.csv \
  --print-joints \
  --limit 1
```

然后根据实际关节名修改 `SOMA_TO_HML3D`。

## 后续训练

转换完成后，还需要在训练脚本中增加 `dataset_name == "bones"` 分支，让代码读取：

```text
dataset/BONES-SOMA/
```

建议配置为 HumanML3D-like：

```text
joints_num = 22
dim_pose = 263
fps = 20
kinematic_chain = t2m_kinematic_chain
```

## G1 原生控制序列

`g1_to_momask.py` 不将机器人动作转成人形 263 维特征，而是保留 35 维控制表示：根平移 3 维（米）、根旋转 3 维（弧度）和 29 个 G1 DoF（弧度）。输入 CSV 的根平移单位为厘米，旋转与 DoF 单位为度；转换与生成时会自动互相还原。

小规模转换：

```bash
python dataset/bones-seed_prepare/g1_to_momask.py \
  --bones-root bones-seed \
  --metadata bones-seed/metadata/seed_metadata_v004.csv \
  --output-root dataset/BONES-G1 \
  --limit 20 --overwrite
```

全量完成或续跑最后一次必须加入 `--rebuild-stats`，让 `Mean.npy`、`Std.npy`、`Min.npy`、`Max.npy` 和 split 基于全部已转换样本重建：

```bash
python dataset/bones-seed_prepare/g1_to_momask.py \
  --bones-root bones-seed \
  --output-root dataset/BONES-G1 \
  --rebuild-stats
```

该转换按官方 120 FPS CSV 每隔 6 帧取样，输出 20 FPS 的最多 196 帧动作窗口。输出目录仍使用 `new_joint_vecs/` 这个历史目录名以复用 MoMask 数据加载器，但其中 `.npy` 的 shape 为 `(frames, 38)`：根平移 3 维（米）、连续 6D 根旋转和 29 个单自由度关节角（弧度），不是 HumanML3D 的 263 维特征。默认写入 `dataset/BONES-G1-rot6d/`，以保留旧的 35 维 Euler 数据集。

### RVQ 关节角可视化

训练 `bones_g1` RVQ 时，每 10 个 epoch 默认会在
`checkpoints/bones_g1/<name>/g1_visualization/E####/` 写入验证集的关节角对比图和误差表。PNG 的每个面板显示一个 G1 DoF：黑线为 Ground Truth，青色虚线为 RVQ 重建，橙线为右轴的绝对角度误差；CSV 提供每个关节的 MAE、RMSE 和最大误差，单位都是度。它用于检查 tokenization/reconstruction，不是 Transformer 随机文本生成的最终评估。

用 `--g1_vis_every_e 10` 调整周期，或用 `--g1_vis_every_e 0` 关闭；`--g1_vis_samples 2` 可在每个周期输出两个验证样本。

也可以独立比较两条同 FPS 轨迹：

```bash
python visualization/plot_g1_joint_angles.py \
  --ground_truth dataset/BONES-G1-rot6d/new_joint_vecs/<sample>.npy \
  --prediction generation/<ext>/motions/<sample>.npy \
  --output_dir generation/<ext>/joint_angle_plots \
  --fps 20 --prediction_label "Generated"
```
