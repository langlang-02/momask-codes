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

