# MoMask 训练笔记

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
