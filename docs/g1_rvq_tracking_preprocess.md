# G1 RVQ 离线预处理与 RL 数据接口

脚本位于 `dataset/bones-seed_prepare/g1_preprocess_rvq.py`。它复用当前
MoMask RVQ 和 G1 数据工具，冻结 checkpoint，只导出条件数据，不训练或修改 RVQ。

## 当前数据与 checkpoint

- 输入：`/EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1`。
- 指定 checkpoint：`/EFS/users/zhaohui.yang/dataset/momask-checkpoints/bones_g1/bones_g1_rvq_rot6d68_v1_nq6/model/frozen_latest_ep0499_it8998000.tar`。
- checkpoint：epoch 499、iteration 8998000；SHA256
  `51d776fa01876b27bc6f675196e6ee75b4881dac35c9f917836294b01689353c`。
- 表征：`g1_local_rot6d68_v1`，20 Hz，68D；stride=4，6 层 RVQ，
  每层 512 个 code、code dimension 512；全部层 embedding 相加为 512D latent。
- 全量输出：`/EFS/users/zhaohui.yang/dataset/momask-rvq-tracking/rot6d68_ep499_v1`。
- 小样本输出：同目录下 `rot6d68_ep499_smoke`，每个 split 32 条，明确标记 `partial=true`。

“全量”指已准备好的 feature-bank 全部 187,777 个窗口：train 168,895、
val 9,294、test 9,588。保持原 split，不将所有原始 CSV 重新切窗，不编码原数据集
之前未选入的帧。窗口之间有各自的边界和可能的重叠，不能直接拼成一条轨迹。
checkpoint 的归一化和 dataset signature 必须与输入库完全一致。

## 运行与恢复

在 MoMask 根目录执行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  dataset/bones-seed_prepare/g1_preprocess_rvq.py \
  --data-root /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 \
  --checkpoint /EFS/users/zhaohui.yang/dataset/momask-checkpoints/bones_g1/bones_g1_rvq_rot6d68_v1_nq6/model/frozen_latest_ep0499_it8998000.tar \
  --output-root /EFS/users/zhaohui.yang/dataset/momask-rvq-tracking/rot6d68_ep499_v1 \
  --batch-size 128 --shard-size 2048 --workers 16
```

中断后在相同命令末尾加 `--resume`。仅复用校验通过的分片；不同 checkpoint、
元数据、代码或 batch/shard 配置拒绝混写。操作系统文件锁防止并发写同一输出目录。
先验证 `.npz.tmp` 再发布 `.npz`，已完成分片会逐次记录在 manifest 中。
输入库和冻结 checkpoint 在运行及续跑期间必须保持不变。

小样本使用独立输出目录，并加 `--limit 32`（每个 split 最多32条）。
`manifest.json` 的 `complete=true` 表示该次选择范围全部完成，只有同时
`partial=false` 才是完整数据集。`preprocess.log` 和 `preprocess.pid` 是此次
全量运行的日志和进程记录；PID 只用于当前开发机排查，不作为完成凭据。

## 分片格式

每个 `{split}-{index:05d}.npz` 包含：

| 数组 | 形状/含义 |
| --- | --- |
| `sample_ids` | 窗口 ID，Unicode 字符串，无 pickle |
| `source_paths` | 原始 CSV 相对于 BONES 根目录的路径 |
| `frame_offsets` | N+1 个原始帧累计偏移 |
| `token_offsets` | N+1 个 latent 时间步累计偏移 |
| `reference_native38` | `(所有帧,38)` float32，20 Hz 原始参考姿态 |
| `tokens` | `(所有latent时间步,6)` int32，所有量化层启用 |
| `latent` | `(所有latent时间步,512)` float32，量化 embedding 求和 |

参考姿态为 `[root_xyz_m(3), root_rot6d(6), joint_angles_rad(29)]`，
rot6d 是旋转矩阵前两列按列拼接，关节名和顺序见 manifest 的 `joint_names`。
使用原始库的关节角、roll/pitch，以及**每帧** GT XYZ/yaw anchors 恢复姿态，
没有经 RVQ decoder，也没有通过速度积分恢复全局位置。这里的“原始”指与 RVQ
训练一致的20Hz轨迹，不是120Hz CSV所有帧；关节速度等供RL使用的量需按目标
参考帧率计算，不能把旧 `rad/frame` 数据直接当 `rad/s`。

输出根目录另存 `mean.npy`、`std.npy`、`representation.json`、`codebooks.npy`
以及 `manifest.json`。完整读取/训练不依赖原 feature bank；从 token 恢复 latent
只需 codebook，不需要 RVQ encoder。原 CSV 仅用于独立验收。

```python
import numpy as np

with np.load('train-00000.npz', allow_pickle=False) as data:
    i = 0
    a, b = data['frame_offsets'][i:i+2]
    c, d = data['token_offsets'][i:i+2]
    reference = data['reference_native38'][a:b]
    latent = data['latent'][c:d]
    tokens = data['tokens'][c:d]
    # 20Hz参考帧j对应的latent；在各窗口内部计算，不能用全局packed帧下标。
    per_frame_latent = latent[np.arange(len(reference)) // 4]
```

时间约定：窗口局部时间 `t`，参考帧 `floor(t*20)`，latent 索引
`floor(t*20/4)`。20Hz帧内相位可取 `(frame % 4)/4`；更高控制频率下连续相位
应按 `t*5 - floor(t*5)` 计算。episode 只持续到最后一个真实参考帧，padding
不会延长参考轨迹。非4倍数窗口仅对RVQ输入重复最后一帧；latent数为 `ceil(T/4)`。

编码器是非因果卷积，按完整窗口编码；5Hz是 token 时间栅格，不等于只看过去
0.2秒。批处理按 padded length 分组，避免为短窗口补到长窗口长度改变边界条件。
将来在线生成 tokens 或流式编码时，需要单独处理上下文和窗口切换。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  dataset/bones-seed_prepare/g1_validate_rvq_export.py \
  /EFS/users/zhaohui.yang/dataset/momask-rvq-tracking/rot6d68_ep499_v1
```

验证器检查全部分片 SHA256、split/ID 覆盖、offsets、形状、范围、有限值，
并对**全部** embedding 做 token/codebook 一致性检查。默认每个split按长度选3个
窗口，单条重新编码并与原始 CSV 对齐，验证姿态。报告写到输出目录的
`validation_report.json`。可用 `--samples-per-split` 增加抽查，挂载路径不同可传
`--data-root`、`--bones-root`、`--checkpoint` 覆盖原路径。

## 后续 MimicLite 接入边界

该产物是独立的 RVQ 条件数据接口，**尚不是 MimicLite 原生 motion 文件**。
下一阶段需要按锁定版本的 mjlab/G1 资产完成关节映射、原始参考轨迹插值、FK/速度
构建和动作加载器接入；Actor 使用本体状态与 latent，奖励/Critic可读原始参考。
重置和随机起始帧必须同步 latent 索引/相位。验证集和测试集不参与训练采样。

火山训练容器只需读取此输出目录，不必每次重跑RVQ。挂载路径可变化，分片在
manifest 中使用相对路径；镜像版本、RL代码commit和输出目录需在训练任务中固定。

## 2026-09-30 实测结果

全量编码完成：187,777 个窗口、15,925,244 帧、4,044,952 个 token 时间步，
93 个分片，约 11 GiB。`validation_report.json` 的 `passed=true`，全部 embedding
在 float32 求和理论误差界内与 codebook 一致；每个 split 按长度抽查3个窗口，
单条编码 token 一致，原始 CSV 最大绝对误差小于 `1e-4`。这验证数据接口，
不表示 RVQ 重建质量或 RL 跟踪质量达到任何阈值。

首次编码过程中审查发现并修复了“续跑再次中断后丢失未访问分片校验记录”的
manifest 写入问题。编码和参考恢复函数未变，数据分片未改。为保留真实来源，
输出中的 `producer_manifest.json`、`producer_source/g1_preprocess_rvq.py` 保留
实际生产版本及原始哈希，`producer_source/writer_fix.diff` 记录修复；当前 manifest
的 identity 对应可安全续跑的新写入器，`writer_maintenance` 记录此兼容迁移。
