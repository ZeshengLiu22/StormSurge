# Configuration overview

训练配置按实验内容命名。四组实验都覆盖 CBBT、Lewes、Battery、Boston；
共享 NCEP Grid4_New 数据、GraphSAGE + Transformer backbone、24 小时历史、
width 128、batch size 256、4 步梯度累积、学习率 5e-3、300 epochs 和 seed 42。

| 目录 | 状态 | 实验内容 | 配置数 |
| --- | --- | --- | ---: |
| [baseline_ablation](baseline_ablation/README.md) | **Temporarily archived（暂时归档）** | S0/D0–D3：Single / Dual、Tail、Amplitude 基础消融 | 20 |
| [wqe](wqe/README.md) | **Archived（已完成并归档）** | W1/W2/W3：WQE 用于 global prediction、excess 分支或两者 | 12 |
| [s0_refresh](s0_refresh/README.md) | 未归档 | Single-head：G0/G1 × T0/T1/T2 | 24 |
| [wqe_factorial_multickpt](wqe_factorial_multickpt/README.md) | 未归档 | Dual-head：G0/G1 × E0/E1 × T0/T1/T2 | 48 |

归档状态用于记录实验整理进度；配置保留供复现和对照使用。
推理配置位于 [configs_infer](configs_infer/)。

## Directory names

| 原目录名 | 现目录名 |
| --- | --- |
| `current` | `baseline_ablation` |
| `wqe` | `wqe` |
| `0924_s0_refresh` | `s0_refresh` |
| `0922_wqe_factorial_multickpt` | `wqe_factorial_multickpt` |

生成器默认将基础消融写入 `configs/baseline_ablation`；CLI 的 `--family current`
保留原有含义。`--family wqe`、`--family s0_refresh` 和 `--family wqe_factorial_multickpt` 写入同名目录。
各目录 README 包含具体配置含义和使用方式。

## G / E / T 和 0 / 1 / 2 的含义

文件名与 run name 中的 G、E、T 是三个独立的 loss 选择；字母后的数字表示
对应选项。MSE 是均方误差（mean squared error），WQE 是加权分位数–期望分位数
损失（weighted quantile–expectile loss）。

| 字母 | 全称与作用 | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| **G** | **Global**：最终预测相对真实值的全局损失，覆盖所有目标小时 | 全局 MSE | 全局 WQE | 不使用 |
| **E** | **Excess**：Dual 原始超额分支的损失，监督乘 gate 之前的 raw excess | 超额分支 MSE | 超额分支 WQE | 不使用 |
| **T** | **Tail**：最终预测在严格极端小时 `y > tau` 上的附加损失 | 关闭，权重 0 | Tail-MSE，权重 0.025 | Tail-WQE，权重 0.025 |

**G0/E0 表示启用 MSE，只有 T0 表示关闭。** G/E 只有 0、1 两档；T 有 0、1、2 三档。
E 监督 GT Event Window 内所有 horizon 的原始超额值；T 只选严格超过固定 TRAIN
Q95 阈值的小时，并使用固定 TRAIN `q_H` 归一化。

例如，`G1_E0_T2` 表示全局 WQE + 超额分支 MSE + 权重 0.025 的 Tail-WQE。
Single 没有超额分支，只写 G/T：`G0_T1` 表示全局 MSE + 权重 0.025 的 Tail-MSE。
Dual 使用完整 G/E/T，保留 body/excess/gate 权重 1/2/0.5。
因此 Single 每站 2×3=6 个配置，Dual 每站 2×2×3=12 个配置。

## Experiment relationships

- `baseline_ablation`：S0 是 Single MSE；D0 是 Dual 基础；D1 加 Tail-MSE
  （权重 0.025）；D2 加 amplitude loss（权重 0.003）；D3 同时加两者。
- `wqe`：以 D0 为对照，W1 只切换 global loss，W2 只切换 excess loss，
  W3 同时切换为 WQE；Tail 和 amplitude 关闭。
- `wqe_factorial_multickpt`：文件名中的 G/E 表示 global/excess loss，
  `0=MSE`、`1=WQE`；T 表示独立 Tail loss：`0=关闭`、`1=MSE`、`2=WQE`，
  开启时权重均为 0.025。
  按 head/loss 设置，`G0_E0_T0` 对应 D0，`G0_E0_T1` 对应 D1，
  `G1_E0_T0`、`G0_E1_T0`、`G1_E1_T0` 分别对应 W1、W2、W3。
- `s0_refresh`：补齐相同 global loss / Tail 设置下的 Single 对照；
  所有配置均为 `HEAD_TYPE=single`、`DUAL_LOSS=0`。每站 6 个 G×T 配置。
  原 `S0_Single_Tail_WQE` 是 global WQE + Tail-MSE，对应 `G1_T1`；
  新增 `G0_T2` 和 `G1_T2` 才使用 Tail-WQE。

WQE 是 weighted quantile–expectile loss。`EXCEEDANCE_LOSS_MODE` 独立选择
Tail-MSE 或 Tail-WQE；两者使用相同的严格 `y > tau` TRAIN Q95 mask 和固定 TRAIN
`q_H` 归一化。Tail-WQE 与 global/excess WQE 共享参数和 `y_std`。完整定义见 [Losses](../docs/LOSSES.md)。

`multickpt` 表示同一次训练按四种 VAL 标准保存 checkpoint：overall、exceedance、
aligned_peak、bea。BEA（Balanced Event-Aware）固定为
`0.50*AllRMSE + 0.25*ExceedanceRMSE + 0.25*GTAlignedPeakRMSE`。
当前共享训练流程对所有实验族
采用该机制；这些配置的主结果均使用 overall。见
[Checkpoint selection](../docs/CHECKPOINT_SELECTION.md)。

## Results locations

配置目录重命名后，输出路径沿用历史设置，便于查找已有结果。

| 配置目录 | 输出根目录 |
| --- | --- |
| `baseline_ablation` | `./All_Results` |
| `wqe` | `/home/exouser/media/share/PACT/WQE_Results` |
| `s0_refresh` | `/home/exouser/media/share/PACT/0924_s0_refresh` |
| `wqe_factorial_multickpt` | `/home/exouser/media/share/PACT/All_results_0922_wqe_factorial_multickpt` |
