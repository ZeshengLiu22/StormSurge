# Configuration overview

更新于 2026-10-06。顶层保留 Single 权重扫描与 loss ablation、CMIP6 训练、
NCEP future transfer 和跨数据集推理配置；历史训练与旧推理配置已移入
[Legacy_Configs](Legacy_Configs/)。

## 当前配置入口

| 目录 | 用途与范围 | 配置数 |
| --- | --- | ---: |
| [single_tail_episodepeak_4x4](single_tail_episodepeak_4x4/README.md) | **ACHIEVED**：四站 NCEP Single；Tail × EpisodePeak 权重扫描及完整附加 loss ablation，64/64 已完成 | 64 |
| [CMIP6_Boston_QuickCheck_0929](CMIP6_Boston_QuickCheck_0929/README.md) | Boston：5 个 GCM × 2 个时间划分 × 4 种 loss 组合 | 40 |
| [CMIP6_Battery_QuickCheck_0929](CMIP6_Battery_QuickCheck_0929/README.md) | Battery：同一套 CMIP6 对照与消融 | 40 |
| [CMIP6_CBBT_QuickCheck_0929](CMIP6_CBBT_QuickCheck_0929/README.md) | CBBT：同一套 CMIP6 对照与消融 | 40 |
| [CMIP6_Lewes_QuickCheck_0929](CMIP6_Lewes_QuickCheck_0929/README.md) | Lewes：同一套 CMIP6 对照与消融 | 40 |
| [NCEP_future_transfer](NCEP_future_transfer/README.md) | 四站历史 NCEP 源模型；TRAIN/VAL-only，用于未来 CMIP6 transfer | 4 |
| [cross_dataset_infer](cross_dataset_infer/README.md) | Boston Source → Target 推理；past-only 36 对、future-year 30 对，当前为开发验证配置 | 66 |

上表配置数表示实验设计规模，不自动代表训练或推理已完成；4×4 的完成状态已单独核实。
当前这些训练家族均采用 Single、PACT (`perceiver3`)、GraphSAGE + Transformer、
24 小时历史、hidden width 128、batch size 256、4 步梯度累积、学习率 5e-3、
300 epochs 和 seed 42。数据来源、时间划分和主 checkpoint selector 以各家族配置为准。

## Single Tail × EpisodePeak 4×4：权重扫描与完整消融

```text
GlobalMSE + lambda_T * TailMSE + lambda_P * EpisodeGTAlignedPeakMSE
```

- Global loss 始终为 **MSE**，权重为 1。
- Tail-MSE weights：**{0, 0.0125, 0.025, 0.05}**，对应 `T0000 / T0125 / T0250 / T0500`。
- EpisodeGTAlignedPeak-MSE weights：**{0, 0.0025, 0.005, 0.01}**，对应 `EP0000 / EP0025 / EP0050 / EP0100`。
- 文件名的 T/EP 数值是实际权重乘以 10,000。

两个因子都包含 0，因此网格同时覆盖权重扫描和以下四类 loss ablation：

| 类别 | Tail 权重 | EpisodePeak 权重 | 每站配置数 | 四站合计 |
| --- | --- | --- | ---: | ---: |
| MSE Only | 0 | 0 | 1 | 4 |
| Tail Only | 非零 | 0 | 3 | 12 |
| Peak Only | 0 | 非零 | 3 | 12 |
| Tail + Peak | 非零 | 非零 | 9 | 36 |
| 合计 | | | **16** | **64** |

**Tail Only / Peak Only 仍保留 Global MSE**；Only 指两个附加项中仅开启其中一个。
因此，针对 Tail 与 EpisodePeak 两个附加 loss 的消融已经齐全。
Tail 覆盖严格 `GT > TRAIN Q95` 小时；EpisodePeak 每个物理 TRAIN episode
只监督一个 GT peak 时间戳。WQE、Dual、forecast-window peak 和 timing loss 不属于此网格。

2026-10-06 只读核查确认：CBBT、Lewes、Battery、Boston 各 16 组，**64/64**
均有最终 VAL/TEST summary、checkpoint comparison 和完成日志。
结果目录下的 `diagnose/run_status.csv` 亦将全部 64 个单元标为 `Complete`。
这是已完成的 NCEP 研究，配置与结果继续保留供分析和复现。

该家族训练时的 primary selector 为 **exceedance**（最小 VAL ExceedanceRMSE），
secondary 为 **overall**（最小 VAL AllRMSE）；两个 checkpoint 都做最终 VAL/TEST 评估。
`T0000_EP0000` 对应早期 `G0_T0`，`T0250_EP0000` 对应早期 `G0_T1`。

完整网格见 [64 行 manifest](single_tail_episodepeak_4x4/manifest_all.csv)，另有
[Boston + Lewes](single_tail_episodepeak_4x4/manifest_boston_lewes.csv) 和
[CBBT + Battery](single_tail_episodepeak_4x4/manifest_cbbt_battery.csv) 各 32 行子集。
本家族没有批量 launcher；单配置使用仓库根目录的 `train.sh`，例如：

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0500_EP0100.sh
```

方法定义见 [EPISODE_PEAK](../docs/EPISODE_PEAK.md)、[LOSSES](../docs/LOSSES.md)、
[CHECKPOINT_SELECTION](../docs/CHECKPOINT_SELECTION.md) 和 [METRICS](../docs/METRICS.md)。

## CMIP6 对照与消融

四个 `CMIP6_<station>_QuickCheck_0929` 目录各有 40 个配置，共 160 个。
每站包含 AWI、CNRM、EC_EARTH、MPI、MRI 五个 GCM，以及两种时间划分：

| 子目录 | 数据目录 | TRAIN / VAL / TEST 年组数 | TEST 范围 |
| --- | --- | --- | --- |
| `past_only` | `Data/Grid4_New_PastOnly` | 22 / 7 / 7 | `2008_2009`–`2014_2015` |
| `future_year` | `Data/Grid4_New` | 30 / 6 / 30 | `2070_2071`–`2099_2100` |

每个 GCM、每种划分均保留以下四组；它们也是 NCEP 4×4 网格中的四个角点：

| 文件名 token | Loss 组合 |
| --- | --- |
| `T0000_EP0000` | MSE Only |
| `T0500_EP0000` | Global MSE + 0.05 Tail-MSE（Tail Only） |
| `T0000_EP0100` | Global MSE + 0.01 EpisodePeak-MSE（Peak Only） |
| `T0500_EP0100` | Global MSE + 0.05 Tail-MSE + 0.01 EpisodePeak-MSE |

Boston 最初的批次只有 MSE / Tail+Peak，之后补入 Tail Only / Peak Only；当前 manifest
已包含全部四组。Boston 的原 `launch_all.sh` 仍只描述最初批次，新增消融入口是
[qsub_ablations.txt](CMIP6_Boston_QuickCheck_0929/qsub_ablations.txt)。
其他三站的 `launch_all.sh` 覆盖各自 40 个配置。具体提交方式见各站 README。
CMIP6 家族以 **overall** 为 primary，同时保留 `best_overall.pt` 和 `best_exceedance.pt`。

## Future transfer 与当前推理配置

[NCEP_future_transfer](NCEP_future_transfer/README.md) 为四站提供 `T0500_EP0100`
源模型配置：30 个 TRAIN 年组（`1979_1980`–`2008_2009`），6 个 VAL 年组
（`2009_2010`–`2014_2015`），**没有源域 held-out TEST**。
训练报告中兼容导出的 TEST 是 VAL mirror；真实未来评估通过外部 CMIP6 推理完成。
主 selector 为 **overall**。

[cross_dataset_infer](cross_dataset_infer/README.md) 是当前跨数据集推理入口，
目前配置范围为 **Boston 开发验证**：

- `past_only`：NCEP + 五个 GCM，6 sources × 6 targets = 36 对。
- `future_year`：同样六个 sources，仅五个 GCM 作为 targets，6 × 5 = 30 对。
- 全部使用 Single / 24 h / `T0500_EP0100` 和固定的 `best_overall.pt`。
- 保留 source checkpoint 的归一化和 TRAIN Q95，并严格检查请求的目标年份。

每个 pair config 对应一次 `infer.sh` 调用；`run_group.sh` 在一个 GPU 上顺序运行
同一 source 的多个 targets。这里的 66 个配置不表示四站正式推理矩阵均已完成。

```bash
DRY_RUN=1 bash infer.sh configs/cross_dataset_infer/past_only/NCEP/NCEP_to_AWI.sh
```

## Legacy_Configs：历史配置归档

| 当前目录 | 历史用途与状态 | 配置数 |
| --- | --- | ---: |
| [s0_refresh](Legacy_Configs/s0_refresh/README.md) | **ACHIEVED**：Single G0/G1 × T0/T1/T2；已完成的早期 Single 对照 | 24 |
| [wqe](Legacy_Configs/wqe/README.md) | **ACHIEVED / ARCHIVED**：WQE 用于 global、excess 或两者的 placement study | 12 |
| [wqe_factorial_multickpt](Legacy_Configs/wqe_factorial_multickpt/README.md) | **ACHIEVED / LEGACY**：Dual G × E × T factorial | 48 |
| [baseline_ablation](Legacy_Configs/baseline_ablation/README.md) | **LEGACY / ARCHIVED — completion unverified**：S0/D0–D3 基础消融 | 20 |
| [configs_infer](Legacy_Configs/configs_infer/) | **LEGACY**：Battery / Dual / 12 h / `P3_Best` 旧推理模板 | 12 个入口 + 1 个 common |

**ACHIEVED** 表示有证据支持该研究已完成；**LEGACY / ARCHIVED** 表示历史用途，
不单独证明完成状态。2026-09-27 的核查记录确认 S0 24/24、Dual 48/48、WQE 12/12
均有最终 VAL/TEST summary 和完成日志；baseline 原正式实验的完成状态未核实。

旧 `configs_infer` 文件的 Git 最后修改日期为 2026-09-10，checkpoint 路径仍是
`./Inference_Checkpoints/*_Battery_P3_Best.pth`。新推理应从 `cross_dataset_infer` 查看。
根目录 `infer_multi.sh` 的无参数默认路径仍是迁移前的
`configs/configs_infer/infer_multi_config_NCEP.sh`；如需复用旧模板，须显式传入
`configs/Legacy_Configs/configs_infer/` 下的配置路径。

旧生成器 `tools/generate_configs.py` 的默认输出目录也尚未随归档迁移：
`--family current` 仍写入 `configs/baseline_ablation`，其他旧 family 写入
`configs/<family>`。重生成时应通过 `--output` 显式选择目标目录。
历史 manifest、配置和子目录文档可能保留迁移前的路径，复用时需按当前目录核对。

### 历史 G / E / T 编码

下表解释旧 factorial 的类别编码；4×4 的 `T0250` / `EP0100` 则直接编码权重。

| 字母 | 作用 | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| **G** | Global：最终预测的全局 loss | MSE | WQE | 不使用 |
| **E** | Excess：Dual 的 raw excess 分支监督 | MSE | WQE | 不使用 |
| **T** | 严格极端小时上的附加 Tail loss | 关闭 | Tail-MSE，权重 0.025 | Tail-WQE，权重 0.025 |

**G0/E0 表示启用 MSE，只有 T0 表示关闭。** Single 无 excess 分支，只写 G/T；
Dual 使用 G/E/T。`G1_E0_T2` 表示全局 WQE + excess MSE + 0.025 Tail-WQE。
WQE 是 weighted quantile–expectile loss；Tail-MSE/WQE 使用相同的严格 TRAIN Q95
mask 和固定 TRAIN `q_H` 归一化。

历史对应关系：S0 = Single MSE；D0 = Dual 基础；D1 在 D0 上加 0.025 Tail-MSE；
D2 加 0.003 amplitude loss；D3 同时加两者。W1/W2/W3 分别只将 global、只将
excess、或同时将两者切为 WQE，Tail 与 amplitude 关闭。
在 Dual factorial 中，`G0_E0_T0` / `G0_E0_T1` 对应 D0 / D1，
`G1_E0_T0` / `G0_E1_T0` / `G1_E1_T0` 对应 W1 / W2 / W3。

## 结果位置

当前训练与推理配置的输出根目录为 `/home/exouser/media/share/PACT/FormalRuns_0925`。
表中前四行为配置记录的写入路径，不保证当前已有结果；历史 S0/Dual 列出现有归档位置。

| 配置家族 | 结果根目录下的相对位置 |
| --- | --- |
| `single_tail_episodepeak_4x4` | `single_tail_episodepeak_4x4/`；诊断在其 `diagnose/` 下 |
| `CMIP6_<station>_QuickCheck_0929` | 同名目录的 `past_only/` 与 `future_year/` |
| `NCEP_future_transfer` | `NCEP_future_transfer/` |
| `cross_dataset_infer` | `CrossDatasetInference_dev/past_only/` 与 `CrossDatasetInference_dev/future_year/` |
| 历史 `s0_refresh` | `Legacy_Results_No_Use/S0/` |
| 历史 `wqe_factorial_multickpt` | `Legacy_Results_No_Use/Dual/` |

历史配置中记录的 `FormalRuns_0925/S0`、`FormalRuns_0925/Dual` 是迁移前输出路径。
WQE 配置仍记录 `/home/exouser/media/share/PACT/WQE_Results`，baseline 仍记录
`./All_Results`；这两个路径当前不存在，不能据此认定历史结果仍存放在那里。
