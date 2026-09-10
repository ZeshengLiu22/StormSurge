# StormSurge 与原版 Emulator：当前差异、影响与验证范围

本报告更新至 **2026-09-10 的当前工作树**，包含上一轮 `p_mean` 删除和本轮 dual 事件阈值、真实事件比例、机制消融与推理诊断改动。它替代首次迁移快照中的支持状态和还原建议。

实际目录是 `/media/volume/PACT-Data/StormSurge`；`/home/exouser/media/volume/PACT-Data/StormSurge` 通过 `media` 链接指向同一目录，所有修改都落在这个独立仓库。数据仍由外部目录提供，不随源码复制。

原版比较对象为本地 `Emulator` 的 [bb62a22](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/tree/bb62a2297a0d37a35db5bff352ee815e05676c93)。它相对上次基准 `3d4be39` 仅提交了此前本地 `.gitignore` 的3行变化，模型和训练源码没有变化。StormSurge 的基础提交仍是 `ce697f9`；这里比较的是包含后续修改的工作树。逐文件内容、原版 Git 状态和生成时间以 [comparison_summary.json](audit/comparison_summary.json)、[source_inventory.json](audit/source_inventory.json)、[CSV](audit/source_inventory.csv) 和 [完整 diff](audit/original_to_current.diff) 为准。

本次扫描 **203 个路径**：134 个逐字相同、35 个修改、25 个新增、9 个删除。第8节列出全部 **69 个差异路径**。

核心结论是：GraphSAGE/CNN 基础空间拓扑、baseline 的默认路径、原训练扫描和主要训练协议继续保留。PACT 仍包含此前确认的稳定性数学与新 dual decoder；本轮完善事件定义、消融和评估接口，没有重写 backbone，也没有改变 excess loss 的 whole-batch 归约。当前实现的正确性不等于已证明 dual 优于 single，正式精度结论仍需匹配条件的独立训练。

## 1. 本轮报告建议的落实情况

| 建议 | 当前实现 | 实验含义 |
|---|---|---|
| 使用真实 TRAIN 事件比例 | `fit_loss_thresholds()` 统计严格 `max_h(Y_h) > tau_phys` 的事件数与比例 `q_E`，只读 TRAIN indices | percentile 处的 ties 不再被假定为恰好5%事件；gate bias 使用真实比例 |
| 显式保存物理阈值与比例 | checkpoint/summary 的 `dual_metadata` 包含 `tau_phys`、`event_prior`、`gate_init_prior`、事件数、TRAIN 窗口数、百分位、事件定义和模式 | 推理和离线校准可追溯事件定义，不必从归一化阈值反推 |
| 解耦 dual 与 tail-loss 阈值 | 新增 `EXCEEDANCE_PERCENTILE` / `--exceedance_percentile`，默认95；`tail_frac` 仅控制最终预测的 tail loss | 改变 tail-loss 子集不会改变 decoder 的事件语义 |
| 显式机制消融 | `DUAL_ABLATION` / `--dual_ablation` 的5种模式，默认 `none`；新增4份引用 Stable Dual 的配置 | 正式模型继续要求完整监督；只有命名消融会关闭指定项 |
| 固定 gate | `fixed_gate` 使用 `p(X) ≡ q_E`，概率作为 buffer 保存，不建立 gate MLP | 可检验输入相关 gate 的贡献；参数预算少一个 gate MLP，应单独注明 |
| 保持 L_r 归约 | 继续对事件窗口 mask 后按整个 `B × K` 平均 | 不除以 batch 内事件数，也不额外除以 q_E；没有把相对权重默认放大约20倍 |
| 推理专项诊断 | `--dual_diagnostics` / `DUAL_DIAGNOSTICS=1` 输出独立 NPZ/JSON | 支持概率和分支分析；常规 Train/Val epoch 不收集这些诊断数组 |

默认 `exceedance_percentile=95` 与 `tail_frac=.05` 对应相同的阈值数值，但 gate 初始化现在依赖实际 q_E。因此默认设置下的训练随机轨迹和预测不承诺与本轮修改前逐位相同。阈值解耦本身不改变默认事件定义。

## 2. Dual head 与 dual loss 的实际定义

令 τ 为 TRAIN 窗口最大值的指定百分位。逐输出标签与窗口事件为：

\[
C_{ih}=\min(Y_{ih},\tau),\qquad D_{ih}=(Y_{ih}-\tau)_+,
\qquad E_i=\mathbf 1\{\exists h:Y_{ih}>\tau\}.
\]

μ_h、σ_h 仍仅由 TRAIN 拟合，归一化阈值为 τ′_h=(τ−μ_h)/σ_h。模型保持：

\[
b'_{ih}=\tau'_h-\operatorname{softplus}(\tau'_h-a_{ih}),\quad
r'_{ih}=\operatorname{softplus}(d_{ih}),\quad
\hat Y_{ih}=\mu_h+\sigma_h(b'_{ih}+p_i r'_{ih}).
\]

物理 body 为 μ+σb′，物理 excess 为 σr′；excess 不加 μ。两个回归 MLP 分别处理每个 horizon context；可学习 window gate 对 horizon context 求均值后进入 gate MLP。预测 forward 不读取未来标签，E 只用于训练监督或预测完成后的评估。

与原版 `base + sigmoid(alpha) × sigmoid(gate) × tail` 相比，当前删除额外 alpha、tail tanh clip 和 horizon gate 分支，加入 body 上界、excess 非负约束与显式分支监督。gate/excess 末层权重仍初始化为0，excess 初始归一化增量仍为.1。

真实 `q_E = event_count / train_windows` 被原样保存。学习 gate 仅在生成有限 logit 时将其截到 `[1e-6, 1−1e-6]`；固定 gate 使用原始比例，允许精确为0或1。全相同 TRAIN 标签产生零事件时不会假报5%。这种无事件数据无法提供事件 excess 的直接学习信号，应在实验分析时结合保存的事件数判断。

用物理 b、r 表示分支输出，监督为：

\[
L_b=\frac1{BK}\sum_{i,h}(b_{ih}-C_{ih})^2,\qquad
L_r=\frac1{BK}\sum_iE_i\sum_h(r_{ih}-D_{ih})^2,
\]
\[
L_g=\operatorname{BCEWithLogits}(g,E),\qquad
L=L_{pred}+\lambda_bL_b+\lambda_rL_r+\lambda_g\operatorname{mean}_h(\sigma_h^2)L_g.
\]

事件窗口内未超阈的 horizon 仍以 D_h=0 监督 excess；非事件窗口的直接 excess 辅助梯度为0。无事件 batch 的 L_r=0，归约分母始终是 BK。总体上该项是 q_E 乘条件事件风险；当 q_E>0 时不改变独立分支的总体最优解，但会改变有限容量共享 backbone 的联合优化权衡，λ_r 的尺度需要实验说明。

| 模式 | 有效 body/excess/gate 权重 | Gate 是否学习 | `dual_loss` |
|---|---|---|---|
| `none` | 三项均保留；零值纠正为1，正值保留 | 是 | 强制1 |
| `no_gate_bce` | body、excess保留；gate权重0 | 是，仍接受最终预测梯度 | 1 |
| `no_excess_loss` | body、gate保留；excess权重0 | 是 | 1 |
| `no_branch_supervision` | 三项均0 | 是 | 0，仅使用所选最终预测loss |
| `fixed_gate` | body、excess保留；gate权重0 | 否，p=q_E | 1 |

默认模式的误关闭仍产生一条带时间戳的 WARNING，并在训练前纠正。消融模式将指定项明确置0，其余启用项仍不能被权重0意外关闭。有效参数写入运行 JSON 和 checkpoint；single/baseline 拒绝非 `none` 的 dual 消融。没有 gate BCE 或完整分支监督的模式，不能直接宣称其 sigmoid 已具备校准概率语义。

`ModelConfig.dual_ablation` 决定 checkpoint 重建，`LossConfig.dual_ablation` 决定辅助项。固定 gate 没有可学习参数，`gate_logits=None`；`ForecastOutput.gate_probability` 对所有 dual 模式都提供实际使用的概率。模型与 loss 的 fixed-gate 设置不一致时直接报错。

最终预测 loss 继续保留10种 mse/wmse/tail/slope 组合。Tail loss 使用独立的 `tail_threshold` 和 `max(Y) >= tail_threshold`；dual 事件使用严格 `>`。原 WMSE/slope 的 pointwise 阈值仍由 `wmse_q` 拟合。`loss_thresholds` 显式保存这些标量。三类阈值都来自 TRAIN，没有从 Val/Test 拟合。

## 3. Backbone、baseline 与训练协议

| 项目 | 原版与当前关系 | 当前决定 |
|---|---|---|
| H=0 baseline | GraphSAGE/CNN → spatial mean pool → dropout → Linear(K) 保持 | 保留；不含PACT attention/temporal/dual |
| H>0 baseline | 各时刻编码/pool → 单层LSTM → 最后时刻 → Linear 保持 | LSTM原来就存在；自定义宽度差异见第7节 |
| GraphSAGE/CNN | SAGEConv、LeakyReLU(.1)、dropout和CNN 3×3/padding1层保持；CNN默认中间宽度29 | 保留模块拆分与合法矩形网格计算 |
| PACT总连接 | spatial → station-query attention → lag/temporal → horizon attention → head 保持 | 没有重写backbone |
| 相对lag/H=0 PACT | 当前forcing为lag0；H=0给context加回horizon query | 原有能力，不作为新增方法贡献 |
| station/head前norm | 当前新增FP32、无affine的LayerNorm | 属于此前确认的稳定性数学，保留 |
| Transformer/MLP | 原PostLN改为PreLN和 `x + exp(log_gain) × branch(LN(x))` | 每分支gain初始.1并由Adam学习；不等于LR缩小10倍或clip |
| LSTM/GRU temporal | `LayerNorm(x + dropout(RNN(x)))` 保持 | 其前后仍受到PACT新增norm影响 |
| Head激活 | 原ReLU改为LeakyReLU(.1) | 保留负区间梯度；不宣称单项已解决所有collapse |
| Adam | 默认betas/eps、weight_decay=1e-5保持 | 没有换AdamW或改变学习率扫描 |
| Cosine/ROP | linear warmup＋cosine由LambdaLR等价表达；ROP及原参数保留 | scheduler选择和科学口径不变 |
| DDP seed/sampler | `seed+rank`；DistributedSampler默认seed0并逐epoch set_epoch | 已恢复的原协议继续保持 |
| X/Y统计 | 原FP32平方、FP64求和/归约和FP32矩/方差次序保留 | TRAIN-only；robust/mag仅抽样统计，不截小模型输入图 |
| 统计抽样/augmentation | nodes_per_graph<=0仍自动最多256；概率1直接生成扰动、不额外抽概率 | 保留已恢复的局部RNG顺序 |
| 归一化存储 | 改用新tensor赋值 | 避免污染共享的原始图存储 |
| Val All/Peak | All保留DDP sampler padding；Peak去重并恢复数据集顺序 | best epoch/ROP继续使用原Val口径；复用forward |
| 梯度累积 | 同组microbatch等权，末组按实际microbatch数 | 不改为按不同microbatch样本数加权 |
| 最终Test/Train报告 | Test改为rank0单次遍历真实样本；Train整理在线样本误差并报告Peak5 | 与原DDP补齐Test/旧Train归约可能不同，需标明口径 |
| 常规日志 | 每轮Train/Val × All/Top5% × RMSE/MAE，另记wall time | 不新增gate/attention/variance的每轮统计 |
| 运行选项 | BF16/TF32 profile保持；确定性可选默认0；clip可选默认0 | 没有自动重训、回滚、跳过异常epoch或特殊残差LR |

原版为诊断执行过额外的 train-mode forward/DataLoader 遍历，当前已移除。即使恢复了seed、统计公式和augmentation局部RNG，整轮训练随机数轨迹仍可能不同。不要为复现无用计算而恢复额外forward；方法对照应在当前同一代码/环境/预算下从头训练。

## 4. 推理、诊断和产物

默认推理仍按 checkpoint 保存的精确 test tags 选择样本；缺失样本或站点时报错。`--test_root_dir`/`--scope all` 显式选择外部全量人口，`--years` 再过滤。架构与H参数用于核对checkpoint；同站点使用保存的station features，换站点才读取新JSON。原逐年、ALL/past/future报告及排除2014_2015的平均年耗时已保留；2014_2015仍参与精度统计。

| 产物/开关 | 当前行为 |
|---|---|
| checkpoint | `model_config/model_state/normalization/station_feat/split_tags/training_config`，新增 `loss_thresholds/dual_metadata` |
| config/summary | 记录有效消融设置；summary也记录TRAIN阈值/比例与最终测试范围 |
| `--save_npz` | 原 `predictions.npz` 和 `preds_*_ALLYEARS.npz` 仍仅含 y_true/y_pred/tags |
| `--dual_diagnostics` | 独立导出 `dual_diagnostics.npz` 与 `dual_diagnostics.json`，即使不启用 `--save_npz` 也可导出 |
| 诊断NPZ | 顺序对齐的truth/prediction/tags、gate_probability、body_phys、excess_phys、contribution_phys、event、τ/q_E与模式 |
| 诊断JSON | 总体/逐年Brier、stepwise average precision、trapezoidal PR-AUC、10个等宽可靠性分箱、正/负事件gate直方图、事件/非事件窗口RMSE、body RMSE与条件事件excess RMSE |
| 未定义指标 | 无正事件时PR指标为null；空分箱/空条件子集为null；概率0/1正确归入端点分箱 |
| 两个shell入口 | `DUAL_DIAGNOSTICS=1`透传，写入 `infer_config_used.sh`；`command_used.sh`保存确切命令 |

PR-AUC的插值定义会影响数值，所以JSON分别命名两种面积，不能把它们混作同一指标。event 标签在预测后使用固定TRAIN物理阈值生成，不进入模型forward。诊断复用原推理forward，常规训练不保留这些数组；开启后会增加推理导出内存和磁盘使用。数组已足够绘制可靠性图、分支轨迹与贡献图，当前不自动绘图。

旧原版checkpoint仍不兼容；含已删除 `p_mean` 配置字段的中间checkpoint也不直接兼容。新的诊断导出要求存在 `dual_metadata`，不会从测试数据补拟合阈值。固定gate/其他消融由保存的配置重建，推理不能临时改变训练模式。

## 5. 配置、预处理与 p_mean 清理

原112份具体训练profile逐字保留；原条件扫描的390组组合仍通过验证。两个training common移除 `p_mean` 并接入当前dual接口，inference common新增默认关闭的诊断开关。Stable Dual继续source Stable Single，只改变head相关设置；四份新消融配置再source Stable Dual，仅改变命名机制和结果目录。配置在 [`configs/dual_ablations`](../configs/dual_ablations)。

`p_mean` 已从CLI/config、模型编码器、head扩展输入维度、数据View、归一化统计、训练/推理日志、NPZ额外均值字段及图属性传递中整体删除。原先为该功能新建的 `pressure.py` 不再存在。**forcing 的压力去空间均值仍保留**，五通道布局仍为 `[u,v,p',lon,lat]`。历史审计日志中的相关文字只记录当时状态，不表示当前继续支持。

五个CMIP6入口仍保留各自路径、NPY/NPZ与3h→6h抽样，但因删除额外均值字段，已不能再称“与原脚本逐字相同”。generic CMIP6入口继续作为额外CLI；NCEP保持main/CLI参数化、原默认值和NPY/NPZ。simulation保持原mesh/station/时间处理与默认2005，额外接受years/argv。time alignment保留fixed315/peryear、缺CSV时间列的显式fallback、缺年skip；main仍默认只跑peryear。历史窗口、标签时间对齐和网格形状都有回归测试。

两个环境YAML、LICENSE和station JSON继续保留原文件内容；环境配方是PyTorch2.8+cu128，不等于本轮测试安装环境。数据、训练结果和checkpoint未复制或改写。相对 `Emulator` 的 `.gitignore` 差异属于仓库管理范围，不是模型变化。

## 6. 验证证据与限度

本轮 **50/50 unittest通过**，完整日志见 [dual_review_tests.txt](audit/evidence/dual_review_tests.txt)，运行范围和源码哈希见 [dual_review_validation.json](audit/evidence/dual_review_validation.json)。覆盖此前的114份训练配置、4份消融配置、390组条件扫描、所有5种dual模式的短训练/保存/重载/推理、两个shell推理入口、分支梯度、零事件/ties、独立阈值、标签独立性、物理重建及诊断指标/顺序。常规预测在诊断开关前后相同；Python与shell语法检查通过；源码/脚本/文档的增量diff空白检查通过，原始diff快照保留其空行context标记，不纳入该空白检查。

本轮使用 **CPU、PyTorch2.6+cu124、PyG2.7**，数据预处理依赖来自现有验证环境。当前GPU驱动不可用，因此没有重跑GPU/BF16/NCCL，也没有做长程训练、跨seed稳定性或正式精度比较。短样本检查证明接口与数学契约能够贯通，不能证明概率已校准、机制消融已得出结论或dual一定优于single。

`publication_tests.txt`、`model_equivalence.json`、`gpu_equivalence.json`、`protocol_*`、`smoke_gpu/ddp`等仍是迁移前/首次发布的历史证据。其关于原版baseline、对应稳定性数学、统计/RNG/loss和DDP的测量保留，但不作为本轮event-prior/消融接口的GPU或DDP验证。来源界限见 [evidence/README.md](audit/evidence/README.md)。`v2_source_hashes.json`与`migration_verification.json`也继续标识首次迁移，不代表当前工作树。

训练CLI声明清单已重新生成：[argparse_comparison.json](audit/argparse_comparison.json)。当前有73项声明保持、6项删除、12项新增、3项声明改变；其中删除包括3个旧残差head参数和3个 `p_mean` 参数，新增包括独立百分位和命名消融。此清单只比较声明，解析后的模式纠正和语义以本报告与测试为准。

## 7. 本轮没有处理的其他差异

下列是此前重构相对原版仍存在的差异，不属于本次dual报告的修复目标。它们不应被写成已经恢复，也不应与本轮新增机制混为一谈。

| 项目 | 当前差异与影响 | 后续判断 |
|---|---|---|
| metadata MLP隐宽 | 原 `max(16, hidden)` 变成hidden；hidden<16时架构不同，128主扫描不受影响 | 若需要复现小宽度实验，再恢复最小宽度 |
| 自定义head/temporal宽度 | head固定2×hidden；baseline LSTM固定hidden，旧自定义参数不再提供 | 有实际自定义对照时恢复参数能力 |
| Python构造默认值 | spatial/head dropout=.05、temporal dropout=0、hidden默认128；与原直接构造默认不同 | 不要反向改变train.py已显式传入的CLI/config值 |
| 模型/API与H | ModelConfig/ForecastOutput及权重键改变；H必须与配置一致 | 保留新API；跨H推理另设明确实验 |
| baseline lag容量校验 | argparse仍对baseline检查max_time_steps，可能误拒绝长H | 后续可将校验限制到PACT |
| station JSON | 精确文件名/lat/lon/elevation_m；部分旧别名不再接受 | 可恢复无害别名，保留finite检查；不要默默把坏经纬度当0 |
| metadata缺失 | 显式关闭metadata可只用learned station token；启用后缺文件报错 | 保留明确开关和错误检查 |
| GraphStore/View | 简化构造、`*_graphs.pt`、新sample_id与tag格式；部分旧pattern/helper移除 | 仅在确有外部调用时恢复便利接口；保留输入历史检查 |
| 模型维度来源 | 仍从store第0图取维度，而非首个TRAIN图 | 维度不一致数据应后续收紧到TRAIN首图 |
| TF32/确定性 | 显式设置后端开关及benchmark=False；与原不指定时保留后端默认有差异 | 当前profile显式值保留；按复现需求单独处理 |
| 临时统计线程 | 去掉NumPy OMP/MKL临时helper | 只在确认统计瓶颈后恢复性能处理 |
| 阈值拟合/广播 | 所有mode都拟合三个阈值；对象广播；空/非finite TRAIN报错 | 可后续按需计算；dual必须保有事件阈值与比例 |
| 极端tail_frac/mag边界 | 不使用原tail_frac夹紧；mag仍同时校验p_lo<p_hi | 默认配置不受影响；极端或非标准调用需独立决定 |
| DDP累积/Slurm | Python支持no_sync累积，shell仍限制原单进程累积；Python主要读取torchrun环境变量 | 不宣称直接srun/非标准GPU映射已兼容；新DDP组合需专测 |
| 汇报归约/JSON | 部分All/Peak归约和旧代码末位不同；部分JSON直接写入 | 不恢复额外forward；关键JSON原子写入可后续处理 |
| parser别名/固定flag | 部分strip行为收窄；stable_arch/dual_mode仍只接受当前值 | 无害别名可恢复；冗余flag删除需同时改全部配置 |

当前没有旧retry/rollback guard、常数输出恢复路径或旧residual dual loader。它们是早期中间试验历史，不是本轮必须恢复的原版基础设置。`p_mean` 已按用户要求退役，不再列为待恢复接口。

## 8. 逐文件索引与再生成

下面列出当前全部差异文件；行数来自实际文件，方法和接口影响以上文为准。生成清单不包含本报告、`docs/audit/`及被忽略的数据/产物，避免自引用。

```bash
python docs/audit/compare_original.py --original /media/volume/PACT-Data/Emulator
```

该命令会更新hash、计数、完整diff和训练CLI声明比较；本报告的语义解释、索引和验证结果应随修改同步维护。原始历史验证证据不被重新解释成新测试结果。

| 状态 | 文件 | 原版→当前行数 | 变化与影响 |
|---|---|---:|---|
| 修改 | [.gitignore](../.gitignore) | 52→49 | 保留独立仓库的数据/产物忽略规则；原 experiment_configs 规则未迁入。 |
| 新增 | [DUAL_HEAD_EXPLAINED.md](../DUAL_HEAD_EXPLAINED.md) | 0→217 | 完整解释 b+p r、物理监督、阈值、先验、消融与诊断。 |
| 修改 | [README.md](../README.md) | 483→148 | 更新入口、当前训练/推理契约、真实事件比例、机制消融与诊断用法。 |
| 新增 | [STABILITY_AND_DUAL_HEAD.md](../STABILITY_AND_DUAL_HEAD.md) | 0→42 | 说明保留的稳定性数学、显式监督模式和诊断边界。 |
| 删除 | [changelog.md](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/changelog.md) | 108→0 | 原分散模块/历史说明不单独保留；当前能力和剩余差异见正文。 |
| 修改 | [configs/configs_infer/infer_config_common.sh](../configs/configs_infer/infer_config_common.sh) | 40→41 | 新增默认关闭的 DUAL_DIAGNOSTICS。 |
| 修改 | [configs/configs_train/train_config_common.sh](../configs/configs_train/train_config_common.sh) | 94→93 | 移除 p_mean，加入独立百分位和命名消融；保留原科学默认值。 |
| 修改 | [configs/configs_train_single/train_config_common.sh](../configs/configs_train_single/train_config_common.sh) | 95→94 | 移除 p_mean，加入独立百分位和命名消融；single 默认继续生效。 |
| 新增 | [configs/dual_ablations/train_config_NCEP_Battery_Fixed_Gate.sh](../configs/dual_ablations/train_config_NCEP_Battery_Fixed_Gate.sh) | 0→5 | 引用Stable Dual，仅改变文件名所示的机制模式和结果目录。 |
| 新增 | [configs/dual_ablations/train_config_NCEP_Battery_No_Branch_Supervision.sh](../configs/dual_ablations/train_config_NCEP_Battery_No_Branch_Supervision.sh) | 0→5 | 引用Stable Dual，仅改变文件名所示的机制模式和结果目录。 |
| 新增 | [configs/dual_ablations/train_config_NCEP_Battery_No_Excess_Loss.sh](../configs/dual_ablations/train_config_NCEP_Battery_No_Excess_Loss.sh) | 0→5 | 引用Stable Dual，仅改变文件名所示的机制模式和结果目录。 |
| 新增 | [configs/dual_ablations/train_config_NCEP_Battery_No_Gate_BCE.sh](../configs/dual_ablations/train_config_NCEP_Battery_No_Gate_BCE.sh) | 0→5 | 引用Stable Dual，仅改变文件名所示的机制模式和结果目录。 |
| 新增 | [configs/train_config_NCEP_Battery_Stable_Dual.sh](../configs/train_config_NCEP_Battery_Stable_Dual.sh) | 0→5 | 引用 Stable Single，仅改变 head 与 dual 监督设置。 |
| 新增 | [configs/train_config_NCEP_Battery_Stable_Single.sh](../configs/train_config_NCEP_Battery_Stable_Single.sh) | 0→32 | 统一稳定性 single 对照，沿用原最佳训练条件。 |
| 修改 | [emulator/common/__init__.py](../emulator/common/__init__.py) | 28→3 | 重组当前公开导出；不提供全部旧Python API兼容层。 |
| 修改 | [emulator/common/cli.py](../emulator/common/cli.py) | 12→20 | 集中布尔值与 temporal 别名解析；部分 strip 行为差异仍在。 |
| 删除 | [emulator/common/distributed.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/emulator/common/distributed.py) | 53→0 | 原分散模块/历史说明不单独保留；当前能力和剩余差异见正文。 |
| 新增 | [emulator/common/dual.py](../emulator/common/dual.py) | 0→19 | 集中5种消融定义与学习 gate 的有限 logit 初始化边界。 |
| 修改 | [emulator/common/inference_artifacts.sh](../emulator/common/inference_artifacts.sh) | 72→72 | 保留命令/配置快照；记录诊断开关，移除 p_mean 字段。 |
| 删除 | [emulator/common/io_utils.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/emulator/common/io_utils.py) | 26→0 | 原分散模块/历史说明不单独保留；当前能力和剩余差异见正文。 |
| 修改 | [emulator/common/runtime.py](../emulator/common/runtime.py) | 99→26 | 集中 seed、线程、确定性/TF32与时间戳；旧helper不再单独保留。 |
| 修改 | [emulator/data/__init__.py](../emulator/data/__init__.py) | 43→7 | 重组当前公开导出；不提供全部旧Python API兼容层。 |
| 修改 | [emulator/data/graph_store.py](../emulator/data/graph_store.py) | 351→82 | 简化CPU存储/分割/View，检查历史；移除全部均值特征传递。 |
| 修改 | [emulator/data/normalization.py](../emulator/data/normalization.py) | 118→22 | TRAIN统计应用使用新tensor，避免污染原图；不再归一化 p_mean。 |
| 修改 | [emulator/data/station_metadata.py](../emulator/data/station_metadata.py) | 103→25 | 保留当前字段和finite校验；部分原别名尚未恢复。 |
| 修改 | [emulator/data/stats.py](../emulator/data/stats.py) | 310→102 | 保留原统计运算/RNG；拟合独立三个阈值和严格TRAIN事件比例。 |
| 修改 | [emulator/inference/__init__.py](../emulator/inference/__init__.py) | 10→5 | 重组当前公开导出；不提供全部旧Python API兼容层。 |
| 新增 | [emulator/inference/dual_diagnostics.py](../emulator/inference/dual_diagnostics.py) | 0→63 | 纯NumPy计算Brier、两种PR面积、可靠性分箱与分支误差。 |
| 删除 | [emulator/inference/engine.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/emulator/inference/engine.py) | 103→0 | 原分散模块/历史说明不单独保留；当前能力和剩余差异见正文。 |
| 修改 | [emulator/inference/grouping.py](../emulator/inference/grouping.py) | 30→46 | 保留逐年及ALL/past/future分组，区分精度人口与耗时人口。 |
| 修改 | [emulator/models/__init__.py](../emulator/models/__init__.py) | 25→4 | 重组当前公开导出；不提供全部旧Python API兼容层。 |
| 修改 | [emulator/models/architectures.py](../emulator/models/architectures.py) | 891→147 | ModelConfig组织原backbone与当前稳定性路径；移除压力编码并支持固定gate。 |
| 新增 | [emulator/models/heads.py](../emulator/models/heads.py) | 0→69 | 拆分single与受约束dual head；输出实际gate概率，支持无MLP固定gate。 |
| 新增 | [emulator/models/spatial.py](../emulator/models/spatial.py) | 0→49 | 拆分原GraphSAGE/CNN空间编码，保留默认卷积与激活。 |
| 新增 | [emulator/models/temporal.py](../emulator/models/temporal.py) | 0→48 | 拆分temporal模块；保留当前PreLN/gain和LSTM/GRU残差形式。 |
| 修改 | [emulator/training/__init__.py](../emulator/training/__init__.py) | 13→5 | 重组当前公开导出；不提供全部旧Python API兼容层。 |
| 新增 | [emulator/training/arguments.py](../emulator/training/arguments.py) | 0→244 | 集中CLI与前置验证；解耦事件百分位，解析命名消融并保留默认防误关。 |
| 修改 | [emulator/training/engine.py](../emulator/training/engine.py) | 586→112 | 训练/评估共享forward，保留累积/All/Peak口径；显式推理时才收集分支。 |
| 修改 | [emulator/training/losses.py](../emulator/training/losses.py) | 86→112 | 保留10种最终预测loss和物理分支监督；命名消融不改变L_r全batch分母。 |
| 新增 | [emulator/training/metrics.py](../emulator/training/metrics.py) | 0→33 | 集中All/Peak回归指标，复用现有预测。 |
| 修改 | [infer.py](../infer.py) | 708→316 | 按checkpoint重建与选择人口；保留旧报告，增加独立诊断导出与训练元数据。 |
| 修改 | [infer.sh](../infer.sh) | 330→340 | 保留逐checkpoint路径解析、快照和原报告参数；透传诊断开关。 |
| 修改 | [infer_multi.sh](../infer_multi.sh) | 289→295 | 保留多结果目录遍历与快照；透传诊断开关。 |
| 新增 | [preprocessing/forcing_cmip6.py](../preprocessing/forcing_cmip6.py) | 0→74 | 额外通用CMIP6 CLI；保留压力空间去均值，不输出额外均值字段。 |
| 修改 | [preprocessing/preprocessing_forcing_CMIP6_AWI_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_AWI_Mean_Removal.py) | 214→209 | 保留各自路径/NPY/NPZ和3h→6h抽样；只删除额外压力均值字段。 |
| 修改 | [preprocessing/preprocessing_forcing_CMIP6_CNRM_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_CNRM_Mean_Removal.py) | 214→209 | 保留各自路径/NPY/NPZ和3h→6h抽样；只删除额外压力均值字段。 |
| 修改 | [preprocessing/preprocessing_forcing_CMIP6_EC_EARTH_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_EC_EARTH_Mean_Removal.py) | 214→209 | 保留各自路径/NPY/NPZ和3h→6h抽样；只删除额外压力均值字段。 |
| 修改 | [preprocessing/preprocessing_forcing_CMIP6_MPI_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_MPI_Mean_Removal.py) | 214→209 | 保留各自路径/NPY/NPZ和3h→6h抽样；只删除额外压力均值字段。 |
| 修改 | [preprocessing/preprocessing_forcing_CMIP6_MRI_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_MRI_Mean_Removal.py) | 214→209 | 保留各自路径/NPY/NPZ和3h→6h抽样；只删除额外压力均值字段。 |
| 修改 | [preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py](../preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py) | 172→159 | main/CLI参数化与NPY/NPZ输出；保留去均值，删除额外均值字段。 |
| 修改 | [preprocessing/preprocessing_simulation.py](../preprocessing/preprocessing_simulation.py) | 355→357 | 保留mesh/station/时间流程与默认2005；额外接受years/argv。 |
| 修改 | [preprocessing/time_align_unified.py](../preprocessing/time_align_unified.py) | 1008→931 | 保留fixed315/peryear、fallback和skip；不再从NPZ读取或向图传递 p_mean。 |
| 新增 | [tests/test_config_interfaces.py](../tests/test_config_interfaces.py) | 0→161 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 删除 | [tests/test_core_architecture.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_core_architecture.py) | 271→0 | 原测试由当前契约测试替代；已退役的均值接口不恢复。 |
| 新增 | [tests/test_dual_experiments.py](../tests/test_dual_experiments.py) | 0→246 | 本轮新增：阈值/先验、5种模式贯通、梯度、诊断与shell入口。 |
| 新增 | [tests/test_dual_loss.py](../tests/test_dual_loss.py) | 0→158 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 新增 | [tests/test_forcing.py](../tests/test_forcing.py) | 0→24 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 删除 | [tests/test_inference_artifacts.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_inference_artifacts.py) | 179→0 | 原测试由当前契约测试替代；已退役的均值接口不恢复。 |
| 删除 | [tests/test_inference_audit.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_inference_audit.py) | 116→0 | 原测试由当前契约测试替代；已退役的均值接口不恢复。 |
| 新增 | [tests/test_inference_reporting.py](../tests/test_inference_reporting.py) | 0→129 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 删除 | [tests/test_missing_pmean.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_missing_pmean.py) | 117→0 | 原测试由当前契约测试替代；已退役的均值接口不恢复。 |
| 删除 | [tests/test_model_audit.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_model_audit.py) | 107→0 | 原测试由当前契约测试替代；已退役的均值接口不恢复。 |
| 新增 | [tests/test_models.py](../tests/test_models.py) | 0→113 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 新增 | [tests/test_pipeline.py](../tests/test_pipeline.py) | 0→135 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 新增 | [tests/test_preprocessing_pipeline.py](../tests/test_preprocessing_pipeline.py) | 0→114 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 修改 | [tests/test_time_alignment.py](../tests/test_time_alignment.py) | 101→124 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 新增 | [tests/test_training.py](../tests/test_training.py) | 0→96 | 当前接口与数学契约回归；具体覆盖和本轮证据范围见第6节。 |
| 修改 | [train.py](../train.py) | 1522→213 | 集中训练编排；TRAIN拟合/广播阈值与先验，保存有效模式和完整元数据。 |
| 修改 | [train.sh](../train.sh) | 649→691 | 保留原条件扫描；透传独立百分位/消融，run tag包含设置，移除 p_mean。 |
