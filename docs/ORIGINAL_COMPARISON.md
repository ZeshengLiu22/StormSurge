**StormSurge 与原版 Emulator：当前差异、影响和还原建议**

本报告对应独立仓库 [ZeshengLiu22/StormSurge](https://github.com/ZeshengLiu22/StormSurge) 的当前源码，已提交代码基准为 `663e7a2`（2026-09-10），包含 `p_mean` 删除及dual阈值、先验、消融和推理诊断更新。本轮在该基准上恢复metadata最小隐宽、自定义head/temporal宽度及Python构造默认值；训练CLI默认和原config读取路径保持。原版基准是本地 `Emulator` 实际文件，对应 [PACT_Storm_Surge_Emulator 的提交 bb62a22](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/tree/bb62a2297a0d37a35db5bff352ee815e05676c93)。它相对上次基准 `3d4be39` 仅提交了此前本地 `.gitignore` 的3行变化，模型与训练源码未变；原版工作树干净。比较生成时间见 [comparison_summary.json](audit/comparison_summary.json)。

**结论：主扫描使用的基础空间拓扑和 H=0 baseline 保留；PACT 包含已确认的新 dual head 与稳定性数学改动。此外仍有自定义模型接口、输入检查、运行开关和输出组织等差异，不能概括为“仅改了 dual/stability”。** 当前已整体退役 `p_mean`，完善真实TRAIN事件先验、独立dual阈值、显式消融与可选推理诊断；此前恢复的DDP seed、归一化统计、验证补齐、loss下限、环境配方和推理/预处理汇报继续保留。

本报告逐项列出当前实现与原版的差异、影响和还原建议。**表中的“建议还原”仍是未执行的建议**；已经完成的修改会明确标为已恢复、未变或已退役。`/home/exouser/media/volume/PACT-Data/StormSurge` 与 `/media/volume/PACT-Data/StormSurge` 指向同一独立仓库，原版目录保留。

源码/config/既有文档比较共 **203 个路径：134 个相同、35 个修改、25 个新增、9 个删除，共69个差异文件**。不包含本报告、`docs/audit/`、数据、checkpoint、结果和缓存。逐文件hash及行数见 [CSV](audit/source_inventory.csv) / [JSON](audit/source_inventory.json)，完整行差异见 [original_to_current.diff](audit/original_to_current.diff)。文末列出全部69个文件。

建议标签含义：**保留**＝不建议退回旧实现；**还原**＝建议恢复该具体能力或语义；**部分还原**＝恢复便利性或实验接口，但保留明确错误检查；**已恢复/未变**＝没有新的还原工作。建议是代码与实验设计判断，不是精度提升的实验证明。

**一、基础模型：哪些没有改，哪些属于已确认的方法变化**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| H=0 baseline | GraphSAGE/CNN → spatial mean pooling → dropout → Linear(K)，另可选p_mean输入 | 保留原关闭p_mean时的计算路径；额外均值接口已删除 | 不包含PACT attention、temporal或dual head；历史默认路径数值对照一致，已退役的压力分支另列 | **保留默认路径**；不恢复已退役功能 |
| H>0 baseline | 各时刻空间编码/mean pool → 单层LSTM → 最后时刻 → Linear，另可选p_mean输入 | 原关闭p_mean时的计算路径保持；不再拼接均值特征 | LSTM原来就存在；自定义temporal_hidden已恢复 | **保留默认路径** |
| GraphSAGE | 相同层数SAGEConv、LeakyReLU(.1)、dropout | 抽到SpatialEncoder，层和计算保留 | 没有换GNN种类、邻接或pooling | **保留模块拆分** |
| CNN | 3×3/padding1卷积，中间宽度29，最后输出hidden | 同样的卷积与默认通道 | 未改变原扫描的CNN参数预算；输入校验另列 | **未变** |
| PACT总连接 | spatial → station-query attention → lag/temporal → horizon attention → head | 总连接保持 | 新的归一化、残差公式和head会改变实际函数 | **保留总连接** |
| 相对lag与H=0 PACT | 当前forcing对应lag0；H=0 forecast context加horizon query | 保留 | 这条H=0 residual原来就有，不能宣称是新修复；也不能与H=0 baseline混淆 | **未变** |
| station attention输出 | 直接进入后续lag/temporal | 加FP32、无affine参数的LayerNorm | 改变表示尺度与梯度，影响PACT各H；不影响baseline | **保留**，属于已确认的稳定性设计 |
| head前context | attention结果及可选pressure扩展输入直接入head | attention context经FP32、无affine参数的LayerNorm后入head；pressure扩展输入已删除 | norm改变head所见尺度及梯度；关闭p_mean的原主扫描不受已删除分支影响 | **保留**，仍需完整扫描验证稳定性效果 |
| Transformer/MLP residual | PostLN残差，无当前可学习gain | PreLN，`x + exp(log_gain) × branch(LN(x))`，gain初值.1 | 数学改变；每分支gain参与Adam学习。.1是初始化，不是LR除10；不是clip | **保留**，不要为了代码对齐删掉已确认的方法变化 |
| LSTM/GRU temporal residual | `LayerNorm(x + dropout(RNN(x)))` | 保留 | 没有把RNN temporal也改成PreLN；其前后仍受到PACT新增norm影响 | **未变** |
| PACT head激活 | 隐层ReLU | LeakyReLU(.1) | 负区间保留梯度，避免ReLU完全截断；不证明它单独根治所有collapse | **保留** |
| dual预测结构 | `base + sigmoid(alpha) × sigmoid(gate) × tail`；tail可tanh截断，gate可window/horizon | 两个回归分支body/excess＋一个window事件gate：`body + p × excess` | body限制在TRAIN阈值以下，excess非负；去掉额外alpha，分工和输出约束改变 | **保留新dual**，按用户决定不重新启用旧head组合 |
| dual初始化 | 旧gate bias/alpha初始化与tail网络 | gate/excess末层权重置0；学习gate bias取真实TRAIN事件比例q_E的logit，excess初始softplus输出.1 | q_E按严格超过阈值统计，ties下不一定等于tail_frac；仅学习logit初始化截到[1e-6,1−1e-6]，原始比例仍保存 | **保留真实先验**，不把初始化当成已证明最优；不承诺与前一版固定5%先验逐位相同 |
| dual监督 | 只有所选最终预测loss，分支无独立语义约束 | 默认dual_ablation=none强制body/excess/gate三项；误关或启用项权重0纠正为1并记录WARNING，正权重保留 | **dual+mse已经不是只有MSE**；single/baseline不添加；只有命名消融关闭指定项 | **保留默认完整监督**，正式模型与消融必须分别标识 |
| dual阈值与事件比例 | 没有当前受约束body/excess的事件阈值；tail_frac控制预测tail子集 | exceedance_percentile独立控制TRAIN窗口峰值的τ，默认95；统计严格max(Y)>τ的事件数和q_E | 改tail_frac不再改变dual事件定义；τ、q_E及事件数写入checkpoint/summary，不使用Val/Test拟合 | **保留解耦与显式元数据** |
| 分支监督消融 | 旧head无这三项独立监督，也没有对应命名接口 | no_gate_bce关闭L_g；no_excess_loss关闭L_r；no_branch_supervision关闭L_b/L_r/L_g，仅保留所选最终预测loss | 可区分监督分解与额外MLP带来的收益；无BCE的gate不能直接宣称已校准；其余启用项仍受防误关约束 | **保留显式消融**，不取消正式模型的防误关规则 |
| 固定gate对照 | 无当前p=q_E的命名对照 | fixed_gate使用原始TRAIN比例作buffer，不建gate MLP，gate_logits=None；保留body/excess监督 | p与输入无关，可检验学习gate的贡献；允许q_E精确为0/1，参数预算少一个gate MLP | **保留消融**；比较时说明参数差异，不能临时在推理时切换模式 |

新dual阈值τ由TRAIN窗口峰值的 `exceedance_percentile` 分位确定，默认95；事件为窗口内至少一个时刻严格超过τ。body目标为 `min(Y,τ)`，excess目标为 `(Y−τ)+`，gate目标为事件0/1。μ/σ也仅由TRAIN拟合，`τ′=(τ−μ)/σ`。模型保持 `b′=τ′−softplus(τ′−a)`、`r′=softplus(d)`、`ŷ′=b′+p·r′`；物理body为 `μ+σb′`，物理excess为 `σr′`，excess不加μ。Excess辅助项mask非事件窗口后仍按整个 `B×K` 平均，不除事件数或q_E；事件窗口内未超阈的horizon仍监督为0。Gate BCE乘TRAIN的 `mean(σ²)`。推理forward不读取未来标签。完整定义和局限见 [dual head说明](../DUAL_HEAD_EXPLAINED.md)。

原来的tail loss仍约束**最终预测在峰值窗口的误差**，使用独立tail_threshold及 `max(Y)>=tail_threshold`；slope loss约束**最终预测相邻horizon的一阶差分误差**，保留软mask和Charbonnier/Huber惩罚，至少需要两个horizon。二者没有被三项dual loss代数替代。`head_type=dual`、`dual_ablation=none`、`loss_mode=mse_tail_slope` 且tail/slope权重为正时，实际是 **MSE＋dual＋tail＋slope**；默认dual配置的mse仅启用MSE＋dual。保留原扫描组合，通过消融决定最终采用哪些项，不能预先断定新head一定更好。

**二、已恢复或一直保持的训练与实验口径**

| 项目 | 原来做什么 | 当前状态 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 原shell配置 | 127份配置，其中112份具体训练profile | 原127份均保留，其中124份逐字相同；2份training common接入新dual并删除p_mean，1份inference common接入诊断；另有2份Stable及4份消融 | 112份原具体训练profile逐字保持；4份消融source Stable Dual，仅改机制与结果目录；原H/LR/loss/站点/数据源组合保持 | **已保留**；不要删扫描轴 |
| 条件扫描 | LOSS/LR/H/WMSE_Q/TAIL_LAMBDA/SLOPE_LAMBDA/SLOPE_MASK_S，无关轴按原条件收缩 | 同样的条件笛卡尔积 | 390组合回归通过；configs_train_single指单GPU＋累积，不是single head | **未变** |
| 优化器 | Adam，weight_decay=1e-5，默认betas/eps | 保持 | 没有替换AdamW或修改LR/衰减值 | **未变** |
| Cosine设置 | linear warmup＋cosine；默认warmup5、起始倍率.1、min_lr1e-6 | 设置保持，内部由LambdaLR表达 | 6组逐epoch曲线最大差约6.1e-18；类名变化不等于LR改变 | **保留等价表达**，不必为类名恢复组合scheduler |
| ROP | 原有ReduceLROnPlateau及参数，按All或Peak物理RMSE | 保持可选；没有改成默认 | Val口径恢复后继续使用指定指标 | **未变** |
| DDP rank seed | `seed+rank` | 已恢复 | rank0/1实际观测为42/43；DDP仍同步模型权重 | **已恢复** |
| DDP sampler | 默认seed0，逐epoch set_epoch | 已恢复 | 保持原样本分配/排序；sampler seed不能按rank不同 | **已恢复** |
| robust/mag自动抽样 | x_nodes_per_graph<=0表示每TRAIN图最多256节点 | 已恢复 | 仅用于拟合归一化，模型仍接收完整图 | **已恢复** |
| X/Y统计 | 训练设备上按原FP32/FP64顺序求矩，DDP归约；另有p_mean统计 | 主要X/Y统计设备、转换、平方/方差和归约已恢复；额外p_mean统计已删除 | 历史CPU/GPU/DDP对照的主要X/Y统计所测差0；当前运行环境与本轮CPU验证范围见第七节 | **保留已恢复的X/Y统计**；不恢复已退役的均值统计 |
| augmentation RNG | probability>=1直接生成缩放/平移，不先抽概率 | 已恢复 | 所测输入和随后随机数一致；扰动没有关闭 | **已恢复** |
| DDP Val All | DistributedSampler补齐；FP32 batch均值、FP64加权累计/all-reduce | 已恢复 | 不整除时重复样本仍计入All，保持best epoch/ROP的原实验口径 | **已恢复**，本研究按用户决定保留此口径 |
| Val Peak | 原rank0完整验证集，按GT峰值选ceil(.05N)窗口 | 复用当前预测，去掉padding，恢复数据集顺序、原argsort/FP32归约 | 同一组预测时所测结果相同；batch内核/额外forward删除可能影响末位或后续RNG | **保留单次forward** |
| loss种类 | mse/wmse/mse_tail/wmse_tail/mse_wtail，各可加_slope | 10种最终预测loss均保留；默认dual额外叠加三项监督，命名消融可去掉指定项 | MSE＋dual＋tail＋slope可同时启用；tail/slope作用于最终预测，dual作用于分支及gate；两类事件阈值已分开 | **保留全部组合**，完整监督与消融分开报告 |
| loss小参数下限 | wmse_s、slope_mask_s至少1e-6；Charb ε、Huber δ至少1e-12 | 已恢复，包括有限的0/负数/极小正数 | 普通常用值不变；极小值不再改变旧公式的有效scale | **已恢复** |
| 梯度累积数学 | 同组microbatch等权，末组按实际microbatch数 | 保持 | 没有改成按不同microbatch样本数重新加权；DDP能力扩展另列 | **未变** |
| 数据划分 | 完整年份分组，TRAIN-only统计，shuffle/future规则 | 合法比例、标准输入时保持 | 上一轮真实Battery 24组split对照一致；异常比例验证更严格 | **保持正常语义** |

DDP seed与augmentation局部随机顺序恢复，不代表整次训练逐位回到原版。原dual训练为日志额外做过train-mode forward，旧Val诊断也额外遍历DataLoader；删除它们会改变后续RNG消耗。**不建议为复现旧随机轨迹恢复无用forward。** 比较方法时应在当前同一代码/环境/预算下重新训练各对照。

**三、仍待决定的模型与输入接口差异**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| metadata MLP隐宽 | `max(16,hidden)` | 恢复`max(16,hidden)`，输出仍为hidden | hidden8时隐宽恢复16；hidden128主扫描不变 | **已恢复** |
| 自定义head宽度 | PACT构造支持head_hidden | ModelConfig.head_hidden控制single及dual各MLP；None时仍为2×hidden | 恢复自定义能力，保持默认实际宽度；配置随checkpoint保存/重建 | **已恢复参数能力** |
| baseline temporal宽度 | 构造支持temporal_hidden | ModelConfig.temporal_hidden控制H>0的LSTM及Linear输入宽度；None时仍为hidden | 恢复自定义LSTM宽度；H=0仍用空间hidden，配置随checkpoint保存/重建 | **已恢复参数能力** |
| Python构造默认值 | spatial/head dropout=0、temporal dropout=.05；hidden必填 | ModelConfig恢复0/0/.05；hidden_channels必填 | train.py仍显式传CLI/config；CLI保持hidden64、dropout .05/0/.05，主扫描config保持hidden128、.05/.05/0 | **已恢复**；训练CLI默认及config读取保持 |
| PACT pressure初始化顺序 | token/global两条可选p_mean编码路径，建层顺序影响同seed初始化 | p_mean编码器、token/global模式、扩展head维度及相关配置整体删除 | 这些压力均值消融不再受支持；原主扫描默认关闭，不需要保留其初始化顺序 | **已退役，不还原**，按用户要求删除 |
| 模型API/权重键 | 多参数constructor，tensor或return_aux；旧类和state_dict命名 | ModelConfig、ForecastOutput和拆分模块；dual输出增加gate_probability，固定gate没有logits；checkpoint保存消融模式 | 旧Python调用与原checkpoint需适配；当前模型由保存的配置重建，模型/loss的fixed_gate设置不一致会报错 | **保留清晰API与模式检查**；用户不要求历史checkpoint兼容 |
| 输入H | PACT主要限制max_time_steps | 必须等于构造时history_steps+1 | 不能直接以不同H喂同一模型，推理同样检查H | **保留默认一致性检查**；确需跨H推理时另恢复显式能力 |
| baseline的max_time_steps校验 | 主要是PACT lag容量参数 | argparse对baseline也检查 | baseline没有该lag embedding，可能误拒绝本可运行的长H | **还原适用范围**，只校验使用该embedding的模型 |
| JSON文件名 | 精确/lower/upper依次查找 | 只读`<station>.json` | Battery请求无法读取仅有的battery.json | **还原别名查找** |
| 字段别名 | latitude/Latitude、longitude/Longitude、elevation/elev_m等可用 | 仅lat/lon/elevation_m | 有效旧输入被拒绝或elevation别名被忽略而使用0 | **还原别名**，保留数值有效性检查 |
| metadata缺失/非finite | 部分字段回退0；缺JSON可只用learned token | 经纬度必需且所有feature finite；启用metadata时缺JSON报错 | 改变异常/不完整数据接受范围。elevation_m缺省仍0；bathymetry启用时原来就必须finite | **部分还原**：可显式选择learned-token模式；不建议默默把坏经纬度当0 |
| pressure缺失 | 启用p_mean后，部分路径以零global编码或省略pressure token回退 | 不再读取、归一化或向图/View传递p_mean与其历史；也不再保留缺失回退策略 | 旧pressure模式不能继续使用；正常forcing仍保留压力空间去均值，五通道布局不变 | **已退役，不还原**；不要恢复缺失压力均值兼容层 |
| CNN网格检查 | batch计数等验证 | ptr.diff＋统一grid元数据检查 | 合法矩形PyG batch计算保持，异常输入接受范围不同 | **保留**，不为接受错误形状回退 |
| GraphStore构造 | 支持pattern/force_cpu/strict_station_filter/log_fn等；`*graphs.pt` | 简化接口、CPU存储；`*_graphs.pt` | 非标准文件名/外部直接调用可能失效；旧station索引helper移除 | **部分还原**：恢复实际使用的pattern能力；不必恢复纯包装函数 |
| View与tag | 旧metadata字段、无sample_id；空版本字段的tag格式不同 | 新sample_id，H0也提供x_hist，严格检查forcing历史；p_mean字段及其长度检查已删除 | 默认模型链路适配；外部消费tag或旧graph属性的脚本可能受影响 | **保留sample_id/forcing历史检查**；需要旧tag的分析脚本可恢复tag格式 |
| 模型维度来源 | 从首个TRAIN graph取in/out维度 | 从store第0个graph取 | 维度一致时相同；过滤外的首图不一致时会错误定模型尺寸 | **还原到TRAIN首图，优先** |

**四、其他训练执行、CLI和性能差异**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 输入归一化存储 | 原地修改tensor | 赋新tensor | 避免输入共享存储被污染；正常PyG batch数值保持。尚不能据此认定它是旧collapse唯一根因 | **保留** |
| 未显式启用TF32 | 不改后端既有默认；启用时设TF32和matmul precision high | 显式设置matmul/cuDNN TF32为开关值；没有单独precision high调用 | 不传--tf32时与原后端默认可能不同；现有TF32=1 profile不受该关闭差异 | **建议还原未指定时的原默认语义**；显式选择仍保留 |
| 确定性与benchmark | 无当前统一deterministic选项 | 可选严格确定性，默认0；显式benchmark=False | 影响复现与内核选择；严格确定性可能明显降速 | **保留可选，默认不强制开启**；记录环境即可 |
| 临时统计线程 | NumPy统计阶段设置OMP/MKL并恢复，含CPU数推断 | 去掉helper，保留torch_threads | 一次性统计耗时可能不同；环境变量中途设置的实际收益需测量 | **不机械还原**，确认瓶颈后再处理 |
| threshold是否计算 | 最终loss需要时才拟合相应阈值 | 每种mode先拟合tail_threshold、wmse_threshold、独立dual τ及真实TRAIN事件比例 | baseline/single纯MSE仍有额外统计；dual即使用MSE也必须有τ/q_E；空或非finite TRAIN标签报错 | **可恢复按需计算**，保留dual所需τ/q_E和有效性检查 |
| threshold广播 | FP32 tensor广播后使用 | rank0拟合并广播Python字典；其中保存物理阈值、原始事件比例、计数和分位，模型使用归一化τ | 原10种预测loss已有对照；新增元数据可追溯；标量精度和对象广播开销与原版不同 | **保留元数据**；小tensor广播可低优先级优化，不能丢失τ/q_E |
| tail_frac极端边界 | threshold函数先将tail_frac夹到[1e-6, .999999]，空输入返回0阈值 | CLI/函数要求0<tail_frac<1，不再按原下限/上限夹紧；空或非finite TRAIN报错；dual百分位独立校验 | 默认.05不变；极端正值的tail阈值不再等价，但不会联动改变dual事件定义 | **可还原tail阈值函数的原极端夹紧**；保留空/坏TRAIN报错，dual先验独立处理 |
| mag校验范围 | mag只检查其使用的p_hi；p_lo不参与 | 合并函数也要求p_lo<p_hi | 可能拒绝原先可用的mag设置，例如p_lo=95、p_hi=90 | **还原按mode校验**，不要检查该mode不使用的参数 |
| DDP梯度累积 | Python拒绝world>1且accum>1 | engine支持no_sync累积；shell仍保持原单进程累积限制 | 新能力不改变原shell扫描；直接Python有更大允许范围 | **保留能力**；使用前单独验证该组合 |
| final test执行 | 原DDP test评估、rank0再导出预测 | rank0一次计算指标并导出 | 无重复forward；不再把DDP padding计入最终Test。可能与旧Test打印数值不同 | **保留**，测试集每个真实样本一次更合适；Val按已决定的原口径 |
| Train指标 | 原按batch累计Train RMSE/MAE，另有诊断 | 在线预测按sample_id整理All/Peak5；DDP重复训练样本在报告中只计一次 | Train报告的加权/归约与旧版不同；优化仍使用sampler补齐数据 | **保留并说明**；Train在线指标不能当作固定模型的train-eval |
| Peak数据收集 | 原rank0独立full pass＋aux诊断 | 各rank每窗口4标量，gather后计算指标 | 少forward，多一次窗口记录收集；内存/通讯随窗口数增长 | **保留**，大数据瓶颈可后续优化收集，不恢复诊断forward |
| 空/非法指标 | 原部分输出NaN，缺少当前统一检查 | 统一窗口汇总的空split为None、非finite预测/目标报错；新dual诊断的无定义项为null；旧推理分组空集仍可能输出NaN | 训练会更早暴露坏结果；诊断JSON不含NaN；不能把旧报告也描述成已全面统一为null | **保留检查与诊断null语义**；旧分组JSON可后续单独统一 |
| 参数解析组织 | 大train.py内argparse，canonical函数部分strip空白 | 独立arguments.py，新增必要检查，部分不再strip | 原常规CLI保留；负warmup、非法split等边界更严格；空白别名接受范围收窄 | **部分还原**：恢复无害的strip/别名；保留不合法训练参数检查 |
| 新/旧参数 | 旧tail_tanh_clip/gate_bias_init/alpha_init_logit、window/horizon gate及3个p_mean参数 | 删除旧3个head参数和3个p_mean参数；gate_mode只window；增加监督、独立百分位、命名消融与运行选项 | 旧head/pressure命令不能原样运行；完整dual和机制对照可明确选择；single/baseline拒绝非none消融 | **保留当前参数语义**，不重新接回已替换或退役功能 |
| 固定值flag | 没有当前stable_arch/dual_mode | stable_arch只接受1，dual_mode只接受exceedance | 无实际选择作用，主要为已有配置声明服务 | **可后续删冗余flag**，需同步全部配置/launcher；本次不删 |
| Slurm与GPU映射 | 识别SLURM_*、master推断、visible-device映射 | Python主要使用RANK/LOCAL_RANK/WORLD_SIZE | 直接srun与非标准GPU映射不兼容；torchrun路径保持 | **保留现状**，用户明确暂不需要Slurm |
| shell便利功能 | tmux/conda/扫描/快照 | 保留并加DRY_RUN、显式PYTHON_BIN、直接执行fallback、scheduler参数透传 | 默认参数不变；过去shell里未生效的自定义scheduler值现在可生效 | **保留**，报告实际解析参数 |
| 梯度裁剪 | 无当前max_grad_norm选项 | 可选普通clip，默认0 | 默认训练未启用裁剪；指定正值时是额外训练设置 | **保留可选**，不默认打开 |

“旧retry/rollback guard”属于此前v2中间试验的历史，不是这里原版基准与当前代码的必然差异。当前没有自动重试、回滚、跳过异常epoch或恢复常数输出的执行路径；不要将它们列为原版必须还原的基础训练设置。

原/现argparse声明的机器清单见 [argparse_comparison.json](audit/argparse_comparison.json)：73项声明保持，删除3个旧head参数及3个p_mean参数，增加12项，另3项的canonical函数或choices改变。声明一致不覆盖解析后的模式纠正和校验，因此上表另外列了默认完整监督、命名消融、baseline容量、mag及tail_frac等边界。

**五、推理、日志、checkpoint与输出**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 训练诊断 | gate/alpha/attention/归一化误差等，含额外计算 | 常规epoch仍仅Train/Val×All/Top5%×RMSE/MAE；dual专项诊断只在显式推理导出时收集 | 常规训练不收集gate/body/excess数组，也不添加诊断forward或逐epoch校准指标 | **保留训练日志精简**；机制研究使用独立推理导出 |
| 时间输出 | UTC文本、epoch time等 | `[Date\|Time]`＋最终wall time，summary保存循环时间与wall_seconds | wall包含准备/最终测试；与逐epoch或单forward时间不是同一口径。日志用本地datetime，当前机器UTC | **保留**；跨时区比较需注意时间标签 |
| 训练artifact命名 | 原可读tag/MD5、旧summary NPZ等 | 参数SHA256 stem、JSONL/JSON、独立配置和压缩test_preds | 旧分析脚本可能需要适配，不影响loss；每组合结果较容易追踪 | **保留当前主格式**，有具体消费者时可加旧格式导出 |
| 同目录重复运行 | 原结果可被同名覆盖 | config使用exclusive创建，相同显式tag/参数会报FileExistsError | 原命令重跑可能需要新run_tag/目录；默认生成新的时间标签 | **保留防覆盖行为**，复跑使用独立run目录 |
| JSON写入 | 部分原helper用atomic替换 | 部分直接write_text | 中断时可能留下不完整报告 | **建议还原关键JSON的原子写入**，无需恢复全部IO包装层 |
| checkpoint | 原schema与模型键；推理重建旧模型 | model_config/model_state/normalization、station_feat和精确split_tags；另存loss_thresholds/dual_metadata及有效消融设置 | 保存物理τ、q_E、gate_init_prior、事件数、TRAIN窗口数、分位及事件定义；原版及含已删p_mean字段的中间checkpoint不直接兼容 | **保留新格式与TRAIN元数据**；从头训练，用保存的模式重建 |
| dual诊断导出 | 原有辅助诊断没有当前受监督body/excess分解与同一事件口径 | 显式--dual_diagnostics导出独立NPZ/JSON，保存对齐的p、物理body/excess、p×excess、E、truth/prediction/tags及TRAIN元数据 | 总体/逐年Brier、stepwise AP、梯形PR-AUC、10个可靠性分箱、正负事件gate直方图和条件误差；无正事件PR为null；复用推理forward，增加导出内存和磁盘 | **保留可选导出**；不自动绘图，不从Test拟合τ，两个PR面积须分别标注 |
| 默认推理样本 | 从checkpoint args对当前目录重新划分年份 | 按训练时保存的test tags | 目录增加/缺失年份时结果不同；避免推理时悄悄更换test population | **保留saved tags**；需要重划分应明确另选scope |
| 推理override | 可覆盖部分架构/H；缺station某些路径fallback全站点 | 模型/H参数用于验证checkpoint；station严格筛选 | 旧命令灵活性减小，减少错误模型/站点混用 | **保留默认检查**，跨H实验另设明确接口 |
| 推理metadata | 同站点可重新读JSON | 同站点使用checkpoint保存特征，换站点才读取新JSON | JSON后来修改时，不会无意改变原checkpoint条件 | **保留**；若要覆盖，做显式实验选项 |
| 逐年/分组报告 | 每年RMSE/MAE/runtime；ALL/past/future；平均年耗时排除2014_2015 | **已恢复**，save_npz不影响报告是否计算 | past按起始年1979–2014，future2070–2099；其他年只进入ALL；2014_2015仍进入精度统计 | **已恢复** |
| 旧推理文件/目录 | metrics_per_year…json、preds…ALLYEARS.npz及标签/目录规则 | **已恢复**；保留metrics.json/predictions.npz；可选dual诊断另存文件，两个shell用DUAL_DIAGNOSTICS=1并写入配置快照 | 原预测NPZ仍仅有y_true/y_pred/tags；诊断可独立于save_npz启用，但要求dual checkpoint含TRAIN事件元数据 | **保留已恢复的旧格式及独立诊断**；确认消费者后才删冗余副本 |
| 逐年归约数值 | 原每batch FP32误差sum、再累计 | 共享engine用FP32 batch均值及FP64权重；总体分组仍用FP64数组 | 数学定义相同，GPU/批次舍入末位可能不同 | **通常不需还原**；要逐位对照特定旧报告时再恢复该归约 |
| inference模块API | 独立infer_one_loader返回tuple | infer.py调用共享run_epoch，grouping包保留分组/标签函数 | 旧Python导入路径不可用，但科学汇报功能已回到当前CLI | **不恢复重复forward实现**；确有外部调用时可提供薄适配 |

**六、环境、预处理与仓库组织**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 训练环境配方 | PyTorch2.8+cu128及原PyG扩展 | **原文件逐字恢复** | 此前2.6/cu124的配方改动不再存在；现有已安装环境未被重装 | **已恢复** |
| dataprep配方 | 原完整依赖/版本列表 | **原文件逐字恢复** | 不再擅自更改Python/NumPy/pandas/torch配方 | **已恢复** |
| 五个CMIP6入口 | AWI/CNRM/EC_EARTH/MPI/MRI独立脚本、各自路径、3h→6h抽样及NPY/NPZ，另输出p_mean字段 | 五个入口与原路径/抽样保持，删除额外均值字段；forcing仍去压力空间均值 | NPY/NPZ主要forcing保留，不能再称五个脚本逐字相同；旧均值字段消费者需适配 | **保留入口及forcing处理**；不恢复已退役的均值字段 |
| generic CMIP6 CLI | 无统一额外入口 | 新增forcing_cmip6.py，显式路径/模型参数，保存NPY/NPZ；不再保存p_mean | 没有替代原5脚本；同样保留压力空间去均值；直接函数对异常输入更严格 | **保留可选入口**，不恢复均值特征支持 |
| NCEP入口 | 顶层脚本，固定默认路径/年份；NPY/NPZ及额外均值字段 | main/CLI参数化，可安全import；原默认、主要warn/skip和forcing去均值保留，删除p_mean字段 | 主要forcing变换保持，可显式换路径；NPZ额外字段与打印细节不同 | **保留参数化与均值字段删除**，不恢复import即处理数据 |
| simulation | 完整mesh/station/时间处理，默认year_list=[2005] | 原逻辑/默认路径/默认2005恢复；额外支持--years及可传argv | 不再默认发现所有年份；批量年需显式指定 | **保留**，将来改默认年份应作为明确行为变更 |
| time alignment | fixed315/peryear、两个out_root参数、缺time回退、缺年skip；可读取并传播p_mean历史 | 保留两种模式、原fallback/skip及--out_root别名；不再向图添加p_mean或其历史 | 原main的fixed调用本就注释，当前默认也只跑peryear；forcing/标签时间对齐保持，额外图属性减少 | **保留恢复结果与别名**；不恢复p_mean图属性 |
| LICENSE与静态资源 | 原许可证、station JSON等 | 对应文件逐字保留 | 仓库独立不删除原授权/归属信息 | **保留** |
| .gitignore | 此前本地experiment_configs忽略项已随bb62a22提交 | 当前未继承该项；大数据/结果/checkpoint仍忽略 | 属于仓库管理差异，无模型影响；基准已改为干净的bb62a22，不能继续称未提交修改 | **不必自动还原该项** |
| README/方法说明 | 原大README及changelog | 当前README、新dual/stability说明；旧changelog移除 | 新入口更清楚，但历史记录变少 | **保留新说明**；若需要历史追溯，建议将旧changelog作为历史文档归档 |
| 测试 | 旧模型/诊断/推理接口测试 | 模型/梯度/dual loss/配置/预处理/往返测试；恢复前全量50项通过，本轮模型接口及相关流程34项通过 | CPU测试不覆盖所有旧接口；旧缺失p_mean测试已随功能退役；历史GPU/DDP证据不能替代当前消融验证 | **保留当前测试**；恢复仍需支持的接口时补相应案例 |
| 独立仓库 | 原Emulator自己的Git历史/remote | StormSurge以ce697f9独立初始化，当前代码663e7a2已同步origin/main | origin指向新repo，原版历史未迁入；当前194份文件纳入源码/config/既有文档比较 | **保留独立仓库** |
| 数据与机器路径 | profile引用外部Data/graph/station目录 | 原profile路径保持，数据未打包进Git | Git独立不意味着自动复制大数据；换机器需要设置ROOT_DIR、STATION_JSON_DIR、Python环境等 | **保留实验配置值**，按部署机器显式覆盖路径 |

**七、建议优先顺序与验证范围**

metadata最小隐宽、自定义head/temporal宽度和直接构造默认值已恢复。后续优先还原的是JSON文件名/字段别名、维度从TRAIN首图读取，以及baseline不该受PACT lag容量限制的问题。tail_frac原极端夹紧、mag按mode校验、未显式配置TF32时的原行为与关键JSON原子写入也值得恢复。`p_mean` 已明确退役，不再建议恢复其建层顺序、缺失回退或数据字段。

建议继续保留新dual、默认完整分支监督、真实TRAIN事件先验、独立阈值、显式机制消融和可选推理诊断；也保留已确认的PACT稳定性数学、输入存储隔离、单次forward指标、严格的默认评估人口/站点检查、新checkpoint格式和全部原扫描组合。消融配置已提供实验接口，效果仍需同代码、同环境、同预算的独立训练，不能把接口实现当作收益证据。

四项恢复前的源码曾执行 `python -m unittest discover -s tests`：**50/50通过**，包括112份原训练profile＋2份Stable、4份新消融、390条件组合、真实TRAIN事件比例/ties、独立阈值、分支梯度、5种dual模式训练/保存/推理、两个shell推理入口、标签独立性和诊断顺序/指标。历史日志：[dual_review_tests.txt](audit/evidence/dual_review_tests.txt)；环境、范围与源码哈希：[dual_review_validation.json](audit/evidence/dual_review_validation.json)。诊断开关前后预测相同。

本轮四项恢复执行 `python -m unittest -v test_models test_pipeline test_config_interfaces test_dual_experiments test_dual_loss`（tests加入PYTHONPATH）：**34/34通过**。新增hidden8/128 metadata、自定义head各分支、H0/H>0 baseline temporal宽度及严格checkpoint往返检查；训练/推理测试确认显式dropout覆盖Python默认，CLI默认和114份profile、390条件组合继续通过。hidden128主扫描设置另做36组修改前后同seed对照，初始化参数、train/eval输出及参数梯度逐位一致。范围与源码哈希见 [model_interface_restore.json](audit/evidence/model_interface_restore.json)。

以下数值证据来自首次迁移前的历史验证；保留其当时的对照结论，但不把含已退役pressure路径或旧初始化的结果视为当前event-prior/ablation接口的验证。首次独立快照的42项日志在 [publication_tests.txt](audit/evidence/publication_tests.txt)，与上面的当前50项证据区分：

| 验证 | 对照方式与结果 | 证据 |
|---|---|---|
| H0/H48 baseline | 实际原版：16组CPU，2 encoder×pressure开关×train/eval；初始化、预测、参数梯度所测差0 | [model_equivalence.json](audit/evidence/model_equivalence.json) |
| PACT single | 实际原版对象只加入已确认stability后，与当时版本128组CPU对照；同权重预测/梯度所测差0 | 同上；不表示未经修改的原版PACT与当前相同 |
| 新dual清理等价 | 清理前已确认的新dual vs 当时版本32组CPU，所测差0；早于真实q_E初始化、消融和p_mean删除 | 同上；不是与旧residual dual等价，也不证明当前默认预测与该快照相同 |
| GPU模型 | H100，16组FP32/BF16，baseline/PACT、两encoder、H0/H48；同权重所测差0 | [gpu_equivalence.json](audit/evidence/gpu_equivalence.json) |
| Cosine曲线 | 6组含300epoch、短运行、warmup边界，最大LR差6.1e-18 | [scheduler_comparison.json](audit/evidence/scheduler_comparison.json) |
| 恢复后统计/RNG/loss/Val | CPU、H100、两rank CPU/Gloo：统计与augmentation局部RNG所测差0；每种环境100组小scale loss差0，梯度差≤1.87e-9；Val补齐/Peak对照一致 | [CPU](audit/evidence/protocol_cpu_1ranks.json)、[GPU](audit/evidence/protocol_cuda_1ranks.json)、[DDP](audit/evidence/protocol_cpu_2ranks.json) |
| 实际入口短训练 | 两轮H48 dual：CPU DDP使用seed42/43、sampler0；H100 BF16/TF32完成反传、保存、最终推理 | [DDP](audit/evidence/smoke_ddp.txt)、[GPU](audit/evidence/smoke_gpu.txt) |

上述恢复前50项及本轮34项测试使用 **CPU、PyTorch2.6+cu124、PyG2.7**；原2.8+cu128 YAML没有重新安装验证。本轮未重跑GPU/BF16/NCCL。上表H100/GPU与CPU/Gloo两rank均为历史记录，历史也未验证两GPU/NCCL；本轮没有长程训练、跨seed稳定性或正式精度比较。短训练、同权重对照、函数级一致性证明不同层面的性质；它们**不能证明长期stability已彻底根治、gate已经校准，也不能证明新dual比single或GNN比CNN更好**。证据边界见 [evidence/README.md](audit/evidence/README.md)。

更新逐文件比较可运行：

```bash
python docs/audit/compare_original.py --original /path/to/Emulator
```

脚本重新生成hash清单、计数、完整diff和训练CLI声明比较；本报告的内容及建议随代码更新，章节和表格格式保持。首次发布的v2来源hash在 [v2_source_hashes.json](audit/v2_source_hashes.json)，迁移核对在 [migration_verification.json](audit/migration_verification.json)，两者继续作为历史快照，不代表当前工作树。

**八、全部69个差异文件的索引**

下面的索引逐个覆盖生成清单中的修改/新增/删除文件。语义影响与还原建议在上表对应项目展开；行数变化本身不等于架构或训练数学变化。

<!-- FILE_INDEX -->

| 状态/行数 | 文件 | 原来 → 现在 | 影响 | 建议 |
|---|---|---|---|---|
| 修改 52→49 | [.gitignore](../.gitignore) | 原版已提交的experiment_configs忽略项 → 未继承该项 | 版本管理范围不同，无训练影响；原版当前工作树干净 | 不自动还原该忽略项 |
| 新增 0→217 | [DUAL_HEAD_EXPLAINED.md](../DUAL_HEAD_EXPLAINED.md) | 无独立说明 → 新dual公式、真实先验、独立阈值、消融与诊断 | 明确分支语义、物理单位和证据局限 | 保留 |
| 修改 483→148 | [README.md](../README.md) | 原综合README → 当前使用说明、独立仓库来源与本报告入口 | 运行说明与当前实现对应，历史细节减少 | 保留；历史内容可归档 |
| 新增 0→42 | [STABILITY_AND_DUAL_HEAD.md](../STABILITY_AND_DUAL_HEAD.md) | 无 → 当前stability/dual及显式推理诊断边界 | 区分默认完整监督、消融与常规训练日志 | 保留 |
| 删除 108→0 | [changelog.md](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/changelog.md) | 旧变更历史 → 删除 | 失去本地历史说明 | 建议作为历史文档归档 |
| 修改 40→41 | [configs/configs_infer/infer_config_common.sh](../configs/configs_infer/infer_config_common.sh) | 原推理common → 加默认关闭的DUAL_DIAGNOSTICS | 显式开关独立诊断，不改变默认推理导出 | 保留可选开关 |
| 修改 94→93 | [configs/configs_train/train_config_common.sh](../configs/configs_train/train_config_common.sh) | 旧head/p_mean配置 → 新dual、独立百分位及命名消融，删除均值接口 | 原扫描超参数保持；dual方法和可选机制改变 | 保留当前dual与退役决定 |
| 修改 95→94 | [configs/configs_train_single/train_config_common.sh](../configs/configs_train_single/train_config_common.sh) | 旧单GPU公共配置 → 同样接入新dual/消融，删除p_mean | 单GPU累积语义保留；名称不表示single head | 保留 |
| 新增 0→5 | [configs/dual_ablations/train_config_NCEP_Battery_Fixed_Gate.sh](../configs/dual_ablations/train_config_NCEP_Battery_Fixed_Gate.sh) | 无 → 引用Stable Dual，固定p=q_E并取消gate MLP/BCE | 检验输入相关gate贡献；参数数目减少；其他训练条件继承，结果目录独立 | 保留命名对照，效果待实验 |
| 新增 0→5 | [configs/dual_ablations/train_config_NCEP_Battery_No_Branch_Supervision.sh](../configs/dual_ablations/train_config_NCEP_Battery_No_Branch_Supervision.sh) | 无 → 引用Stable Dual，关闭三项分支辅助loss | 仅优化所选最终预测loss，不保证分解具有独立语义；其他训练条件继承，结果目录独立 | 保留命名对照，效果待实验 |
| 新增 0→5 | [configs/dual_ablations/train_config_NCEP_Battery_No_Excess_Loss.sh](../configs/dual_ablations/train_config_NCEP_Battery_No_Excess_Loss.sh) | 无 → 引用Stable Dual，关闭直接excess辅助loss | excess仍从最终预测loss接受梯度；其他训练条件继承，结果目录独立 | 保留命名对照，效果待实验 |
| 新增 0→5 | [configs/dual_ablations/train_config_NCEP_Battery_No_Gate_BCE.sh](../configs/dual_ablations/train_config_NCEP_Battery_No_Gate_BCE.sh) | 无 → 引用Stable Dual，关闭gate BCE | gate仍从最终预测loss接受梯度；其他训练条件继承，结果目录独立 | 保留命名对照，效果待实验 |
| 新增 0→5 | [configs/train_config_NCEP_Battery_Stable_Dual.sh](../configs/train_config_NCEP_Battery_Stable_Dual.sh) | 无 → 新dual对照profile | 新增实验入口，引用Single公共条件 | 保留 |
| 新增 0→32 | [configs/train_config_NCEP_Battery_Stable_Single.sh](../configs/train_config_NCEP_Battery_Stable_Single.sh) | 无 → 新single对照profile，修正移动后的source路径 | 新增对照且已恢复可执行，不替代原profile | 保留 |
| 修改 28→3 | [emulator/common/__init__.py](../emulator/common/__init__.py) | 导出多个runtime/DDP/IO helper → configure_runtime | 外部旧导入需适配 | 保留精简；必要时薄适配 |
| 修改 12→20 | [emulator/common/cli.py](../emulator/common/cli.py) | bool解析 → 保留bool并加入temporal名称解析 | attn别名保留，部分strip空白能力丢失 | 恢复无害strip |
| 删除 53→0 | [emulator/common/distributed.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/emulator/common/distributed.py) | rank/DDP/all-reduce/print包装 → 删除并直接调用torch | Slurm能力及旧公开导入不同 | 保留精简，Slurm暂不还原 |
| 新增 0→19 | [emulator/common/dual.py](../emulator/common/dual.py) | 无 → 统一5种模式和有限gate初始化先验 | 默认完整监督，命名例外共用同一规则 | 保留 |
| 修改 72→72 | [emulator/common/inference_artifacts.sh](../emulator/common/inference_artifacts.sh) | 原快照变量 → 加运行选择及诊断开关，删p_mean | 重放记录实际推理选择 | 保留 |
| 删除 26→0 | [emulator/common/io_utils.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/emulator/common/io_utils.py) | atomic JSON helper → 删除，调用方直接写 | 中断时报告可能不完整 | 建议恢复关键写入的原子性 |
| 修改 99→26 | [emulator/common/runtime.py](../emulator/common/runtime.py) | 分散seed/线程/标签helper → 统一runtime及时间戳 | TF32默认、确定性控制、临时线程不同 | 恢复未指定TF32语义；其余按第四表 |
| 修改 43→7 | [emulator/data/__init__.py](../emulator/data/__init__.py) | 旧store/split/stats导出 → 简化新接口 | 外部Python导入改变 | 不为旧checkpoint恢复全部包装 |
| 修改 351→82 | [emulator/data/graph_store.py](../emulator/data/graph_store.py) | 宽泛加载/独立split/pressure属性 → store.split、严格forcing历史View，无p_mean | pattern/tag/旧公开接口不同；已退役的均值不再传递 | 按第三表部分还原；不恢复p_mean |
| 修改 118→22 | [emulator/data/normalization.py](../emulator/data/normalization.py) | 原地修改输入及可选均值统计 → 新tensor赋值，无p_mean | 避免共享存储污染；保留原forcing归一化和augmentation RNG | 保留赋值方式与退役决定 |
| 修改 103→25 | [emulator/data/station_metadata.py](../emulator/data/station_metadata.py) | 别名/缺省解析与encoder → 严格解析，encoder并入模型 | 丢失有效旧字段/文件名兼容；模型内metadata最小隐宽已恢复 | 恢复别名，保留finite检查 |
| 修改 310→102 | [emulator/data/stats.py](../emulator/data/stats.py) | 多个stats/可选均值函数 → 统一X/Y统计，独立三个阈值与严格TRAIN q_E | 原主要统计语义保留；默认gate初始化现在依赖真实事件比例 | 保留τ/q_E；按第二/四表处理其他差异 |
| 修改 10→5 | [emulator/inference/__init__.py](../emulator/inference/__init__.py) | 导出独立推理engine及grouping → grouping/标签函数 | 旧infer_one_loader公开入口不再存在 | 共享engine保留，按需薄适配 |
| 新增 0→63 | [emulator/inference/dual_diagnostics.py](../emulator/inference/dual_diagnostics.py) | 无 → 固定TRAIN事件口径的分支与校准汇总 | 独立Brier/PR/可靠性分箱与误差；空条件返回null | 保留可选诊断，精度结论另做实验 |
| 删除 103→0 | [emulator/inference/engine.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/emulator/inference/engine.py) | 独立forward/计时 → 删除，复用run_epoch | 逐年汇报已接回infer.py，归约末位可能不同 | 不恢复重复forward实现 |
| 修改 30→46 | [emulator/inference/grouping.py](../emulator/inference/grouping.py) | 年份/分组helper → 保留并接回dataset标签函数 | 原source/target和past/future规则恢复 | 保留 |
| 修改 25→4 | [emulator/models/__init__.py](../emulator/models/__init__.py) | 旧模型类导出 → ModelConfig/build_model/PACT/Baseline/ForecastOutput | 公开API变化 | 保留当前API |
| 修改 891→158 | [emulator/models/architectures.py](../emulator/models/architectures.py) | 单个大模型文件和可选p_mean → 组合模块，删除均值分支，按模式建head | 保留稳定性数学；metadata最小隐宽、自定义head/temporal宽度及Python默认已恢复 | 保留方法与已恢复接口；其余按第三表 |
| 新增 0→71 | [emulator/models/heads.py](../emulator/models/heads.py) | 模型内旧residual dual → 受约束body/excess、真实先验学习gate或固定gate | 新输出契约与参数语义；恢复head_hidden，默认仍2×hidden；固定gate没有MLP | 保留新head/消融及自定义宽度能力 |
| 新增 0→49 | [emulator/models/spatial.py](../emulator/models/spatial.py) | 模型内空间encoder → 独立SpatialEncoder | 同层计算保持，CNN检查改变 | 保留 |
| 新增 0→48 | [emulator/models/temporal.py](../emulator/models/temporal.py) | 模型内temporal → 独立4种block | MLP/Transformer PreLN+gain，RNN residual保持 | 保留已确认稳定性数学 |
| 修改 13→5 | [emulator/training/__init__.py](../emulator/training/__init__.py) | 多个epoch函数 → ForecastLoss/run_epoch/metric接口 | 训练循环复用，公开导入变化 | 保留 |
| 新增 0→244 | [emulator/training/arguments.py](../emulator/training/arguments.py) | 大train.py中的parser → 独立parser，阈值/消融/运行参数，删除p_mean | 默认完整监督；明确消融可去掉指定项，其他边界仍有差异 | 保留模式规则；恢复误收紧的范围/别名 |
| 修改 586→112 | [emulator/training/engine.py](../emulator/training/engine.py) | 训练/验证/预测/诊断多个循环 → 共享run_epoch，显式推理时才收集dual数组 | 单次forward；Val原口径保留，Train/Test报告差异另列 | 保留共享循环与可选导出 |
| 修改 86→112 | [emulator/training/losses.py](../emulator/training/losses.py) | 旧预测loss函数 → 10种最终loss＋物理分支监督＋命名消融 | MSE/dual/tail/slope可叠加；excess仍用全batch分母 | 保留默认完整监督、显式消融及原预测模式 |
| 新增 0→33 | [emulator/training/metrics.py](../emulator/training/metrics.py) | 无独立模块 → 4指标/窗口汇总 | 新增Train Peak、去重及finite检查 | 保留 |
| 修改 708→316 | [infer.py](../infer.py) | 旧checkpoint/逐年推理 → 当前checkpoint及共享engine，独立dual诊断 | 旧报告保留；新导出固定TRAIN阈值，scope/override仍有差异 | 按第五表保留检查、旧汇报和可选诊断 |
| 修改 330→340 | [infer.sh](../infer.sh) | 原tmux启动 → 显式Python/fallback，透传DUAL_DIAGNOSTICS并记录快照 | 可控制环境及独立诊断 | 保留 |
| 修改 289→295 | [infer_multi.sh](../infer_multi.sh) | 原RUNS启动 → 显式Python优先，透传诊断开关 | RUNS/目标组合保持；快照记录新增选择 | 保留 |
| 新增 0→74 | [preprocessing/forcing_cmip6.py](../preprocessing/forcing_cmip6.py) | 无 → 可选统一CLI，保存forcing NPY/NPZ，不输出p_mean | 原5入口仍在；压力空间去均值保持 | 保留可选入口，不恢复均值特征 |
| 修改 214→209 | [preprocessing/preprocessing_forcing_CMIP6_AWI_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_AWI_Mean_Removal.py) | 原forcing及额外均值输出 → 保留forcing，删除p_mean字段 | 路径、3h→6h抽样和NPY/NPZ保留；不再逐字相同 | 保留原入口，不恢复均值字段 |
| 修改 214→209 | [preprocessing/preprocessing_forcing_CMIP6_CNRM_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_CNRM_Mean_Removal.py) | 原forcing及额外均值输出 → 保留forcing，删除p_mean字段 | 路径、3h→6h抽样和NPY/NPZ保留；不再逐字相同 | 保留原入口，不恢复均值字段 |
| 修改 214→209 | [preprocessing/preprocessing_forcing_CMIP6_EC_EARTH_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_EC_EARTH_Mean_Removal.py) | 原forcing及额外均值输出 → 保留forcing，删除p_mean字段 | 路径、3h→6h抽样和NPY/NPZ保留；不再逐字相同 | 保留原入口，不恢复均值字段 |
| 修改 214→209 | [preprocessing/preprocessing_forcing_CMIP6_MPI_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_MPI_Mean_Removal.py) | 原forcing及额外均值输出 → 保留forcing，删除p_mean字段 | 路径、3h→6h抽样和NPY/NPZ保留；不再逐字相同 | 保留原入口，不恢复均值字段 |
| 修改 214→209 | [preprocessing/preprocessing_forcing_CMIP6_MRI_Mean_Removal.py](../preprocessing/preprocessing_forcing_CMIP6_MRI_Mean_Removal.py) | 原forcing及额外均值输出 → 保留forcing，删除p_mean字段 | 路径、3h→6h抽样和NPY/NPZ保留；不再逐字相同 | 保留原入口，不恢复均值字段 |
| 修改 172→159 | [preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py](../preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py) | 顶层处理及均值字段 → main/CLI和原forcing输出，删除p_mean | 安全import，原默认与主要warn/skip保留；额外字段减少 | 保留参数化与退役决定 |
| 修改 355→357 | [preprocessing/preprocessing_simulation.py](../preprocessing/preprocessing_simulation.py) | 原处理 → 原逻辑/默认恢复＋argv和--years | 默认仍2005，可明确选择其他年 | 保留扩展 |
| 修改 1008→931 | [preprocessing/time_align_unified.py](../preprocessing/time_align_unified.py) | 原fixed/peryear及p_mean历史传递 → 保留两种对齐/fallback和别名，删除均值图属性 | forcing/标签对齐保持；旧均值属性消费者需适配 | 保留主流程，不恢复已退役属性 |
| 新增 0→161 | [tests/test_config_interfaces.py](../tests/test_config_interfaces.py) | 无 → 原profile/390组合/loss/scheduler检查，删除p_mean专用断言 | 原扫描及10种loss有回归保护 | 保留 |
| 删除 271→0 | [tests/test_core_architecture.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_core_architecture.py) | 旧架构API测试 → 删除，由新测试覆盖部分功能 | 不能把旧每个接口视为仍被测试 | 恢复接口时补回相应案例 |
| 新增 0→246 | [tests/test_dual_experiments.py](../tests/test_dual_experiments.py) | 无 → 真实TRAIN先验、阈值解耦、5种模式/梯度及诊断/shell贯通 | 覆盖当前新增接口、边界及标签独立性；不是长期精度实验 | 保留 |
| 新增 0→158 | [tests/test_dual_loss.py](../tests/test_dual_loss.py) | 无 → 强制dual loss、warning、保存与梯度测试 | 验证三项确实参与训练 | 保留 |
| 新增 0→24 | [tests/test_forcing.py](../tests/test_forcing.py) | 无 → generic forcing公式测试 | 验证新可选入口数学 | 保留 |
| 删除 179→0 | [tests/test_inference_artifacts.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_inference_artifacts.py) | 旧shell快照/重放测试 → 删除 | 旧覆盖未完全由当前测试替代 | 建议适配后补回仍适用的行为测试 |
| 删除 116→0 | [tests/test_inference_audit.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_inference_audit.py) | 旧checkpoint/推理回退测试 → 删除 | 原部分scope/fallback不再支持 | 按决定保留的接口补回案例 |
| 新增 0→129 | [tests/test_inference_reporting.py](../tests/test_inference_reporting.py) | 无 → 当前逐年/分组/时间/格式测试 | 覆盖恢复后的科学汇报与样本权重 | 保留 |
| 删除 117→0 | [tests/test_missing_pmean.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_missing_pmean.py) | 旧缺压力均值回退测试 → 随功能退役删除 | 对应旧模式已不受支持 | 不恢复已退役功能的测试 |
| 删除 107→0 | [tests/test_model_audit.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/bb62a2297a0d37a35db5bff352ee815e05676c93/tests/test_model_audit.py) | 旧diagnostic/模型审计 → 删除 | attention等旧输出已删除 | 不恢复纯诊断测试；保留有效数学断言 |
| 新增 0→191 | [tests/test_models.py](../tests/test_models.py) | 无 → 当前encoder/head/H组合及梯度检查 | 覆盖新模型API与标签独立性 | 保留 |
| 新增 0→142 | [tests/test_pipeline.py](../tests/test_pipeline.py) | 无 → 从头训练/保存/推理及split回归 | 验证新checkpoint闭环 | 保留 |
| 新增 0→114 | [tests/test_preprocessing_pipeline.py](../tests/test_preprocessing_pipeline.py) | 无 → NetCDF/CSV/forcing/graph实际小样本管道 | 验证恢复后的默认与输出 | 保留 |
| 修改 101→124 | [tests/test_time_alignment.py](../tests/test_time_alignment.py) | 旧timestamp测试 → 保留并加fixed315、输出形状和fallback验证 | forcing历史与标签时间对齐受检；均值字段已删除 | 保留当前主流程测试 |
| 新增 0→96 | [tests/test_training.py](../tests/test_training.py) | 无 → 物理指标/单次forward/累积/存储隔离测试 | 防止精简后数值与输入污染回归 | 保留 |
| 修改 1522→213 | [train.py](../train.py) | 原大入口 → 模块化编排、TRAIN阈值/q_E拟合广播和元数据保存 | 旧协议大部保留；默认dual与消融分开记录，无p_mean | 按表逐项处理，不整体重写还原 |
| 修改 649→691 | [train.sh](../train.sh) | 原条件扫描 → 保留并接入当前dual/百分位/消融，删除p_mean | 原组合保留，run tag及快照标明新机制 | 保留，勿删组合 |
