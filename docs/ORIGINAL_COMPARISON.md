**StormSurge 与原版 Emulator：当前差异、影响和还原建议**

本报告对应独立仓库 [ZeshengLiu22/StormSurge](https://github.com/ZeshengLiu22/StormSurge) 的首次源码快照。原版基准是本地 `Emulator` 实际文件，对应 [PACT_Storm_Surge_Emulator 的提交 3d4be39](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/tree/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de)。比较生成时间见 [comparison_summary.json](audit/comparison_summary.json)。原目录预先存在的 `.gitignore` 本地修改也计入比较，未把它误认为该上游提交的内容。

**结论：主扫描使用的基础空间拓扑和 H=0 baseline 保留；PACT 包含已确认的新 dual head 与稳定性数学改动。此外仍有自定义模型接口、输入检查、运行开关和输出组织等差异，不能概括为“仅改了 dual/stability”。** 上一轮已经恢复的 DDP seed、归一化统计、验证补齐、loss 下限、环境配方和推理/预处理汇报，在下文按当前状态列出。

本次迁移只创建独立 Git 历史、复制当前代码/配置、更新 README 和审计文档。**表中的“建议还原”尚未执行**；没有趁迁移继续修改模型或训练设置。原版和原 v2 目录保留。

源码/config/既有文档比较共 **197 个路径：140 个相同、29 个修改、19 个新增、9 个删除，共57个差异文件**。不包含本报告、`docs/audit/`、数据、checkpoint、结果和缓存。逐文件hash及行数见 [CSV](audit/source_inventory.csv) / [JSON](audit/source_inventory.json)，完整行差异见 [original_to_current.diff](audit/original_to_current.diff)。文末列出全部57个文件。

建议标签含义：**保留**＝不建议退回旧实现；**还原**＝建议恢复该具体能力或语义；**部分还原**＝恢复便利性或实验接口，但保留明确错误检查；**已恢复/未变**＝没有新的还原工作。建议是代码与实验设计判断，不是精度提升的实验证明。

**一、基础模型：哪些没有改，哪些属于已确认的方法变化**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| H=0 baseline | GraphSAGE/CNN → spatial mean pooling → dropout → Linear(K) | 同一计算路径 | 不包含PACT attention、temporal或dual head；有效默认配置的原版数值对照一致 | **未变** |
| H>0 baseline | 各时刻空间编码/mean pool → 单层LSTM → 最后时刻 → Linear | 同一计算路径 | LSTM原来就存在；自定义temporal_hidden问题另列 | **未变** |
| GraphSAGE | 相同层数SAGEConv、LeakyReLU(.1)、dropout | 抽到SpatialEncoder，层和计算保留 | 没有换GNN种类、邻接或pooling | **保留模块拆分** |
| CNN | 3×3/padding1卷积，中间宽度29，最后输出hidden | 同样的卷积与默认通道 | 未改变原扫描的CNN参数预算；输入校验另列 | **未变** |
| PACT总连接 | spatial → station-query attention → lag/temporal → horizon attention → head | 总连接保持 | 新的归一化、残差公式和head会改变实际函数 | **保留总连接** |
| 相对lag与H=0 PACT | 当前forcing对应lag0；H=0 forecast context加horizon query | 保留 | 这条H=0 residual原来就有，不能宣称是新修复；也不能与H=0 baseline混淆 | **未变** |
| station attention输出 | 直接进入后续lag/temporal | 加FP32、无affine参数的LayerNorm | 改变表示尺度与梯度，影响PACT各H；不影响baseline | **保留**，属于已确认的稳定性设计 |
| head前context | attention/pressure拼接结果直接入head | 入head前再做FP32、无affine参数的LayerNorm | 改变head所见输入尺度及梯度 | **保留**，仍需完整扫描验证效果 |
| Transformer/MLP residual | PostLN残差，无当前可学习gain | PreLN，`x + exp(log_gain) × branch(LN(x))`，gain初值.1 | 数学改变；每分支gain参与Adam学习。.1是初始化，不是LR除10；不是clip | **保留**，不要为了代码对齐删掉已确认的方法变化 |
| LSTM/GRU temporal residual | `LayerNorm(x + dropout(RNN(x)))` | 保留 | 没有把RNN temporal也改成PreLN；其前后仍受到PACT新增norm影响 | **未变** |
| PACT head激活 | 隐层ReLU | LeakyReLU(.1) | 负区间保留梯度，避免ReLU完全截断；不证明它单独根治所有collapse | **保留** |
| dual预测结构 | `base + sigmoid(alpha) × sigmoid(gate) × tail`；tail可tanh截断，gate可window/horizon | 两个回归分支body/excess＋一个window事件gate：`body + p × excess` | body限制在TRAIN阈值以下，excess非负；去掉额外alpha，分工和输出约束改变 | **保留新dual**，按用户决定不重新启用旧head组合 |
| dual初始化 | 旧gate bias/alpha初始化与tail网络 | gate/excess末层权重置0；gate bias对应tail_frac先验，excess初始softplus输出.1 | 改变起始预测和学习路径；第一步gate/excess直接监督主要到末层 | **保留**，不把初始化当成已证明最优 |
| dual监督 | 只有所选最终预测loss，分支无独立语义约束 | 强制body/excess/gate三项dual loss；禁用或权重0会纠正为1并记录WARNING | **dual+mse已经不是只有MSE**，新增监督参与真实反传；single/baseline不添加 | **保留**，这是已确认的设计，报告精度时必须说明 |

新dual阈值由TRAIN窗口峰值的分位数确定，默认tail_frac=.05对应95%分位。事件是窗口内至少一个时刻超过阈值。body目标为 `min(y,τ)`，excess目标为 `(y−τ)+`，gate目标为事件0/1。Excess辅助项mask非事件窗口后仍按整个batch平均；gate使用BCE并乘TRAIN方差尺度。推理forward不读取未来标签。完整定义和局限见 [dual head说明](../DUAL_HEAD_EXPLAINED.md)。

原来的tail loss仍约束**最终预测在峰值窗口的误差**；slope loss约束**最终预测的一阶变化**。二者没有被三项dual loss代数替代，保留原扫描组合，并通过消融决定是否用于最终模型。不能为了保留dual的论文叙事预先断定新head一定更好。

**二、已恢复或一直保持的训练与实验口径**

| 项目 | 原来做什么 | 当前状态 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 原shell配置 | 127份配置，其中112份具体训练profile | 全部保留；125份逐字相同，2份common接入新head/dual loss；另有2份Stable | 原H/LR/loss/站点/数据源组合未被换掉 | **已保留**；不要删扫描轴 |
| 条件扫描 | LOSS/LR/H/WMSE_Q/TAIL_LAMBDA/SLOPE_LAMBDA/SLOPE_MASK_S，无关轴按原条件收缩 | 同样的条件笛卡尔积 | 390组合回归通过；configs_train_single指单GPU＋累积，不是single head | **未变** |
| 优化器 | Adam，weight_decay=1e-5，默认betas/eps | 保持 | 没有替换AdamW或修改LR/衰减值 | **未变** |
| Cosine设置 | linear warmup＋cosine；默认warmup5、起始倍率.1、min_lr1e-6 | 设置保持，内部由LambdaLR表达 | 6组逐epoch曲线最大差约6.1e-18；类名变化不等于LR改变 | **保留等价表达**，不必为类名恢复组合scheduler |
| ROP | 原有ReduceLROnPlateau及参数，按All或Peak物理RMSE | 保持可选；没有改成默认 | Val口径恢复后继续使用指定指标 | **未变** |
| DDP rank seed | `seed+rank` | 已恢复 | rank0/1实际观测为42/43；DDP仍同步模型权重 | **已恢复** |
| DDP sampler | 默认seed0，逐epoch set_epoch | 已恢复 | 保持原样本分配/排序；sampler seed不能按rank不同 | **已恢复** |
| robust/mag自动抽样 | x_nodes_per_graph<=0表示每TRAIN图最多256节点 | 已恢复 | 仅用于拟合归一化，模型仍接收完整图 | **已恢复** |
| X/Y统计 | 训练设备上按原FP32/FP64顺序求矩，DDP归约 | 已恢复设备、精度转换、平方/方差运算和归约 | 所测CPU/GPU/DDP统计与原函数最大差0；线程helper另列 | **已恢复** |
| augmentation RNG | probability>=1直接生成缩放/平移，不先抽概率 | 已恢复 | 所测输入和随后随机数一致；扰动没有关闭 | **已恢复** |
| DDP Val All | DistributedSampler补齐；FP32 batch均值、FP64加权累计/all-reduce | 已恢复 | 不整除时重复样本仍计入All，保持best epoch/ROP的原实验口径 | **已恢复**，本研究按用户决定保留此口径 |
| Val Peak | 原rank0完整验证集，按GT峰值选ceil(.05N)窗口 | 复用当前预测，去掉padding，恢复数据集顺序、原argsort/FP32归约 | 同一组预测时所测结果相同；batch内核/额外forward删除可能影响末位或后续RNG | **保留单次forward** |
| loss种类 | mse/wmse/mse_tail/wmse_tail/mse_wtail，各可加_slope | 10种最终预测loss均保留 | 物理单位与阈值用途保持；dual另加上表所列辅助项 | **未变** |
| loss小参数下限 | wmse_s、slope_mask_s至少1e-6；Charb ε、Huber δ至少1e-12 | 已恢复，包括有限的0/负数/极小正数 | 普通常用值不变；极小值不再改变旧公式的有效scale | **已恢复** |
| 梯度累积数学 | 同组microbatch等权，末组按实际microbatch数 | 保持 | 没有改成按不同microbatch样本数重新加权；DDP能力扩展另列 | **未变** |
| 数据划分 | 完整年份分组，TRAIN-only统计，shuffle/future规则 | 合法比例、标准输入时保持 | 上一轮真实Battery 24组split对照一致；异常比例验证更严格 | **保持正常语义** |

DDP seed与augmentation局部随机顺序恢复，不代表整次训练逐位回到原版。原dual训练为日志额外做过train-mode forward，旧Val诊断也额外遍历DataLoader；删除它们会改变后续RNG消耗。**不建议为复现旧随机轨迹恢复无用forward。** 比较方法时应在当前同一代码/环境/预算下重新训练各对照。

**三、仍待决定的模型与输入接口差异**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| metadata MLP隐宽 | `max(16,hidden)` | `hidden` | hidden8时16→8，是真架构变化；hidden128主扫描不受此项影响 | **还原，优先** |
| 自定义head宽度 | PACT构造支持head_hidden | 固定2×hidden | 默认一致，自定义实验能力丢失 | **还原参数能力**，保留当前默认实际宽度 |
| baseline temporal宽度 | 构造支持temporal_hidden | 固定hidden | 自定义LSTM宽度不可再复现 | **还原参数能力** |
| Python构造默认值 | spatial/head dropout=0、temporal dropout=.05；hidden必填 | .05/.05/0；hidden默认128 | 直接构造会不同；train.py显式传CLI/config，不依赖这些新默认 | **还原，优先**；不要反过来修改原训练CLI默认 |
| PACT pressure初始化顺序 | token层在head之后建立，global/token顺序与当前不同 | PressureFeatures先于head | tokens/both同seed初始权重对应关系改变；同权重数学等价 | **建议还原顺序**，若要严格对照pressure消融 |
| 模型API/权重键 | 多参数constructor，tensor或return_aux；旧类和state_dict命名 | ModelConfig、ForecastOutput、拆分模块 | 外部Python调用需适配；直接构造baseline还需指定head_type=single（CLI已自动处理）；不影响已验证的对应模块函数 | **保留清晰API**；用户不要求历史checkpoint兼容 |
| 输入H | PACT主要限制max_time_steps | 必须等于构造时history_steps+1 | 不能直接以不同H喂同一模型，推理同样检查H | **保留默认一致性检查**；确需跨H推理时另恢复显式能力 |
| baseline的max_time_steps校验 | 主要是PACT lag容量参数 | argparse对baseline也检查 | baseline没有该lag embedding，可能误拒绝本可运行的长H | **还原适用范围**，只校验使用该embedding的模型 |
| JSON文件名 | 精确/lower/upper依次查找 | 只读`<station>.json` | Battery请求无法读取仅有的battery.json | **还原别名查找** |
| 字段别名 | latitude/Latitude、longitude/Longitude、elevation/elev_m等可用 | 仅lat/lon/elevation_m | 有效旧输入被拒绝或elevation别名被忽略而使用0 | **还原别名**，保留数值有效性检查 |
| metadata缺失/非finite | 部分字段回退0；缺JSON可只用learned token | 经纬度必需且所有feature finite；启用metadata时缺JSON报错 | 改变异常/不完整数据接受范围。elevation_m缺省仍0；bathymetry启用时原来就必须finite | **部分还原**：可显式选择learned-token模式；不建议默默把坏经纬度当0 |
| pressure缺失 | 部分路径用零global编码或省略pressure token | 启用pressure且缺少所需历史时报错 | 旧数据/跨数据源输入可能失败；原无pressure主扫描不受影响 | **保留默认严格检查**；若研究缺失pressure，恢复明确的可选策略 |
| CNN网格检查 | batch计数等验证 | ptr.diff＋统一grid元数据检查 | 合法矩形PyG batch计算保持，异常输入接受范围不同 | **保留**，不为接受错误形状回退 |
| GraphStore构造 | 支持pattern/force_cpu/strict_station_filter/log_fn等；`*graphs.pt` | 简化接口、CPU存储；`*_graphs.pt` | 非标准文件名/外部直接调用可能失效；旧station索引helper移除 | **部分还原**：恢复实际使用的pattern能力；不必恢复纯包装函数 |
| View与tag | 旧metadata字段、无sample_id；空版本字段的tag格式不同 | 新sample_id，H0也提供x_hist，严格history/pressure长度 | 默认模型链路适配；外部消费tag或graph属性的脚本可能受影响 | **保留sample_id/检查**；需旧tag的分析脚本可恢复tag格式 |
| 模型维度来源 | 从首个TRAIN graph取in/out维度 | 从store第0个graph取 | 维度一致时相同；过滤外的首图不一致时会错误定模型尺寸 | **还原到TRAIN首图，优先** |

**四、其他训练执行、CLI和性能差异**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 输入归一化存储 | 原地修改tensor | 赋新tensor | 避免输入共享存储被污染；正常PyG batch数值保持。尚不能据此认定它是旧collapse唯一根因 | **保留** |
| 未显式启用TF32 | 不改后端既有默认；启用时设TF32和matmul precision high | 显式设置matmul/cuDNN TF32为开关值；没有单独precision high调用 | 不传--tf32时与原后端默认可能不同；现有TF32=1 profile不受该关闭差异 | **建议还原未指定时的原默认语义**；显式选择仍保留 |
| 确定性与benchmark | 无当前统一deterministic选项 | 可选严格确定性，默认0；显式benchmark=False | 影响复现与内核选择；严格确定性可能明显降速 | **保留可选，默认不强制开启**；记录环境即可 |
| 临时统计线程 | NumPy统计阶段设置OMP/MKL并恢复，含CPU数推断 | 去掉helper，保留torch_threads | 一次性统计耗时可能不同；环境变量中途设置的实际收益需测量 | **不机械还原**，确认瓶颈后再处理 |
| threshold是否计算 | 最终loss需要时才拟合 | 每种mode都先计算peak与pointwise两个阈值 | baseline/single纯MSE增加不必要统计；dual必须有TRAIN阈值 | **恢复按需计算**，dual始终计算其必要阈值 |
| threshold广播 | FP32 tensor广播后使用 | Python对象广播，loss使用时按tensor dtype处理 | 常用FP32预测loss已对照；保存的标量精度和广播开销不同 | **可恢复小tensor广播**，属于低优先级实现一致性 |
| tail_frac极端边界 | threshold函数先将tail_frac夹到[1e-6, .999999]，空输入返回0阈值 | CLI要求0<tail_frac<1，但函数不再夹到原下限/上限；空输入报错 | 默认.05不变；1e-9等合法极端正值的阈值不再等价 | **建议恢复阈值函数的原边界**；dual的事件先验有效性仍需单独检查 |
| mag校验范围 | mag只检查其使用的p_hi；p_lo不参与 | 合并函数也要求p_lo<p_hi | 可能拒绝原先可用的mag设置，例如p_lo=95、p_hi=90 | **还原按mode校验**，不要检查该mode不使用的参数 |
| DDP梯度累积 | Python拒绝world>1且accum>1 | engine支持no_sync累积；shell仍保持原单进程累积限制 | 新能力不改变原shell扫描；直接Python有更大允许范围 | **保留能力**；使用前单独验证该组合 |
| final test执行 | 原DDP test评估、rank0再导出预测 | rank0一次计算指标并导出 | 无重复forward；不再把DDP padding计入最终Test。可能与旧Test打印数值不同 | **保留**，测试集每个真实样本一次更合适；Val按已决定的原口径 |
| Train指标 | 原按batch累计Train RMSE/MAE，另有诊断 | 在线预测按sample_id整理All/Peak5；DDP重复训练样本在报告中只计一次 | Train报告的加权/归约与旧版不同；优化仍使用sampler补齐数据 | **保留并说明**；Train在线指标不能当作固定模型的train-eval |
| Peak数据收集 | 原rank0独立full pass＋aux诊断 | 各rank每窗口4标量，gather后计算指标 | 少forward，多一次窗口记录收集；内存/通讯随窗口数增长 | **保留**，大数据瓶颈可后续优化收集，不恢复诊断forward |
| 空/非法指标 | 原部分输出NaN，缺少当前统一检查 | 空split为None，非finite预测/目标报错 | 更早暴露坏结果；无自动重训/回滚 | **保留** |
| 参数解析组织 | 大train.py内argparse，canonical函数部分strip空白 | 独立arguments.py，新增必要检查，部分不再strip | 原常规CLI保留；负warmup、非法split等边界更严格；空白别名接受范围收窄 | **部分还原**：恢复无害的strip/别名；保留不合法训练参数检查 |
| 新/旧参数 | 旧tail_tanh_clip/gate_bias_init/alpha_init_logit，window/horizon gate | 去掉旧3参数；gate_mode只window；新增dual/stability/device/metadata选项 | 旧head命令不能原样运行；对应方法已按用户决定替换 | **保留新head参数语义**；不要重新接回旧残差head |
| 固定值flag | 没有当前stable_arch/dual_mode | stable_arch只接受1，dual_mode只接受exceedance | 无实际选择作用，主要为已有配置声明服务 | **可后续删冗余flag**，需同步全部配置/launcher；本次不删 |
| Slurm与GPU映射 | 识别SLURM_*、master推断、visible-device映射 | Python主要使用RANK/LOCAL_RANK/WORLD_SIZE | 直接srun与非标准GPU映射不兼容；torchrun路径保持 | **保留现状**，用户明确暂不需要Slurm |
| shell便利功能 | tmux/conda/扫描/快照 | 保留并加DRY_RUN、显式PYTHON_BIN、直接执行fallback、scheduler参数透传 | 默认参数不变；过去shell里未生效的自定义scheduler值现在可生效 | **保留**，报告实际解析参数 |
| 梯度裁剪 | 无当前max_grad_norm选项 | 可选普通clip，默认0 | 默认训练未启用裁剪；指定正值时是额外训练设置 | **保留可选**，不默认打开 |

“旧retry/rollback guard”属于此前v2中间试验的历史，不是这里原版基准与当前代码的必然差异。当前没有自动重试、回滚、跳过异常epoch或恢复常数输出的执行路径；不要将它们列为原版必须还原的基础训练设置。

原/现argparse声明的机器清单见 [argparse_comparison.json](audit/argparse_comparison.json)：76项声明保持，删除3个旧head参数，增加10项，另3项的canonical函数或choices改变。声明一致不覆盖后续校验/生效逻辑，因此上表另外列了baseline容量、mag、tail_frac等边界。

**五、推理、日志、checkpoint与输出**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 训练诊断 | gate/alpha/attention/归一化误差等，含额外计算 | 仅Train/Val×All/Top5%×RMSE/MAE，物理单位 | 精简符合用户要求；减少诊断耗时；特殊gate校准研究需另做离线分析 | **保留精简** |
| 时间输出 | UTC文本、epoch time等 | `[Date|Time]`＋最终wall time，summary保存循环时间与wall_seconds | wall包含准备/最终测试；与逐epoch或单forward时间不是同一口径。日志用本地datetime，当前机器UTC | **保留**；跨时区比较需注意时间标签 |
| 训练artifact命名 | 原可读tag/MD5、旧summary NPZ等 | 参数SHA256 stem、JSONL/JSON、独立配置和压缩test_preds | 旧分析脚本可能需要适配，不影响loss；每组合结果较容易追踪 | **保留当前主格式**，有具体消费者时可加旧格式导出 |
| 同目录重复运行 | 原结果可被同名覆盖 | config使用exclusive创建，相同显式tag/参数会报FileExistsError | 原命令重跑可能需要新run_tag/目录；默认生成新的时间标签 | **保留防覆盖行为**，复跑使用独立run目录 |
| JSON写入 | 部分原helper用atomic替换 | 部分直接write_text | 中断时可能留下不完整报告 | **建议还原关键JSON的原子写入**，无需恢复全部IO包装层 |
| checkpoint | 原schema与模型键；推理重建旧模型 | 新model_config/model_state/normalization，保存station_feat和精确split_tags | 旧checkpoint不可载入；新训练可恢复同一评估人口 | **保留**，用户明确从头重训 |
| 默认推理样本 | 从checkpoint args对当前目录重新划分年份 | 按训练时保存的test tags | 目录增加/缺失年份时结果不同；避免推理时悄悄更换test population | **保留saved tags**；需要重划分应明确另选scope |
| 推理override | 可覆盖部分架构/H；缺station某些路径fallback全站点 | 模型/H参数用于验证checkpoint；station严格筛选 | 旧命令灵活性减小，减少错误模型/站点混用 | **保留默认检查**，跨H实验另设明确接口 |
| 推理metadata | 同站点可重新读JSON | 同站点使用checkpoint保存特征，换站点才读取新JSON | JSON后来修改时，不会无意改变原checkpoint条件 | **保留**；若要覆盖，做显式实验选项 |
| 逐年/分组报告 | 每年RMSE/MAE/runtime；ALL/past/future；平均年耗时排除2014_2015 | **已恢复**，save_npz不影响报告是否计算 | past按起始年1979–2014，future2070–2099；其他年只进入ALL；2014_2015仍进入精度统计 | **已恢复** |
| 旧推理文件/目录 | metrics_per_year…json、preds…ALLYEARS.npz及标签/目录规则 | **已恢复**；另保留metrics.json/predictions.npz | 旧报告消费者可读；保存预测时有额外一份当前格式副本 | **保留已恢复的旧格式**；确认消费者后才删冗余副本 |
| 逐年归约数值 | 原每batch FP32误差sum、再累计 | 共享engine用FP32 batch均值及FP64权重；总体分组仍用FP64数组 | 数学定义相同，GPU/批次舍入末位可能不同 | **通常不需还原**；要逐位对照特定旧报告时再恢复该归约 |
| inference模块API | 独立infer_one_loader返回tuple | infer.py调用共享run_epoch，grouping包保留分组/标签函数 | 旧Python导入路径不可用，但科学汇报功能已回到当前CLI | **不恢复重复forward实现**；确有外部调用时可提供薄适配 |

**六、环境、预处理与仓库组织**

| 项目 | 原来做什么 | 现在做什么 | 差异的影响 | 是否建议还原 |
|---|---|---|---|---|
| 训练环境配方 | PyTorch2.8+cu128及原PyG扩展 | **原文件逐字恢复** | 此前2.6/cu124的配方改动不再存在；现有已安装环境未被重装 | **已恢复** |
| dataprep配方 | 原完整依赖/版本列表 | **原文件逐字恢复** | 不再擅自更改Python/NumPy/pandas/torch配方 | **已恢复** |
| 五个CMIP6入口 | AWI/CNRM/EC_EARTH/MPI/MRI独立脚本、各自默认路径与NPY/NPZ | 五个原脚本逐字恢复 | 原命令/默认路径/输出保留 | **已恢复** |
| generic CMIP6 CLI | 无统一额外入口 | 新增forcing_cmip6.py，显式路径/模型参数；同样保存NPY/NPZ | 额外便利入口，没有替代原5脚本；直接函数对异常输入更严格 | **保留可选** |
| NCEP入口 | 顶层脚本，固定默认路径/年份；NPY/NPZ | main/CLI参数化且可安全import；原默认、NPY/NPZ、主要warn/skip恢复 | 同数据核心变换保持，可显式换路径；打印细节未完全照抄 | **保留参数化**，不恢复import即处理数据 |
| simulation | 完整mesh/station/时间处理，默认year_list=[2005] | 原逻辑/默认路径/默认2005恢复；额外支持--years及可传argv | 不再默认发现所有年份；批量年需显式指定 | **保留**，将来改默认年份应作为明确行为变更 |
| time alignment | fixed315/peryear函数、两个out_root参数、CSV缺time回退、缺年skip | 已恢复；只新增--out_root作为peryear别名 | 原main的fixed调用本就注释，当前默认也只跑peryear | **保留恢复结果和别名** |
| LICENSE与静态资源 | 原许可证、station JSON等 | 对应文件逐字保留 | 仓库独立不删除原授权/归属信息 | **保留** |
| .gitignore | 原工作目录另有本地/experiment_configs忽略项 | 当前不包含该本地忽略项；大数据/结果/checkpoint仍忽略 | 不影响模型；本地未提交ignore策略与源码版本应分开理解 | **不必自动还原本地项** |
| README/方法说明 | 原大README及changelog | 当前README、新dual/stability说明；旧changelog移除 | 新入口更清楚，但历史记录变少 | **保留新说明**；若需要历史追溯，建议将旧changelog作为历史文档归档 |
| 测试 | 旧模型/诊断/推理接口测试 | 新模型/梯度/dual loss/配置/预处理/报告/训练往返测试 | 42项通过不等于所有旧接口都覆盖；缺失pressure等旧语义测试被删除与当前严格策略一致 | **保留当前测试**；恢复某项接口时同步恢复相应测试 |
| 独立仓库 | 原Emulator自己的Git历史/remote | StormSurge独立初始提交，origin仅指向新repo | 不向原版推送，不迁移旧历史；当前188份源码/config/原有文档完整复制 | **保留独立仓库** |
| 数据与机器路径 | profile引用外部Data/graph/station目录 | 原profile路径保持，数据未打包进Git | Git独立不意味着自动复制大数据；换机器需要设置ROOT_DIR、STATION_JSON_DIR、Python环境等 | **保留实验配置值**，按部署机器显式覆盖路径 |

**七、建议优先顺序与验证范围**

后续优先还原的是非方法创新引起的接口收缩：metadata最小隐宽、自定义head/temporal宽度、直接构造默认值、JSON文件名/字段别名、维度从TRAIN首图读取，以及baseline不该受PACT lag容量限制的问题。pressure建层顺序可在对应消融前恢复。tail_frac原边界、mag按mode校验、未显式配置TF32时的原行为与关键JSON原子写入也值得恢复。

建议继续保留新dual＋强制dual loss、已确认的PACT稳定性数学、输入存储隔离、单次forward指标、严格的默认评估人口/站点检查、新checkpoint格式和全部原扫描组合。不能为了“与原版一样”把已恢复的报告或原配置再次删除。

本次在独立目录重新执行 `python -m unittest discover -s tests`：**42/42通过**，包括112份原训练profile＋2份Stable、390条件组合、loss/梯度、预处理和新checkpoint训练/推理往返。日志：[publication_tests.txt](audit/evidence/publication_tests.txt)。

以下数值证据来自迁移前的近期验证；相关生产代码与迁移快照一致，迁移没有重跑完整模型扫描：

| 验证 | 对照方式与结果 | 证据 |
|---|---|---|
| H0/H48 baseline | 实际原版：16组CPU，2 encoder×pressure开关×train/eval；初始化、预测、参数梯度所测差0 | [model_equivalence.json](audit/evidence/model_equivalence.json) |
| PACT single | 实际原版对象只加入已确认stability后，与当前128组CPU对照；同权重预测/梯度所测差0 | 同上；不表示未经修改的原版PACT与当前相同 |
| 新dual清理等价 | 清理前已确认的新dual vs 当前32组CPU，所测差0 | 同上；这不是与旧residual dual等价 |
| GPU模型 | H100，16组FP32/BF16，baseline/PACT、两encoder、H0/H48；同权重所测差0 | [gpu_equivalence.json](audit/evidence/gpu_equivalence.json) |
| Cosine曲线 | 6组含300epoch、短运行、warmup边界，最大LR差6.1e-18 | [scheduler_comparison.json](audit/evidence/scheduler_comparison.json) |
| 恢复后统计/RNG/loss/Val | CPU、H100、两rank CPU/Gloo：统计与augmentation局部RNG所测差0；每种环境100组小scale loss差0，梯度差≤1.87e-9；Val补齐/Peak对照一致 | [CPU](audit/evidence/protocol_cpu_1ranks.json)、[GPU](audit/evidence/protocol_cuda_1ranks.json)、[DDP](audit/evidence/protocol_cpu_2ranks.json) |
| 实际入口短训练 | 两轮H48 dual：CPU DDP使用seed42/43、sampler0；H100 BF16/TF32完成反传、保存、最终推理 | [DDP](audit/evidence/smoke_ddp.txt)、[GPU](audit/evidence/smoke_gpu.txt) |

测试使用现有 **PyTorch2.6+cu124** 安装环境；恢复的2.8+cu128 YAML没有在本次被重新安装验证。只有一张H100，两个rank的通讯验证使用CPU/Gloo，没有进行两GPU/NCCL测试。短训练、同权重对照、函数级一致性分别证明不同层面的性质；它们**不能证明长期stability已彻底根治，也不能证明新dual比single或GNN比CNN更好**。

更新逐文件比较可运行：

```bash
python docs/audit/compare_original.py --original /path/to/Emulator
```

脚本重新生成hash清单、计数和完整diff；本报告的语义解释及建议需随实际代码变化人工更新。发布时的v2来源hash在 [v2_source_hashes.json](audit/v2_source_hashes.json)，迁移核对在 [migration_verification.json](audit/migration_verification.json)。

**八、全部57个差异文件的索引**

下面的索引逐个覆盖生成清单中的修改/新增/删除文件。语义影响与还原建议在上表对应项目展开；行数变化本身不等于架构或训练数学变化。

<!-- FILE_INDEX -->

| 状态/行数 | 文件 | 原来 → 现在 | 影响 | 建议 |
|---|---|---|---|---|
| 修改 52→49 | [.gitignore](../.gitignore) | 原目录本地忽略experiment_configs → 未继承该本地项 | 版本管理范围不同，无训练影响 | 不自动还原本地忽略项 |
| 新增 0→183 | [DUAL_HEAD_EXPLAINED.md](../DUAL_HEAD_EXPLAINED.md) | 无 → 新dual的公式、监督、推理与限制说明 | 明确方法含义及不能保证的结论 | 保留 |
| 修改 483→132 | [README.md](../README.md) | 原综合README → 当前使用说明、独立仓库来源与本报告入口 | 运行说明与当前实现对应，历史细节减少 | 保留；历史内容可归档 |
| 新增 0→40 | [STABILITY_AND_DUAL_HEAD.md](../STABILITY_AND_DUAL_HEAD.md) | 无 → 当前stability/dual与精简边界说明 | 区分结构、确定性与日志 | 保留 |
| 删除 108→0 | [changelog.md](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/changelog.md) | 旧变更历史 → 删除 | 失去本地历史说明 | 建议作为历史文档归档 |
| 修改 94→97 | [configs/configs_train/train_config_common.sh](../configs/configs_train/train_config_common.sh) | 旧head参数 → 新head、mandatory dual loss与运行选项 | 原扫描超参数保持，dual方法改变 | 保留新dual配置 |
| 修改 95→98 | [configs/configs_train_single/train_config_common.sh](../configs/configs_train_single/train_config_common.sh) | 旧单GPU公共配置 → 同样接入新head/运行选项 | 单GPU累积语义保留 | 保留 |
| 新增 0→5 | [configs/train_config_NCEP_Battery_Stable_Dual.sh](../configs/train_config_NCEP_Battery_Stable_Dual.sh) | 无 → 新dual对照profile | 新增实验入口，引用Single公共条件 | 保留 |
| 新增 0→32 | [configs/train_config_NCEP_Battery_Stable_Single.sh](../configs/train_config_NCEP_Battery_Stable_Single.sh) | 无 → 新single对照profile，修正移动后的source路径 | 新增对照且已恢复可执行，不替代原profile | 保留 |
| 修改 28→3 | [emulator/common/__init__.py](../emulator/common/__init__.py) | 导出多个runtime/DDP/IO helper → configure_runtime | 外部旧导入需适配 | 保留精简；必要时薄适配 |
| 修改 12→20 | [emulator/common/cli.py](../emulator/common/cli.py) | bool解析 → 保留bool并加入temporal名称解析 | attn别名保留，部分strip空白能力丢失 | 恢复无害strip |
| 删除 53→0 | [emulator/common/distributed.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/emulator/common/distributed.py) | rank/DDP/all-reduce/print包装 → 删除并直接调用torch | Slurm能力及旧公开导入不同 | 保留精简，Slurm暂不还原 |
| 修改 72→72 | [emulator/common/inference_artifacts.sh](../emulator/common/inference_artifacts.sh) | 原快照变量 → 加入PYTHON_BIN/USE_TMUX | 重放包含新增运行选择 | 保留 |
| 删除 26→0 | [emulator/common/io_utils.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/emulator/common/io_utils.py) | atomic JSON helper → 删除，调用方直接写 | 中断时报告可能不完整 | 建议恢复关键写入的原子性 |
| 修改 99→26 | [emulator/common/runtime.py](../emulator/common/runtime.py) | 分散seed/线程/标签helper → 统一runtime及时间戳 | TF32默认、确定性控制、临时线程不同 | 恢复未指定TF32语义；其余按第四表 |
| 修改 43→7 | [emulator/data/__init__.py](../emulator/data/__init__.py) | 旧store/split/stats导出 → 简化新接口 | 外部Python导入改变 | 不为旧checkpoint恢复全部包装 |
| 修改 351→92 | [emulator/data/graph_store.py](../emulator/data/graph_store.py) | 宽泛加载/独立split helper → store.split与严格view | pattern、缺失输入、tag和自定义接口不同 | 按第三表部分还原 |
| 修改 118→26 | [emulator/data/normalization.py](../emulator/data/normalization.py) | 原地修改输入 → 新tensor赋值；原augmentation RNG已恢复 | 避免共享存储污染，正常输入数值保持 | 保留赋值方式 |
| 修改 103→25 | [emulator/data/station_metadata.py](../emulator/data/station_metadata.py) | 别名/缺省解析与encoder → 严格解析，encoder并入模型 | 丢失有效旧字段/文件名兼容，hidden最小宽度另变 | 恢复别名/原隐宽，保留finite检查 |
| 修改 310→110 | [emulator/data/stats.py](../emulator/data/stats.py) | 多个stats函数 → 统一拟合，原主要数值语义已恢复 | 线程、threshold按需/边界、mag校验仍不同 | 按第二/四表处理 |
| 修改 10→5 | [emulator/inference/__init__.py](../emulator/inference/__init__.py) | 导出独立推理engine及grouping → grouping/标签函数 | 旧infer_one_loader公开入口不再存在 | 共享engine保留，按需薄适配 |
| 删除 103→0 | [emulator/inference/engine.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/emulator/inference/engine.py) | 独立forward/计时 → 删除，复用run_epoch | 逐年汇报已接回infer.py，归约末位可能不同 | 不恢复重复forward实现 |
| 修改 30→46 | [emulator/inference/grouping.py](../emulator/inference/grouping.py) | 年份/分组helper → 保留并接回dataset标签函数 | 原source/target和past/future规则恢复 | 保留 |
| 修改 25→4 | [emulator/models/__init__.py](../emulator/models/__init__.py) | 旧模型类导出 → ModelConfig/build_model/PACT/Baseline/ForecastOutput | 公开API变化 | 保留当前API |
| 修改 891→157 | [emulator/models/architectures.py](../emulator/models/architectures.py) | 单个大模型文件 → 组合独立模块 | 含已确认方法改变及部分误改接口 | 保留方法；还原第三表的非方法变化 |
| 新增 0→60 | [emulator/models/heads.py](../emulator/models/heads.py) | 无独立文件 → 单头与新exceedance head | 新dual结构、初始化、输出契约 | 保留新head，恢复自定义宽度能力 |
| 新增 0→30 | [emulator/models/pressure.py](../emulator/models/pressure.py) | 模型内pressure分支 → 独立PressureFeatures | 合法同权重数学保持；初始化顺序/缺失策略不同 | 考虑还原初始化顺序，策略显式决定 |
| 新增 0→49 | [emulator/models/spatial.py](../emulator/models/spatial.py) | 模型内空间encoder → 独立SpatialEncoder | 同层计算保持，CNN检查改变 | 保留 |
| 新增 0→48 | [emulator/models/temporal.py](../emulator/models/temporal.py) | 模型内temporal → 独立4种block | MLP/Transformer PreLN+gain，RNN residual保持 | 保留已确认稳定性数学 |
| 修改 13→5 | [emulator/training/__init__.py](../emulator/training/__init__.py) | 多个epoch函数 → ForecastLoss/run_epoch/metric接口 | 训练循环复用，公开导入变化 | 保留 |
| 新增 0→259 | [emulator/training/arguments.py](../emulator/training/arguments.py) | 无独立文件 → 从train抽出parser并加入新选项 | 原正常CLI保留，部分验证和边界改变 | 恢复误收紧的适用范围/别名 |
| 修改 586→99 | [emulator/training/engine.py](../emulator/training/engine.py) | 训练/验证/预测/诊断多个循环 → 一次forward共享循环 | 删除诊断计算；Val原口径恢复；Train/Test报告另有差异 | 保留单循环，明确报告口径 |
| 修改 86→97 | [emulator/training/losses.py](../emulator/training/losses.py) | 旧预测loss函数 → ForecastLoss＋dual_loss_terms | 预测10模式及floor恢复，新dual加入真实辅助反传 | 保留强制dual loss和原预测模式 |
| 新增 0→33 | [emulator/training/metrics.py](../emulator/training/metrics.py) | 无独立模块 → 4指标/窗口汇总 | 新增Train Peak、去重及finite检查 | 保留 |
| 修改 708→294 | [infer.py](../infer.py) | 旧checkpoint模型重建/逐年推理 → 新checkpoint＋共享engine | 旧科学报告/文件名已恢复，输入scope/override不同 | 按第五表保留默认检查/旧汇报 |
| 修改 330→336 | [infer.sh](../infer.sh) | 原tmux启动 → 加PYTHON_BIN和直接执行fallback | 更容易显式选择环境 | 保留 |
| 修改 289→291 | [infer_multi.sh](../infer_multi.sh) | 原RUNS启动 → 增加显式Python优先 | RUNS和目标组合保持 | 保留 |
| 新增 0→75 | [preprocessing/forcing_cmip6.py](../preprocessing/forcing_cmip6.py) | 无 → 额外统一参数化CLI | 原5入口仍在；可选新入口也输出NPY/NPZ | 保留可选入口 |
| 修改 172→163 | [preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py](../preprocessing/preprocessing_forcing_NCEP_Mean_Removal.py) | 顶层处理 → main/CLI，NPY及默认/skip已恢复 | 可安全import，打印细节不同 | 保留参数化 |
| 修改 355→357 | [preprocessing/preprocessing_simulation.py](../preprocessing/preprocessing_simulation.py) | 原处理 → 原逻辑/默认恢复＋argv和--years | 默认仍2005，可明确选择其他年 | 保留扩展 |
| 修改 1008→1008 | [preprocessing/time_align_unified.py](../preprocessing/time_align_unified.py) | 原两个输出参数 → 已恢复并加--out_root别名 | 原fixed/peryear及fallback保留 | 保留 |
| 新增 0→207 | [tests/test_config_interfaces.py](../tests/test_config_interfaces.py) | 无 → 全profile、390组合、loss/pressure/scheduler检查 | 防止配置/训练语义回归 | 保留 |
| 删除 271→0 | [tests/test_core_architecture.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/tests/test_core_architecture.py) | 旧架构API测试 → 删除，由新测试覆盖部分功能 | 不能把旧每个接口视为仍被测试 | 恢复接口时补回相应案例 |
| 新增 0→158 | [tests/test_dual_loss.py](../tests/test_dual_loss.py) | 无 → 强制dual loss、warning、保存与梯度测试 | 验证三项确实参与训练 | 保留 |
| 新增 0→23 | [tests/test_forcing.py](../tests/test_forcing.py) | 无 → generic forcing公式测试 | 验证新可选入口数学 | 保留 |
| 删除 179→0 | [tests/test_inference_artifacts.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/tests/test_inference_artifacts.py) | 旧shell快照/重放测试 → 删除 | 旧覆盖未完全由当前测试替代 | 建议适配后补回仍适用的行为测试 |
| 删除 116→0 | [tests/test_inference_audit.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/tests/test_inference_audit.py) | 旧checkpoint/推理回退测试 → 删除 | 原部分scope/fallback不再支持 | 按决定保留的接口补回案例 |
| 新增 0→129 | [tests/test_inference_reporting.py](../tests/test_inference_reporting.py) | 无 → 当前逐年/分组/时间/格式测试 | 覆盖恢复后的科学汇报与样本权重 | 保留 |
| 删除 117→0 | [tests/test_missing_pmean.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/tests/test_missing_pmean.py) | 旧缺pressure回退测试 → 删除 | 当前严格策略与旧预期冲突 | 若恢复可选回退则补回 |
| 删除 107→0 | [tests/test_model_audit.py](https://github.com/BinaLab/PACT_Storm_Surge_Emulator/blob/3d4be39c48214cbcd3afbdc51ec48f7fb839e0de/tests/test_model_audit.py) | 旧diagnostic/模型审计 → 删除 | attention等旧输出已删除 | 不恢复纯诊断测试；保留有效数学断言 |
| 新增 0→113 | [tests/test_models.py](../tests/test_models.py) | 无 → 当前encoder/head/H组合及梯度检查 | 覆盖新模型API与标签独立性 | 保留 |
| 新增 0→135 | [tests/test_pipeline.py](../tests/test_pipeline.py) | 无 → 从头训练/保存/推理及split回归 | 验证新checkpoint闭环 | 保留 |
| 新增 0→111 | [tests/test_preprocessing_pipeline.py](../tests/test_preprocessing_pipeline.py) | 无 → NetCDF/CSV/forcing/graph实际小样本管道 | 验证恢复后的默认与输出 | 保留 |
| 修改 101→122 | [tests/test_time_alignment.py](../tests/test_time_alignment.py) | 旧timestamp测试 → 保留并加fixed315/pressure检查 | 覆盖fixed路径和原CSV fallback | 保留 |
| 新增 0→96 | [tests/test_training.py](../tests/test_training.py) | 无 → 物理指标/单次forward/累积/存储隔离测试 | 防止精简后数值与输入污染回归 | 保留 |
| 修改 1522→204 | [train.py](../train.py) | 原大入口 → 模块化训练编排 | 原协议大部已恢复；新增dual、runtime/API/artifact差异仍在 | 不整体重写还原；按表逐项处理 |
| 修改 649→708 | [train.sh](../train.sh) | 原条件扫描启动 → 保留并接入新head、参数透传、dry-run和wall time | 原组合保留，新增便利功能/同名策略 | 保留，勿删组合 |
