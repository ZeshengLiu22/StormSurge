# 新 dual head：从监督目标到推理公式

这是待验证的方法设计，不是已经证明优于 single head 的结论。代码选项为
`--head_type dual`。它包含两个回归分支和一个事件概率 gate，
共享空间 encoder、temporal block 和 horizon readout；不是训练两个完整的 GNN。

## 1. 先明确要拆开的量

令 X 表示 forcing history、空间结构和站点信息，K=6 表示输出时刻数。
这里的 K 不等于 history 长度 H。令 Y_h 为第 h 个输出时刻的目标 surge，单位为 m。

阈值 τ 仅从 TRAIN 计算：先对每个训练窗口取 max_h Y_h，再取这些最大值的第 `exceedance_percentile` 百分位（默认95）。
该参数与最终预测 tail loss 的 `tail_frac` 独立；改变 tail-loss 子集不会改变 decoder 的事件定义。
事件 E 是“这个预测窗口中至少有一个输出超过 τ”：

\[
E=\mathbf 1\{\max_hY_h>\tau\}.
\]

真实 TRAIN 事件比例为 `q_E = mean(max_h Y_h > τ)`，可能因为分位点 ties 而不同于 .05。
可学习 gate 的 bias 初始化为 `logit(clip(q_E, 1e-6, 1-1e-6))`；截断只保证初始化有限。
checkpoint 的 `dual_metadata` 保存物理 `tau_phys`、原始 `event_prior=q_E`、实际 `gate_init_prior`、
事件数、TRAIN 窗口数、百分位和消融模式。归一化后的 τ′ 仍随 model_config/state_dict 保存。
全相同标签等无事件 TRAIN 可以得到 q_E=0；不把它伪装成5%。

逐时刻把真值精确拆成两部分：

\[
C_h=\min(Y_h,\tau),\qquad D_h=(Y_h-\tau)_+,\qquad Y_h=C_h+D_h.
\]

C 是截顶后的水平；D 是超过阈值的正增量。它们不是 tide/surge 的物理分解。
本方案关注正向极端 surge，尚不包含负向极值的独立分支。

例如 τ=0.8 m：

| 输出时刻 h | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| 真值 Y | .35 | .60 | .95 | 1.20 | .70 | .40 |
| Body 标签 C | .35 | .60 | .80 | .80 | .70 | .40 |
| Excess 标签 D | 0 | 0 | .15 | .40 | 0 | 0 |

该窗口 E=1。Excess 分支也要学习事件窗口内那些未超阈时刻的零值，才能学习峰出现的位置和形状。

## 2. 三条分支具体学习什么

共享网络产生每个输出时刻的表示 c_h；两个回归 MLP 分别处理 c_h。
window gate 使用整个预测窗口的 context 汇总。统计上的目标是：

\[
b_h(X)\approx\mathbb E[C_h\mid X],
\]
\[
r_h(X)\approx\mathbb E[D_h\mid X,E=1],
\]
\[
p(X)\approx\Pr(E=1\mid X).
\]

Body 学全部样本的 capped level，包含事件窗口；不是只用非事件样本训练。
Excess 只在事件窗口上接受直接的 excess 标签监督。
Gate 在全部窗口上接受事件/非事件二分类监督。

三个输出因此分别回答：截顶后的平均水平是多少、若发生事件增量轨迹是什么、事件有多大概率发生。

## 3. 为什么是 b + p r

因为 E=0 时所有 D_h 都等于零，全期望公式给出：

\[
\begin{aligned}
\mathbb E[Y_h\mid X]
&=\mathbb E[C_h\mid X]+\mathbb E[D_h\mid X]\\
&=\mathbb E[C_h\mid X]+\Pr(E=1\mid X)\mathbb E[D_h\mid X,E=1].
\end{aligned}
\]

因此最终预测为：

\[
\boxed{\widehat Y_h=b_h(X)+p(X)r_h(X)}.
\]

这里不能再给 b 乘 (1-p)：b 已经是全部情形下 capped level 的条件均值。
也不能在全部窗口上用包含大量零值的 D 直接训练 r 后，再乘 p：那样 r 已经在学
无条件于事件的 E[D|X]，再次乘概率会把极端增量压低两次。

一个自洽的概率例子：给定同样的 X，假设 40% 情形 Y=.55 m，60% 情形 Y=1.20 m，τ=.8 m。
则 p=.6，b=.4×.55+.6×.8=.70，r=1.20-.8=.40。
最终预测 .70+.6×.40=.94 m，正好等于 .4×.55+.6×1.20。
这也说明预测的是条件均值，而不是“只要可能发生事件，就输出事件的峰高”。

window gate 的一个 p 作用于整个六步轨迹，在统计上是成立的；r_h 负责逐时刻的差异。
当前实现只保留 window gate，不再提供 horizon gate 分支。

## 4. 完整训练目标

对大小 B 的 batch，在物理单位下计算：

\[
L_{\mathrm{pred}}=\frac{1}{BK}\sum_{i,h}(b_{ih}+p_i r_{ih}-y_{ih})^2,
\]
\[
L_b=\frac{1}{BK}\sum_{i,h}\big(b_{ih}-\min(y_{ih},\tau)\big)^2,
\]
\[
L_r=\frac{1}{BK}\sum_iE_i\sum_h\big(r_{ih}-(y_{ih}-\tau)_+\big)^2,
\]
\[
L_g=-\frac1B\sum_i\big[E_i\log p_i+(1-E_i)\log(1-p_i)\big],
\]
\[
L_{\mathrm{dual}}=\lambda_bL_b+\lambda_rL_r+\lambda_gs_y^2L_g,
\qquad L=L_{\mathrm{pred}}+L_{\mathrm{dual}}.
\]

三项辅助监督统一命名为 **dual loss**。正式模型使用 `dual_ablation=none`，必须包含完整的 dual loss，独立于所选 `loss_mode`。
`DUAL_LOSS=1`（Python：`--dual_loss 1`）为默认；若设为0，自动恢复为1并在训练日志输出带时间戳的 warning。
三个辅助项的权重若为0，也分别恢复为默认1并写入同一条 warning；正权重保持配置值。
修正后的有效值写入运行配置 JSON 和 checkpoint。Baseline／single head 不使用 dual loss；原来的 tail/slope 配置仍独立控制最终预测项。
Tail loss 保持 `max(y) >= tail_threshold`；dual 的事件定义保持严格 `max(y) > tau_phys`，两者各自拟合阈值。

机制消融必须显式选择 `--dual_ablation`（shell：`DUAL_ABLATION`）：

| 模式 | 直接分支监督 | 事件 gate |
|---|---|---|
| `none` | L_b、L_r、L_g | 学习 |
| `no_gate_bce` | L_b、L_r | 仍通过 L_pred 学习 |
| `no_excess_loss` | L_b、L_g | 学习 |
| `no_branch_supervision` | 无，仅 L_pred | 学习 |
| `fixed_gate` | L_b、L_r | `p(X) ≡ q_E` |

消融只将指定项的有效权重置0；仍启用的项不能通过权重0意外关闭。`no_branch_supervision` 将 `dual_loss` 置0。
新增可选的[物理 excess 峰值幅度监督](EXCESS_AMPLITUDE.md)默认权重为0，独立于上述三项，且不会被自动恢复为1。
`no_excess_loss` 和 `no_branch_supervision` 同时关闭轨迹 excess 与峰值幅度监督；`no_gate_bce`、`fixed_gate` 可保留正的幅度权重。
可选的 [severity-shape 分解](SEVERITY_SHAPE.md)用物理幅度 severity 与窗口内最大值为1的 shape 重建 excess，逐 horizon 除以 TRAIN `y_std` 后仍写入 `output.excess`。默认 `EXCESS_FORMULATION=direct` 完整保留现有头和旧 checkpoint 加载；新增 `SHAPE_LOSS_WEIGHT=0` 独立控制无量纲 shape 监督，两个 excess 消融也将其关闭。幅度监督继续复用 #2 的权重和目标。
固定 gate 不建立可学习 gate MLP，原始 q_E 作为 buffer 保存，允许精确取0或1；没有 BCE 项。
其参数预算少一个 gate MLP，应在结果中注明。其他消融保留相同 decoder 参数量和最终预测 loss。
固定 gate 输出 `gate_logits=None`、`gate_probability=q_E`，其余模式返回有限 logits 和 sigmoid 概率。

没有事件的 batch 令 L_r=0。L_r 与其他项统一按整个 batch 平均；非事件窗口被 mask 掉，其 excess 梯度仍为零。
早期原型除以事件样本数，会将这项相对整体样本风险放大约 1/事件频率，稀有事件时还随 batch 的事件数量波动。
当前 L_r 在总体上等于 P(E=1) × 条件 excess 风险；这个正比例系数不改变独立分支的条件均值最优解，
但会改变有限容量共享 backbone 的联合优化权衡。因此不把这次归一化改动说成无影响的代数重写。
当前实现固定采用上述整个 batch 平均，没有 reduction 模式开关。
实现使用 logits 版 BCE，避免直接 log(sigmoid) 的数值问题。
s_y²=mean_h(TRAIN y_std_h²)，用于把无量纲 BCE 与物理平方误差的量级对齐。
λ_b、λ_r、λ_g 默认均为1，可以配置其他正值；这不是已经通过 ablation 确定的最优权重。
默认没有正类加权 BCE；如果引入类权重，不能继续直接把未经校准的 sigmoid 当作事件概率。

最终预测 loss 让三个分支协作；另外三个监督目标为分支规定含义。
“Excess 只监督事件窗口”指 L_r 的直接分支监督；L_pred 仍在全部样本上通过三个分支反传。

这还改变了训练信号：若只有最终 MSE，传给 r 的梯度包含乘子 p；早期 gate 很小，
excess 分支就很难学起来。加入 L_r 后，事件窗口的分支梯度直接包含 (r-D)，不再先乘 p。
Gate 也获得 BCE 的直接梯度 p-E，而不必只依赖尾部回归是否已经有用。
Body 的独立标签与上界约束则阻止它独自承担全部超阈高度、让另一头闲置。

tail/slope loss 仍可以加在最终预测上，但应作为单独 ablation，因为它们会改变优化目标。
在无限容量、总体 MSE/BCE 的理想条件下，上述条件均值/概率共同满足各监督目标；
有限样本、有限容量、共享 backbone 与联合训练并不保证这种语义和概率校准精确成立。

## 5. 归一化输出怎样对应物理量

代码沿用 y'_h=(y_h-μ_h)/σ_h，μ、σ 都来自 TRAIN。保存逐输出阈值
τ'_h=(τ-μ_h)/σ_h，并约束：

\[
b'_h=\tau'_h-\operatorname{softplus}(\tau'_h-a_h),\qquad
r'_h=\operatorname{softplus}(d_h),\qquad p=\operatorname{sigmoid}(g).
\]

所以 b'_h≤τ'_h、r'_h≥0。物理 body 为 μ_h+σ_hb'_h，物理 excess 为 σ_hr'_h，
最终 y_hat=μ_h+σ_h(b'_h+p r'_h)。增量不加 μ。
softplus 使输出约束光滑，有限参数下只可逼近零增量或精确上界。

旧结构中的额外可学习 α 被取消；否则 α、gate 和 excess 的尺度容易互相补偿，
削弱“gate 是事件概率、excess 是多少米”的解释。
TRAIN 阈值随 checkpoint 保存；推理只需要 X，完全不需要未来真值或事件标签。

## 6. 与旧 dual head 的实质差别

旧公式 base + sigmoid(α)×sigmoid(gate)×tail 只有最终预测监督。
两个回归 head 可以互相接管任务，constant gate 也不违反任何训练目标。
给最终预测添加 tail/slope loss 并不会自动让 tail 分支学极端事件、gate 学事件概率。

新方法的分工来自显式输出分解、符号/上界约束和分支监督共同作用。
因此论文应称为一个有监督的事件/增量 decoder，而不能把整个收益仅归给“多一个 head”。
它仍可能不如 single：事件样本稀少会增加方差，辅助目标可能争夺 backbone 容量，
条件均值会对不确定峰值作概率折中。

需要固定 backbone/H/预算，以 single+MSE、single+tail/slope、新 dual+MSE、
新 dual+tail/slope 为主要对照。再使用上述显式机制消融检查 gate BCE、excess监督、整体分支监督和输入相关 gate 的贡献。
同时报告总体/peak RMSE、事件 PR-AUC、Brier/可靠性图，以及固定 gate 对照。
单独移除共同训练模型的一个 head 是诊断，不能替代独立训练 single 的性能比较。

极端事件的 hurdle/mixture 思想已有先例，例如
[Wang & Gao, 2023](https://arxiv.org/abs/2310.07435)。
这个先例不等于上述 decoder 的相同实现，但意味着不能把“事件概率×条件增量”本身当作首次提出的概念。
在本研究中可争取的贡献是具体的时空表示、六步事件轨迹分解和经实验证明的有效性。

## 7. 与新的 backbone 残差系数区分

本轮 backbone 中的 `gamma_l = exp(log_gain_l)` 是每层的残差尺度，初始化为 0.1 后参与学习。
它不依赖当前窗口的输入，不表示事件概率；dual head 的 `p(X)` 才是逐样本的监督事件 gate。
因此，新 backbone 的 gamma 与本说明取消的旧 dual-head 整体系数 alpha 是不同位置的参数。
单头和新双头共用同一套 backbone，比较时必须采用相同的稳定结构、确定性设置和训练预算。

实现位置：[`heads.py`](../emulator/models/heads.py) 负责前向，[`losses.py`](../emulator/training/losses.py) 负责监督，
[`stats.py`](../emulator/data/stats.py) 拟合 TRAIN 阈值和事件比例。

推理增加 `--dual_diagnostics`（两个 shell launcher 使用 `DUAL_DIAGNOSTICS=1`）时，输出：

- `dual_diagnostics.npz`：p_i、物理 body/excess、p×excess、严格事件标签 E_i、真值/预测/tags、TRAIN τ/q_E 和模式。
- `dual_diagnostics.json`：总体和逐年 Brier、stepwise average precision、trapezoid PR-AUC、10个等宽可靠性分箱、事件/非事件 gate 直方图、事件窗口RMSE和分支误差。

PR 面积的两种定义分别命名；没有正事件时 PR 指标为 null，空可靠性分箱也为 null。
事件阈值始终读取 TRAIN metadata，不在 test 或 external 数据上重新拟合。
原始概率和分支轨迹支持后续可靠性图、贡献分析和误差检查；当前不自动生成图片。
诊断直接复用推理的那次 forward。训练/验证 epoch 不收集上述数组，日志继续只记录原来的四项物理误差。
