# 当前 v2 的稳定结构与清理边界

v2 只实现当前结构。版本号分支、旧 bypass 原型、旧 residual dual head、重试/回滚 guard 和历史 checkpoint 兼容均已删除。冗余的 `--stable_arch` / `STABLE_ARCH` 已移除，当前稳定性结构始终启用；`--dual_mode exceedance` 继续保留。

## Backbone

1. GraphSAGE/CNN 空间编码后，由 station query 读取每个时刻的节点表示。
2. station attention 输出使用 FP32、无 affine 参数的 LayerNorm，再加相对 lag embedding。
3. MLP 和 Transformer 使用 pre-norm 残差。每条残差支路直接保存一个 `log_gain` 参数：

   `x_next = x + exp(log_gain) * branch(LayerNorm(x))`

   `log_gain` 初始化为 `log(0.1)`，与其他参数一起由 Adam 学习，不再暴露额外的 scale 调参项。正数参数化避免直接对 0.1 大小的线性系数做绝对步长更新。
4. LSTM/GRU 保留 `LayerNorm(memory + dropout(rnn(memory)))`。
5. horizon attention 输出在进入 head 前再做 FP32、无 affine 参数的 LayerNorm。
6. H=0 时 attention 只有一个 memory token，因此给 context 加回 horizon query，保留各预测时刻的身份。

没有为了诊断而强制输出 attention weights、计算 embedding 方差或读取 residual gains。模型、loss 和运行时指标分别放在独立模块。

## Dual head

`--head_type dual` 只表示当前的 body/exceedance head，完整公式见 [DUAL_HEAD_EXPLAINED.md](DUAL_HEAD_EXPLAINED.md)。三个辅助损失对应 body、事件窗口的 conditional excess 和 window-event gate。

三项统一称为 **dual loss**。正式模型使用默认 `--dual_ablation none`，强制启用完整监督，独立于 `loss_mode`。`--dual_loss 0` 或上述三项任一辅助权重为0时恢复对应默认值1，并输出带时间戳的 warning；已有正权重保留。显式消融可选 `no_gate_bce`、`no_excess_loss`、`no_branch_supervision`、`fixed_gate`，只关闭该模式指定的项。有效设置写入运行配置和 checkpoint。Single／baseline 不接受这些消融。

可选的[物理 excess 峰值幅度监督](EXCESS_AMPLITUDE.md)使用独立 `EXCESS_AMP_LOSS_WEIGHT`，默认0且不自动恢复为1。它按 horizon 转换为米后取峰值，用已拟合的精确 TRAIN `event_prior` 归一化，不改变上述轨迹 excess loss。`no_excess_loss`、`no_branch_supervision` 也关闭此新项；模型结构和推理不变。

[severity-shape 模式](SEVERITY_SHAPE.md)是另一项独立的可选架构：`EXCESS_FORMULATION=severity_shape`，默认仍为 `direct`。severity 以米预测，shape 按窗口最大值归一化，重建 excess 使用每个 horizon 的 TRAIN `y_std`。初始 severity 约0.1米、raw shape为1；两分支末层权重为0。复用幅度监督并新增默认0的 `SHAPE_LOSS_WEIGHT`；两个 excess 消融同时关闭 shape loss。direct 的模块、初始化和 state dict 不增加新 buffer。

`--exceedance_percentile 95` 从 TRAIN 窗口峰值拟合 dual 阈值，独立于 tail-loss 的 `--tail_frac`。Gate 初始化使用严格超阈的实际 TRAIN 比例 q_E，物理 τ、q_E、事件数和有效初始化概率都保存。可学习 logit 的初始化将 q_E 夹到[1e-6,1−1e-6]；固定 gate 使用原始 q_E，作为无可学习参数的 buffer。

Excess 辅助项先 mask 非事件窗口，再按整个 `B × K` 平均；不能恢复成除以事件数。后者会在稀有事件时放大辅助项相对于主任务的权重。这里的 mask 是监督定义，不是运行时 guard。

模型统一返回 `ForecastOutput`。Single 只填 prediction；dual 填 body、excess、threshold 和 gate_probability。可学习 gate 还返回 gate_logits；fixed_gate 的 logits 为 None，BCE 不参与训练。预测不读取真值。

## 运行设置

`--deterministic 0` 默认保持非严格确定性；`--deterministic 1` 统一启用 cuBLAS workspace 配置和 PyTorch/cuDNN 确定性。cuDNN benchmark 保持关闭。确定性控制复现条件，不代替模型结构中的尺度处理。

原配置默认启用 BF16 和 TF32；直接调用 Python 时使用原来的 `--amp`、`--tf32` 开关。没有自动重训、回滚、异常 epoch 跳过或独立残差 LR。`--max_grad_norm 0` 默认关闭 clipping；保留用户显式配置正值时的普通梯度裁剪。FP16 可选时使用 PyTorch 标准 GradScaler。

## 日志与验证解释

每轮只记录 Train/Val × All/Top5% × RMSE/MAE，全部为物理单位。每行保留 `[YYYY-MM-DD|HH:MM:SS]`；运行摘要保存训练循环时间、总 wall time 和最终 Test 指标，终端最后也输出 wall time。旧 guard、额外诊断 forward 和 attention/variance 统计仍已删除。仅显式 `infer.py --dual_diagnostics` 导出 gate/body/excess/事件数组和独立校准报告，不加入常规 epoch 日志。

清理后的验证区分两件事：相同权重下的计算等价，以及新代码能正确完成训练和推理。短程检查不能证明所有站点、H、seed 的长期训练都不会失稳；最终精度仍需从头训练后比较。精简初始化/模块结构后，同 seed 的整个随机数消耗顺序也不承诺与历史程序一致。
