**验证证据的范围**

最新mag/strip恢复（2026-09-10）见 `mag_strip_validation.json` 和 `mag_strip_tests.txt`：**全量72项测试通过**。mag只校验实际使用的0<p_hi<=100；robust保持上下分位检查；训练/推理head和temporal名称补回首尾strip。18组修改前后单进程CPU训练及6组新增合法mag设置对照均逐位一致，覆盖三种归一化、baseline/PACT single/dual、累积1/2步以及dropout/augmentation。所有config、模型、loss、train.py及engine保持，阈值函数逐字未变。另修正一个随机临时目录末尾下划线触发的旧测试误报，生产目录命名不变。本次未测GPU速度或GPU逐位结果。

此前第四部分复核见 `runtime_review.json` 和 `runtime_review_tests.txt`：**全量70项测试通过**。删除stable_arch参数、透传/快照及三处配置声明，dual_mode保留；训练结束从最佳checkpoint取完整Val，与一次Test结果共同打印并保存summary。四组训练/推理往返确认Val对应选模epoch且没有多一次forward；配置扫描及现有tail/CPU-Gloo回归继续通过。三份配置仅删STABLE_ARCH=1，其他130份配置逐字保持；模型、loss、阈值、runtime、engine、归一化与指标代码未改。该轮TF32/线程/阈值/mag/strip提问只更新解释和建议；当时GPU驱动不可用，未测GPU速度。

此前sample-additive tail更新见 `sample_additive_tail.json` 和 `sample_additive_tail_tests.txt`：**全量70项测试通过**，包括新增7项tail测试。六种tail模式验证分批loss/梯度、固定样本梯度、空事件、加权语义、CLI/保存阈值、原有累积及两进程CPU/Gloo的平均梯度和DDP＋累积。该轮唯一运行代码变化为LossConfig.tail_frac及tail归约分母；配置、模型、阈值计算、slope/dual项、engine及指标逐项核对未变。没有旧条件归约的运行选项。使用CPU、PyTorch2.6+cu124、PyG2.7；未验证新公式的GPU/BF16/NCCL执行或长程精度。

此前dual接口复核结果在 `dual_review_tests.txt`：**50项测试通过**。`dual_review_validation.json` 保存当时覆盖范围、运行环境和 Python/shell 源码哈希。验证包括真实 TRAIN 事件比例、阈值解耦、五种 dual 模式的训练/保存/推理、两个 shell 推理入口和独立诊断输出。测试使用 CPU、PyTorch2.6+cu124、PyG2.7；当时 GPU 驱动不可用，未重跑 GPU/BF16/NCCL，也未做长程精度比较。

随后CNN网格核对见 `cnn_grid_equivalence.json`、`cnn_grid_tests.txt` 和 `cnn_model_tests.txt`：72组原版CNN直接对照＋72组当前完整模型仅替换网格检查的隔离对照，输出、梯度、一步Adam更新及RNG逐位一致；另14项模型/训练推理测试通过。只恢复CNN的原地LeakyReLU写法，消除1×1双层eval案例中约2e-10的参数梯度舍入差，保留现有网格检查。覆盖正常PyG Batch的合法正整数矩形网格；未验证手工破坏的batch/ptr一致性。均为本次CPU FP32结果，与历史GPU记录区分。可用 `python docs/audit/verify_cnn_grid.py --original /path/to/Emulator` 复现数值对照。

`publication_tests.txt` 是在独立 StormSurge 目录发布首次源码快照时运行42项测试的完整输出。

以下文件来自迁移前的历史对照，不能作为新 event-prior/ablation 接口的本轮验证：

- `model_equivalence.json`、`gpu_equivalence.json`：原版模型及已确认stability数学的对应验证。新dual的清理对照对象是清理前的新dual，不是旧residual dual。
- `scheduler_comparison.json`：仅提取当时的6组scheduler曲线对照，未将当时尚未修复的其他问题混入当前验证状态。
- `protocol_*ranks.json`：恢复原训练协议后，与原版源码对照的统计、augmentation RNG、小scale loss/梯度和Val指标结果。其中tail loss使用当时的条件归约；本次有意改变tail目标，因此这些记录不证明当前tail与原版相等。
- `smoke_ddp.txt`、`smoke_gpu.txt`及observed_rank文件：当时真实入口的两轮小型训练和rank/seed观测。
- `integrity.json`：前一轮协议恢复相对其开始快照的文件变动记录，不是本次发布相对原版的完整差异。当前完整差异在上一级source_inventory文件中。

原始日志/JSON保留当时的本地路径与数值。历史测试使用已安装的PyTorch2.6+cu124；仓库环境配方是原2.8+cu128文件，本轮没有重新安装该环境或运行完整训练扫描。上一级 source_inventory、argparse_comparison 与 original_to_current.diff 则按当前工作树重新生成。

源码迁移保持Python/shell/config原字节，原始日志及unified diff也保留原有空格。因此对“全新初始提交”运行全量`git diff --check`会把部分原始空白一并列出；这不代表迁移改写了这些源码。新编写的报告、比较脚本、JSON/CSV另做了格式检查。
