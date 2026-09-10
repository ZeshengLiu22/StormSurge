**验证证据的范围**

`publication_tests.txt` 是在独立 StormSurge 目录运行当前42项测试的完整输出。

其他文件来自迁移前、同一套相关生产代码的近期对照：

- `model_equivalence.json`、`gpu_equivalence.json`：原版模型及已确认stability数学的对应验证。新dual的清理对照对象是清理前的新dual，不是旧residual dual。
- `scheduler_comparison.json`：仅提取当时的6组scheduler曲线对照，未将当时尚未修复的其他问题混入当前验证状态。
- `protocol_*ranks.json`：恢复原训练协议后，与原版源码对照的统计、augmentation RNG、小scale loss/梯度和Val指标结果。
- `smoke_ddp.txt`、`smoke_gpu.txt`及observed_rank文件：当时真实入口的两轮小型训练和rank/seed观测。
- `integrity.json`：前一轮协议恢复相对其开始快照的文件变动记录，不是本次发布相对原版的完整差异。当前完整差异在上一级source_inventory文件中。

原始日志/JSON保留当时的本地路径与数值。测试使用已安装的PyTorch2.6+cu124；恢复的环境配方是原2.8+cu128文件，本次没有重新安装该环境或运行完整训练扫描。

源码迁移保持Python/shell/config原字节，原始日志及unified diff也保留原有空格。因此对“全新初始提交”运行全量`git diff --check`会把部分原始空白一并列出；这不代表迁移改写了这些源码。新编写的报告、比较脚本、JSON/CSV另做了格式检查。
