"""
Memory management patch for test_advanced.py

添加到 test_advanced.py 的第 363 行之后（results.append(result) 之后）
在每个 sample 的 QA 循环结束后添加内存清理
"""

import gc
import torch

# 在 for sample_idx, sample in enumerate(samples): 循环的末尾
# （即处理完该 sample 的所有 QA 之后）添加以下代码：

# ==================== 添加位置示例 ====================
#
# for sample_idx, sample in enumerate(samples):
#     agent = advancedMemAgent(...)
#
#     # ... 处理 memories ...
#
#     for qa in sample.qa:
#         # ... 处理 QA ...
#         results.append(result)
#
#     # ====== 在这里添加以下内存清理代码 ======
#
#     # 清理 agent 对象
#     del agent
#
#     # 强制垃圾回收
#     gc.collect()
#
#     # 如果使用了 PyTorch，清理 CUDA 缓存
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#
#     # 每 10 个 sample 输出内存使用情况
#     if sample_idx % 10 == 0:
#         import psutil
#         import os
#         process = psutil.Process(os.getpid())
#         mem_info = process.memory_info()
#         logger.info(f"Sample {sample_idx}: Memory usage = {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
#
# ====================================================

# 具体修改位置：test_advanced.py 第 363 行之后，在 for sample_idx 循环结束前
