# DialSim数据集适配说明

## 概述
本次更新为A-mem代码库添加了对DialSim数据集（Friends对话数据集）的支持。现在你可以使用与LoComo数据集相同的评估框架来测试DialSim数据集。

## 新增文件

### 1. load_dialsim_dataset.py
**功能**: DialSim数据集加载器
- 解析`friends_dialsim.pickle`格式的数据
- 支持筛选特定问题类别
- 支持限制加载的剧集数量
- 提供数据集统计信息

**主要类**:
- `DialSimQA`: 问答对数据结构
- `DialSimScene`: 场景数据结构
- `DialSimEpisode`: 剧集数据结构
- `DialSimSample`: 样本数据结构（一个完整剧集）

**主要函数**:
- `load_dialsim_dataset()`: 加载数据集
- `get_dialsim_statistics()`: 获取统计信息

### 2. test_advanced_dialsim.py
**功能**: DialSim数据集评估主脚本
- 改编自原始的`test_advanced.py`
- 支持多选题格式的问答
- 自动缓存记忆以加速重复评估
- 生成详细的评估日志和结果

**主要功能**:
- 对每个剧集构建记忆系统
- 使用LLM回答多选题
- 计算各种评估指标（F1, ROUGE, BLEU等）
- 支持按类别筛选问题
- 支持限制问题数量

### 3. README_DIALSIM.md
**功能**: 详细的使用文档
- 数据集结构说明
- 使用示例
- 命令行参数说明
- 问题类别列表
- 与LoComo数据集的差异说明

### 4. explore_dialsim.py
**功能**: 数据集探索工具
- 用于理解和验证数据集结构
- 显示样本问题和答案
- 统计信息输出

## 数据集信息

### 数据集规模
- **总剧集数**: 118集Friends剧集
- **每集场景数**: 平均8-10个场景
- **总问题数**: 约150,000+个问题（全数据集）
- **前3集统计**:
  - Hard questions: 14,317个
  - Easy questions: 516个

### 问题类型

#### Hard Questions (困难问题)
时间推理类问题，共12个子类别:
- `past`: 过去事件
- `cur`: 当前事件
- `fu`: 未来事件
- `past_past`, `past_cur`, `past_fu`: 过去相关的多重推理
- `cur_past`, `cur_cur`, `cur_fu`: 当前相关的多重推理
- `fu_past`, `fu_cur`, `fu_fu`: 未来相关的多重推理

#### Easy Questions (简单问题)
可回答性问题，共5个子类别:
- `ans_w_time`: 带时间上下文的可回答问题
- `ans_wo_time`: 不带时间上下文的可回答问题
- `before_event_unans`: 事件发生前的不可回答问题
- `dont_know_unans`: 不可回答问题
- `dont_know_unans_time`: 带时间上下文的不可回答问题

## 快速开始

### 1. 基本测试（前3集，所有问题）
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model gpt-4o-mini \
    --backend openai \
    --max_episodes 3 \
    --retrieve_k 10
```

### 2. 测试特定类别（推荐用于快速测试）
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model gpt-4o-mini \
    --backend openai \
    --max_episodes 5 \
    --question_categories ans_w_time ans_wo_time past cur fu \
    --max_questions_per_episode 100 \
    --retrieve_k 10
```

### 3. 完整评估（警告：耗时很长）
```bash
python test_advanced_dialsim.py \
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \
    --model gpt-4o-mini \
    --backend openai \
    --retrieve_k 10 \
    --output results/dialsim_full_results.json
```

## 输出说明

### 日志文件
保存在`logs/`目录，命名格式:
```
eval_dialsim_{model}_{backend}_ep{max_episodes}_cat{categories}_{timestamp}.log
```

### 结果文件
JSON格式，包含:
- 模型和数据集信息
- 总问题数和剧集数
- 类别分布
- 聚合指标
- 每个问题的详细结果

### 缓存文件
保存在`cached_memories_dialsim_{backend}_{model}/`目录:
- `memory_cache_{episode_name}.pkl`: 记忆缓存
- `retriever_cache_{episode_name}.pkl`: 检索器缓存
- `retriever_cache_embeddings_{episode_name}.npy`: 嵌入缓存

删除缓存目录可以强制重新生成记忆。

## 与原LoComo评估的主要差异

1. **问题格式**: DialSim使用多选题，LoComo使用开放式问答
2. **问题类别系统**: DialSim使用past/cur/fu系统，LoComo使用1-5类别
3. **对话结构**: DialSim使用剧本格式，LoComo使用对话会话格式
4. **组织方式**: DialSim按剧集-场景组织，LoComo按样本-会话组织

## 常见问题

### Q: 为什么问题数量这么多？
A: DialSim数据集为每个场景生成了大量的多选题，用于测试不同时间推理能力。建议使用`--max_questions_per_episode`和`--question_categories`来限制评估范围。

### Q: 如何选择合适的问题类别？
A:
- 快速测试: `ans_w_time ans_wo_time past cur fu`
- 时间推理: `past cur fu past_fu fu_past`
- 可回答性: `ans_w_time ans_wo_time dont_know_unans`

### Q: 评估太慢怎么办？
A:
1. 使用`--max_episodes 3`限制剧集数
2. 使用`--max_questions_per_episode 100`限制每集问题数
3. 使用`--question_categories`选择特定类别
4. 第二次运行会使用缓存，速度会快很多

### Q: 如何清除缓存？
A: 删除对应的缓存目录即可，例如:
```bash
rm -rf cached_memories_dialsim_openai_gpt-4o-mini/
```

## 下一步

1. 安装所需依赖（如果还没有）:
   ```bash
   pip install sentence-transformers nltk rouge-score bert-score
   ```

2. 运行探索脚本了解数据:
   ```bash
   python explore_dialsim.py
   ```

3. 运行小规模测试验证环境:
   ```bash
   python test_advanced_dialsim.py --max_episodes 1 --max_questions_per_episode 10
   ```

4. 根据需要调整参数进行完整评估

## 技术支持

如有问题，请查看:
- [README_DIALSIM.md](README_DIALSIM.md) - 详细文档
- `explore_dialsim.py` - 数据探索工具
- `logs/` - 评估日志
