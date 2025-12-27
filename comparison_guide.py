"""
对比LoComo和DialSim数据集格式及使用方法
"""

# ==========================================
# 数据集格式对比
# ==========================================

# LoComo 数据格式 (locomo10.json)
locomo_format = {
    "qa": [
        {
            "question": "When did Caroline go to the LGBTQ support group?",
            "answer": "7 May 2023",
            "evidence": ["D1:3"],
            "category": 2  # 类别1-5
        }
    ],
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_1": [  # 会话列表
            {
                "speaker": "Caroline",
                "dia_id": "D1:1",
                "text": "Hey! How are you doing?"
            }
        ],
        "session_1_date_time": "7 May 2023"
    }
}

# DialSim 数据格式 (friends_dialsim.pickle)
dialsim_format = {
    "S01E01 Monica Gets A Roommate.txt": {  # 剧集名
        1: {  # 场景ID
            "date": "September 22, 1994",
            "script": "Monica: There's nothing to tell!\nJoey: C'mon!",  # 剧本格式
            "hard_q": {
                "fu": [  # 问题类别
                    {
                        "questions": {  # 多个speaker版本的问题
                            "default": "So, who's Ross's mom?",
                            "Monica": "Do you know who Ross's mother is?",
                            "Joey": "Hey, who's Ross's mom?"
                        },
                        "options": [  # 多选项
                            "Mrs. greene",
                            "Mrs. geller",
                            "I don't know."
                        ],
                        "answer": "I don't know."
                    }
                ]
            },
            "easy_q": {
                "ans_w_time": {
                    350: {  # 问题ID
                        "questions": {...},
                        "options": [...],
                        "answer": "..."
                    }
                }
            }
        }
    }
}

# ==========================================
# 使用方法对比
# ==========================================

print("="*60)
print("LoComo 数据集使用方法")
print("="*60)

# 基本运行
locomo_basic = """
python test_advanced.py \\
    --dataset data/locomo10.json \\
    --model gpt-4o-mini \\
    --backend openai \\
    --ratio 1.0 \\
    --retrieve_k 10
"""
print("\n1. 基本运行:")
print(locomo_basic)

# 带参数运行
locomo_advanced = """
python test_advanced.py \\
    --dataset data/locomo10.json \\
    --model gpt-4o-mini \\
    --backend openai \\
    --ratio 0.5 \\
    --retrieve_k 10 \\
    --temperature_c5 0.5 \\
    --output results/locomo_results.json
"""
print("\n2. 带参数运行:")
print(locomo_advanced)

print("\n" + "="*60)
print("DialSim 数据集使用方法")
print("="*60)

# 基本运行
dialsim_basic = """
python test_advanced_dialsim.py \\
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \\
    --model gpt-4o-mini \\
    --backend openai \\
    --max_episodes 3 \\
    --retrieve_k 10
"""
print("\n1. 基本运行 (前3集):")
print(dialsim_basic)

# 限制问题类别
dialsim_categories = """
python test_advanced_dialsim.py \\
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \\
    --model gpt-4o-mini \\
    --backend openai \\
    --max_episodes 5 \\
    --question_categories ans_w_time ans_wo_time past cur fu \\
    --max_questions_per_episode 100 \\
    --retrieve_k 10
"""
print("\n2. 限制问题类别:")
print(dialsim_categories)

# 完整评估
dialsim_full = """
python test_advanced_dialsim.py \\
    --dataset data/dialsim_v1.1/friends_dialsim.pickle \\
    --model gpt-4o-mini \\
    --backend openai \\
    --retrieve_k 10 \\
    --output results/dialsim_full_results.json
"""
print("\n3. 完整评估 (所有118集):")
print(dialsim_full)

# ==========================================
# 参数对比
# ==========================================

print("\n" + "="*60)
print("参数对比")
print("="*60)

comparison = """
LoComo (test_advanced.py)        | DialSim (test_advanced_dialsim.py)
--------------------------------|-----------------------------------
--ratio 0.5                     | --max_episodes 10
  (使用50%的样本)                |   (使用前10集)
                                |
(无限制)                         | --max_questions_per_episode 100
                                |   (每集最多100个问题)
                                |
(无筛选)                         | --question_categories past cur fu
                                |   (只评估特定类别的问题)
                                |
--temperature_c5 0.5            | --temperature 0.7
  (针对category 5的温度)         |   (全局温度)
                                |
category: 1, 2, 3, 4, 5        | category: past, cur, fu, ans_w_time等
  (5种问题类型)                  |   (17种问题类型)

共同参数:
--dataset, --model, --backend, --retrieve_k, --output
--sglang_host, --sglang_port
"""
print(comparison)

# ==========================================
# 问题类别对比
# ==========================================

print("\n" + "="*60)
print("问题类别对比")
print("="*60)

categories_comparison = """
LoComo Categories:
  Category 1: 事实性问题 (Factual)
  Category 2: 时间相关问题 (Temporal)
  Category 3: 推理问题 (Reasoning)
  Category 4: 总结性问题 (Summary)
  Category 5: 对抗性问题 (Adversarial)

DialSim Categories:
  Hard Questions (时间推理):
    - past: 过去事件
    - cur: 当前事件
    - fu: 未来事件
    - past_past, past_cur, past_fu: 过去相关的多重推理
    - cur_past, cur_cur, cur_fu: 当前相关的多重推理
    - fu_past, fu_cur, fu_fu: 未来相关的多重推理

  Easy Questions (可回答性):
    - ans_w_time: 带时间的可回答问题
    - ans_wo_time: 不带时间的可回答问题
    - before_event_unans: 事件前不可回答
    - dont_know_unans: 不可回答
    - dont_know_unans_time: 带时间的不可回答
"""
print(categories_comparison)

# ==========================================
# 数据量对比
# ==========================================

print("\n" + "="*60)
print("数据量对比")
print("="*60)

data_stats = """
LoComo:
  - 样本数: 10
  - 平均每个样本的QA: ~20-30个
  - 总QA数: ~200-300个
  - 会话轮次: 每个样本多个会话，每个会话多个轮次

DialSim (前3集):
  - 剧集数: 3
  - 平均每集QA: ~4,944个
  - 总QA数: ~14,833个
  - Hard questions: 14,317个
  - Easy questions: 516个

DialSim (完整):
  - 剧集数: 118
  - 估计总QA: ~580,000个
  - 场景总数: ~940个
"""
print(data_stats)

# ==========================================
# 推荐使用策略
# ==========================================

print("\n" + "="*60)
print("推荐使用策略")
print("="*60)

recommendations = """
1. LoComo数据集 - 适合:
   - 快速原型验证
   - 多轮对话理解
   - 小规模实验
   - 推荐: 使用 --ratio 1.0 评估所有样本

2. DialSim数据集 - 适合:
   - 大规模评估
   - 时间推理能力测试
   - 多选题QA系统

   推荐配置:

   a) 快速测试 (5-10分钟):
      --max_episodes 1
      --max_questions_per_episode 50
      --question_categories ans_w_time past fu

   b) 中等测试 (1-2小时):
      --max_episodes 10
      --max_questions_per_episode 100
      --question_categories ans_w_time ans_wo_time past cur fu

   c) 完整评估 (数小时到数天):
      --max_episodes 118
      (不限制问题数)
      (包含所有类别)
"""
print(recommendations)

print("\n" + "="*60)
print("完成!")
print("="*60)
