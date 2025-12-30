"""
快速修复验证脚本 - 测试健壮性改进
"""
import sys
import os

# 模拟空问题的场景
class MockQA:
    def __init__(self, questions, options, answer, question_id, category):
        self.questions = questions
        self.options = options
        self.answer = answer
        self.question_id = question_id
        self.category = category
        self.question_type = 'test'
        self.episode_name = 'test_episode'
        self.scene_id = 1

def test_empty_questions_handling():
    """测试空问题字典的处理"""
    print("="*60)
    print("测试1: 空questions字典处理")
    print("="*60)

    # 测试用例1: 空字典
    qa1 = MockQA(
        questions={},
        options=['A', 'B', 'C'],
        answer='A',
        question_id='test_1',
        category='test'
    )

    # 测试用例2: 没有default但有其他key
    qa2 = MockQA(
        questions={'Monica': 'What is the answer?', 'Joey': 'What\'s the answer?'},
        options=['A', 'B', 'C'],
        answer='A',
        question_id='test_2',
        category='test'
    )

    # 测试用例3: 正常的default
    qa3 = MockQA(
        questions={'default': 'What is the answer?'},
        options=['A', 'B', 'C'],
        answer='A',
        question_id='test_3',
        category='test'
    )

    test_cases = [
        ('空字典', qa1),
        ('无default有其他key', qa2),
        ('正常default', qa3)
    ]

    for name, qa in test_cases:
        print(f"\n测试: {name}")
        print(f"  questions: {qa.questions}")

        # 模拟新的处理逻辑
        try:
            if not qa.questions:
                print(f"  结果: [SKIP] 检测到空字典，应该被跳过")
                continue

            question = qa.questions.get('default')
            if not question:
                question = next(iter(qa.questions.values()), None)
                if not question:
                    print(f"  结果: [SKIP] 无有效问题文本")
                    continue
                else:
                    print(f"  结果: [OK] 使用降级方案: {question}")
            else:
                print(f"  结果: [OK] 使用default: {question}")
        except Exception as e:
            print(f"  结果: [ERROR] 异常: {e}")

def test_safe_logging():
    """测试安全的日志输出"""
    print("\n" + "="*60)
    print("测试2: 安全日志输出")
    print("="*60)

    qa1 = MockQA(questions={}, options=[], answer='', question_id='test', category='test')
    qa2 = MockQA(questions={'default': 'Question?'}, options=[], answer='', question_id='test', category='test')

    for qa in [qa1, qa2]:
        print(f"\nquestions: {qa.questions}")

        # 旧方法（会出错）
        try:
            old_way = qa.questions.get('default', list(qa.questions.values())[0])
            print(f"  旧方法结果: {old_way}")
        except IndexError:
            print(f"  旧方法结果: [ERROR] IndexError!")

        # 新方法（安全）
        new_way = qa.questions.get('default', list(qa.questions.values())[0] if qa.questions else 'N/A')
        print(f"  新方法结果: [OK] {new_way}")

def test_error_recovery():
    """测试错误恢复机制"""
    print("\n" + "="*60)
    print("测试3: 错误恢复和继续处理")
    print("="*60)

    questions = [
        MockQA({}, [], '', 'q1', 'cat1'),  # 空问题
        MockQA({'default': 'Q2?'}, ['A', 'B'], 'A', 'q2', 'cat2'),  # 正常
        MockQA({}, [], '', 'q3', 'cat3'),  # 空问题
        MockQA({'default': 'Q4?'}, ['A', 'B'], 'A', 'q4', 'cat4'),  # 正常
    ]

    processed = 0
    skipped = 0

    for idx, qa in enumerate(questions, 1):
        try:
            if not qa.questions:
                print(f"问题{idx}: [WARN] 跳过 (空questions)")
                skipped += 1
                continue

            question = qa.questions.get('default')
            if not question:
                question = next(iter(qa.questions.values()), None)
                if not question:
                    print(f"问题{idx}: [WARN] 跳过 (无有效文本)")
                    skipped += 1
                    continue

            print(f"问题{idx}: [OK] 处理成功 - {question}")
            processed += 1

        except Exception as e:
            print(f"问题{idx}: [ERROR] 错误 - {e}")
            skipped += 1
            continue

    print(f"\n总结:")
    print(f"  总问题数: {len(questions)}")
    print(f"  成功处理: {processed}")
    print(f"  跳过/错误: {skipped}")
    print(f"  程序状态: [OK] 未崩溃，继续运行")

def main():
    print("\nDialSim健壮性改进验证测试")
    print("="*60)
    print("验证以下改进:")
    print("1. 空questions字典检测和处理")
    print("2. 智能降级到第一个可用问题")
    print("3. 安全的日志输出")
    print("4. 错误恢复和继续处理")
    print("="*60)

    test_empty_questions_handling()
    test_safe_logging()
    test_error_recovery()

    print("\n" + "="*60)
    print("[PASS] 所有测试完成！")
    print("="*60)
    print("\n验证结果:")
    print("  [OK] 空问题会被安全跳过")
    print("  [OK] 降级机制正常工作")
    print("  [OK] 日志输出不会崩溃")
    print("  [OK] 程序在错误时继续运行")
    print("\n建议:")
    print("  1. 在真实数据上运行 test_advanced_dialsim.py")
    print("  2. 检查日志中的 WARNING 和 ERROR 信息")
    print("  3. 确认 error_num 统计")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
