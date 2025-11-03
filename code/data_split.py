"""
数据划分脚本 - 用于判决预测和摘要提取任务
根据unsloth训练框架的要求，将数据划分为训练集和测试集
"""

import jsonlines
import random
from pathlib import Path

# 设置随机种子以保证可复现性
random.seed(42)

# 数据划分比例
TRAIN_RATIO = 0.9  # 90%训练集，10%测试集


def split_dataset(input_file, output_dir, task_name):
    """
    划分数据集为训练集和测试集

    Args:
        input_file: 输入的jsonl文件路径
        output_dir: 输出目录
        task_name: 任务名称（用于输出目录命名）
    """
    print(f"\n处理任务: {task_name}")
    print(f"读取文件: {input_file}")

    # 读取所有数据
    dataset = []
    with jsonlines.open(input_file) as reader:
        for each_line in reader:
            dataset.append(each_line)

    total_count = len(dataset)
    print(f"总数据量: {total_count}")

    # 打乱数据
    random.shuffle(dataset)

    # 计算划分点
    train_count = int(total_count * TRAIN_RATIO)

    # 划分数据
    train_data = dataset[:train_count]
    test_data = dataset[train_count:]

    print(f"训练集数量: {len(train_data)}")
    print(f"测试集数量: {len(test_data)}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 写入训练集
    train_file = output_path / "train.jsonl"
    with jsonlines.open(train_file, 'w') as writer:
        for item in train_data:
            writer.write(item)
    print(f"训练集已保存到: {train_file}")

    # 写入测试集
    test_file = output_path / "test.jsonl"
    with jsonlines.open(test_file, 'w') as writer:
        for item in test_data:
            writer.write(item)
    print(f"测试集已保存到: {test_file}")

    # 显示数据样例
    print(f"\n{task_name} 数据样例:")
    print("训练集第一条:")
    print(train_data[0])
    print("\n测试集第一条:")
    print(test_data[0])


def main():
    """主函数"""
    print("=" * 80)
    print("开始数据划分")
    print("=" * 80)

    # 1. 处理判决预测任务
    split_dataset(
        input_file="./判决预测/DISC-Law-SFT-Triplet-released.jsonl",
        output_dir="./data_judgement",
        task_name="判决预测"
    )

    # 2. 处理摘要提取任务
    split_dataset(
        input_file="./摘要提取/DISC-Law-SFT-Pair.jsonl",
        output_dir="./data_sum",
        task_name="摘要提取"
    )

    print("\n" + "=" * 80)
    print("数据划分完成！")
    print("=" * 80)
    print("\n目录结构:")
    print("  ./data_judgement/")
    print("    - train.jsonl  (判决预测训练集)")
    print("    - test.jsonl   (判决预测测试集)")
    print("  ./data_sum/")
    print("    - train.jsonl  (摘要提取训练集)")
    print("    - test.jsonl   (摘要提取测试集)")


if __name__ == "__main__":
    main()
