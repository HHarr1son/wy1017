"""
统一测试脚本 - 支持判决预测、摘要提取和阅读理解三个任务
基于unsloth框架进行推理
"""

from unsloth import FastLanguageModel
import torch
import jsonlines
from tqdm import tqdm
import argparse
import os
import json

# 推理配置
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Alpaca prompt模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def load_test_data(data_path, task_type):
    """
    加载测试数据

    Args:
        data_path: 测试数据路径
        task_type: 任务类型

    Returns:
        测试数据列表
    """
    test_dataset = []

    with jsonlines.open(data_path) as reader:
        for each_line in reader:
            if task_type == "judgement":
                item = {
                    "id": each_line.get("id", ""),
                    "instruction": "根据案件事实和法律条文，预测判决结果",
                    "input": each_line["input"],
                    "output": each_line.get("output", "")
                }
            elif task_type == "summary":
                item = {
                    "id": each_line.get("id", ""),
                    "instruction": "请对以下法律文书进行摘要",
                    "input": each_line["input"],
                    "output": each_line.get("output", "")
                }
            elif task_type == "reading":
                item = {
                    "id": each_line.get("id", ""),
                    "instruction": each_line["instruction"],
                    "input": each_line["input"],
                    "output": each_line.get("output", "")
                }
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            test_dataset.append(item)

    return test_dataset


def test(task_type, model_path, output_file, max_new_tokens=256, sample_size=None):
    """
    测试模型

    Args:
        task_type: 任务类型
        model_path: 训练好的模型路径
        output_file: 预测结果输出文件
        max_new_tokens: 生成的最大token数
        sample_size: 采样数量（None表示全部测试）
    """
    # 确定测试数据路径
    data_paths = {
        "judgement": "./data_judgement/test.jsonl",
        "summary": "./data_sum/test.jsonl",
        "reading": "./阅读理解/test.jsonl"
    }

    if task_type not in data_paths:
        raise ValueError(f"Unknown task type: {task_type}")

    data_path = data_paths[task_type]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found: {data_path}")

    print(f"\n{'='*80}")
    print(f"开始测试任务: {task_type}")
    print(f"测试数据: {data_path}")
    print(f"模型路径: {model_path}")
    print(f"输出文件: {output_file}")
    print(f"{'='*80}\n")

    # 加载测试数据
    print("正在加载测试数据...")
    test_dataset = load_test_data(data_path, task_type)

    if sample_size is not None and sample_size < len(test_dataset):
        print(f"采样 {sample_size} 条数据进行测试")
        test_dataset = test_dataset[:sample_size]

    print(f"测试数据大小: {len(test_dataset)} 条")

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 加载模型
    print("正在加载模型...")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            local_files_only=True,  # 强制使用本地文件
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print(f"\n提示: 请确保模型已训练完成并保存在 {model_path}")
        raise

    # 启用推理模式
    FastLanguageModel.for_inference(model)

    # 进行预测
    print("\n开始预测...")
    predictions = []

    for item in tqdm(test_dataset, desc="预测进度"):
        # 构建输入
        prompt = alpaca_prompt.format(
            item["instruction"],
            item["input"],
            ""
        )

        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # 生成输出
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # 解码输出
        output_text = tokenizer.batch_decode(outputs)[0]

        # 提取生成的响应
        start = output_text.find("### Response:") + len("### Response:")
        end = output_text.find('<|endoftext|>')
        if end == -1:
            end = len(output_text)

        prediction = output_text[start:end].strip()

        # 保存预测结果
        result = {
            "id": item["id"],
            "instruction": item["instruction"],
            "input": item["input"][:200] + "..." if len(item["input"]) > 200 else item["input"],  # 截断显示
            "ground_truth": item["output"],
            "prediction": prediction
        }

        predictions.append(result)

    # 保存预测结果
    print(f"\n保存预测结果到: {output_file}")

    # 保存为JSONL格式
    with jsonlines.open(output_file, 'w') as writer:
        for pred in predictions:
            writer.write(pred)

    # 同时保存为可读的JSON格式
    json_output = output_file.replace('.jsonl', '_readable.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"可读格式保存到: {json_output}")

    # 显示一些样例
    print(f"\n{'='*80}")
    print("预测样例:")
    print(f"{'='*80}\n")

    for i, pred in enumerate(predictions[:3], 1):
        print(f"样例 {i}:")
        print(f"ID: {pred['id']}")
        print(f"指令: {pred['instruction']}")
        print(f"输入: {pred['input']}")
        print(f"\n真实答案: {pred['ground_truth'][:200]}...")
        print(f"\n预测结果: {pred['prediction'][:200]}...")
        print(f"\n{'-'*80}\n")

    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"总共处理: {len(predictions)} 条数据")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="统一测试脚本")
    parser.add_argument("--task", type=str, required=True,
                        choices=["judgement", "summary", "reading"],
                        help="任务类型: judgement(判决预测), summary(摘要提取), reading(阅读理解)")
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型路径")
    parser.add_argument("--output", type=str, default=None,
                        help="预测结果输出文件，默认为 predictions_{task}.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="生成的最大token数")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="采样测试的数据量（默认全部测试）")

    args = parser.parse_args()

    # 设置默认输出文件
    if args.output is None:
        args.output = f"predictions_{args.task}.jsonl"

    # 开始测试
    test(
        task_type=args.task,
        model_path=args.model,
        output_file=args.output,
        max_new_tokens=args.max_new_tokens,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()