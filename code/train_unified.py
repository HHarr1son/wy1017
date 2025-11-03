"""
统一训练脚本 - 支持判决预测、摘要提取和阅读理解三个任务
基于unsloth框架进行高效训练
"""

import unsloth
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported, FastLanguageModel
from datasets import Dataset
import torch
import jsonlines
from tqdm import tqdm
import argparse
import os

# 训练配置
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


def load_dataset(data_path, task_type):
    """
    加载数据集并转换为统一格式

    Args:
        data_path: 数据文件路径
        task_type: 任务类型 (judgement/summary/reading)

    Returns:
        处理后的数据列表
    """
    dataset = []

    with jsonlines.open(data_path) as reader:
        for each_line in reader:
            if task_type == "judgement":
                # 判决预测任务
                item = {
                    "instruction": "根据案件事实和法律条文，预测判决结果",
                    "input": each_line["input"],
                    "output": each_line["output"]
                }
            elif task_type == "summary":
                # 摘要提取任务
                item = {
                    "instruction": "请对以下法律文书进行摘要",
                    "input": each_line["input"],
                    "output": each_line["output"]
                }
            elif task_type == "reading":
                # 阅读理解任务（已有instruction字段）
                item = {
                    "instruction": each_line["instruction"],
                    "input": each_line["input"],
                    "output": each_line["output"]
                }
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            dataset.append(item)

    return dataset


def train(task_type, model_name, output_dir, max_steps=500, batch_size=2, 
          gradient_accumulation_steps=4, save_steps=100, logging_steps=10, 
          warmup_steps=5):
    """
    训练模型

    Args:
        task_type: 任务类型 (judgement/summary/reading)
        model_name: 预训练模型名称
        output_dir: 模型输出目录
        max_steps: 最大训练步数
        batch_size: 批次大小
        gradient_accumulation_steps: 梯度累积步数
        save_steps: 保存检查点的步数间隔
        logging_steps: 日志记录的步数间隔
        warmup_steps: 学习率预热步数
    """
    # 确定数据路径
    data_paths = {
        "judgement": "./data_judgement/train.jsonl",
        "summary": "./data_sum/train.jsonl",
        "reading": "./阅读理解/train.jsonl"
    }

    if task_type not in data_paths:
        raise ValueError(f"Unknown task type: {task_type}. Choose from {list(data_paths.keys())}")

    data_path = data_paths[task_type]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"\n{'='*80}")
    print(f"开始训练任务: {task_type}")
    print(f"数据路径: {data_path}")
    print(f"模型: {model_name}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")

    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset(data_path, task_type)
    print(f"数据集大小: {len(dataset)} 条")

    # 加载模型和tokenizer
    print("正在加载模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # 配置LoRA
    print("配置LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        """格式化prompt"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    # 转换为HuggingFace Dataset格式
    print("转换数据格式...")
    dataset = Dataset.from_list(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 创建训练器
    print("创建训练器...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            save_steps=save_steps,
            save_total_limit=2,  # 只保留最近2个检查点
            report_to="none",
        ),
    )

    # 开始训练
    print("\n开始训练...")
    trainer_stats = trainer.train()
    print("\n训练统计:")
    print(trainer_stats)

    # 保存最终模型
    print(f"\n保存最终模型到: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n{'='*80}")
    print("训练完成！")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="统一训练脚本")
    parser.add_argument("--task", type=str, required=True,
                        choices=["judgement", "summary", "reading"],
                        help="任务类型: judgement(判决预测), summary(摘要提取), reading(阅读理解)")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-3B",
                        help="预训练模型名称，例如: unsloth/Qwen2.5-3B, unsloth/gemma-3-4b-it")
    parser.add_argument("--output", type=str, default=None,
                        help="模型输出目录，默认为 model_{task}")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="最大训练步数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="保存检查点的步数间隔")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="日志记录的步数间隔")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="学习率预热步数")

    args = parser.parse_args()

    # 设置默认输出目录
    if args.output is None:
        args.output = f"model_{args.task}"

    # 开始训练
    train(
        task_type=args.task,
        model_name=args.model,
        output_dir=args.output,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_steps=args.warmup_steps
    )


if __name__ == "__main__":
    main()