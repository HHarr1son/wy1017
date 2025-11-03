"""
法律AI任务统一API服务 - 真实模型版本
支持判决预测、摘要提取两个任务（使用训练好的LoRA权重）
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from typing import Optional
import uvicorn
import os

app = FastAPI(title="Legal AI API (Real Models)", version="1.0.0")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 推理配置
max_seq_length = 2048

# Alpaca prompt模板
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 全局模型缓存
models_cache = {}

# 基础模型名称
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# 模型路径配置（LoRA适配器路径）
MODEL_PATHS = {
    "judgement": "./model_judgement",
    "summary": "./model_summary"
}


class PredictionRequest(BaseModel):
    instruction: str
    input: str
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class PredictionResponse(BaseModel):
    output: str
    task: str


def load_model(task_type: str):
    """加载基础模型+LoRA适配器"""
    if task_type in models_cache:
        return models_cache[task_type]

    adapter_path = MODEL_PATHS.get(task_type)
    if not adapter_path:
        raise ValueError(f"Unknown task type: {task_type}")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")

    print(f"Loading model for task: {task_type}")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  LoRA adapter: {adapter_path}")

    try:
        # 1. 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # 2. 加载基础模型
        print("  Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )

        # 3. 加载 LoRA 适配器
        print("  Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        model.eval()

        models_cache[task_type] = (model, tokenizer)
        device = next(model.parameters()).device
        print(f"✓ Model loaded successfully for task: {task_type}")
        print(f"  Device: {device}")
        print(f"  Model type: Base + LoRA adapter")

        return model, tokenizer
    except Exception as e:
        print(f"✗ Failed to load model for task {task_type}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


def predict(task_type: str, instruction: str, input_text: str,
            max_new_tokens: int = 512, temperature: float = 0.7,
            top_p: float = 0.9) -> str:
    """执行预测"""
    model, tokenizer = load_model(task_type)

    # 构建prompt
    prompt = alpaca_prompt.format(instruction, input_text, "")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_length)

    # 移动到模型所在的设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 生成输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码输出
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 提取生成的响应
    start = output_text.find("### Response:") + len("### Response:")
    end_markers = ['<|endoftext|>', '<|im_end|>', '</s>']
    end = len(output_text)

    for marker in end_markers:
        marker_pos = output_text.find(marker, start)
        if marker_pos != -1:
            end = min(end, marker_pos)

    prediction = output_text[start:end].strip()

    return prediction


@app.get("/")
async def root():
    """健康检查端点"""
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    return {
        "status": "running",
        "version": "Real Models (Base + LoRA)",
        "base_model": BASE_MODEL,
        "device": device_info,
        "tasks": list(MODEL_PATHS.keys()),
        "loaded_models": list(models_cache.keys())
    }


@app.post("/api/judgement", response_model=PredictionResponse)
async def judgement_prediction(request: PredictionRequest):
    """判决预测API"""
    try:
        output = predict(
            task_type="judgement",
            instruction=request.instruction,
            input_text=request.input,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return PredictionResponse(output=output, task="judgement")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summary", response_model=PredictionResponse)
async def summary_extraction(request: PredictionRequest):
    """摘要提取API"""
    try:
        output = predict(
            task_type="summary",
            instruction=request.instruction,
            input_text=request.input,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        return PredictionResponse(output=output, task="summary")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reading", response_model=PredictionResponse)
async def reading_comprehension(request: PredictionRequest):
    """阅读理解API - 暂时返回模拟响应"""
    # 阅读理解任务暂时没有训练好的模型，返回提示信息
    mock_response = f"""**问题**：{request.instruction}

**基于文本的回答**：
阅读理解模型尚未训练完成。目前仅支持判决预测和摘要提取两个任务。

如需使用阅读理解功能，请：
1. 完成阅读理解任务的模型训练
2. 将模型权重放置在 ./model_reading 目录
3. 在 MODEL_PATHS 中添加配置

提供的文本：{request.input[:100]}...
"""
    return PredictionResponse(output=mock_response, task="reading")


if __name__ == "__main__":
    print("="*80)
    print("Legal AI API Server (Real Models with LoRA)")
    print("="*80)
    print(f"Base Model: {BASE_MODEL}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Available tasks with trained models: {list(MODEL_PATHS.keys())}")
    print(f"LoRA adapter paths: {MODEL_PATHS}")
    print("="*80)
    print("\nNote: Models will be loaded on first request to save memory")
    print("="*80)

    uvicorn.run(app, host="0.0.0.0", port=8000)
