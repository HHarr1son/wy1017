# 法律 AI 智能系统

基于深度学习技术的法律智能辅助平台，提供判决预测、摘要提取、阅读理解和知识检索四大核心功能。

## 快速启动

### 1. 安装 GPU 支持（推荐，速度快 10-50 倍）

```bash
cd /mnt/d/Project/WY20251027/code
./install_gpu_version.sh
```

### 2. 启动后端 API 服务器

```bash
source venv_wsl/bin/activate
python api_server_real.py
```

后端运行在：`http://localhost:8000`

### 3. 启动前端 Web 应用

```bash
cd /mnt/d/Project/WY20251027/chatbot-ui
npm run dev
```

前端运行在：`http://localhost:3000`

### 4. 访问应用

浏览器打开：`http://localhost:3000/zh/temp/legal`

---

## 项目结构

```
code/
├── api_server_real.py          # API 服务器（支持真实模型推理）
├── train_unified.py            # 模型训练脚本
├── test_unified.py             # 模型测试脚本
├── data_split.py               # 数据集分割工具
├── model_judgement/            # 判决预测模型权重（114MB LoRA）
├── model_summary/              # 摘要提取模型权重（114MB LoRA）
├── data_judgement/             # 判决预测训练数据
├── data_sum/                   # 摘要提取训练数据
├── venv_wsl/                   # WSL Python 虚拟环境
├── requirements.txt            # 训练依赖
├── requirements_real_model.txt # API 服务器依赖
└── install_gpu_version.sh      # GPU 版本安装脚本
```

---

## 技术架构

### 后端
- **框架**: FastAPI + Uvicorn
- **基础模型**: Qwen/Qwen2.5-3B-Instruct
- **微调方法**: LoRA (PEFT)
- **推理**: PyTorch 2.9 + Transformers 4.57

### 前端
- **框架**: Next.js 14 (App Router)
- **UI**: React 18 + TypeScript + Tailwind CSS
- **设计**: Apple 风格，全屏展示，渐变主题

### 模型
- **判决预测**: LoRA adapter (114MB)
- **摘要提取**: LoRA adapter (114MB)
- **阅读理解**: 待训练
- **知识检索**: 向量数据库检索

---

## API 端点

### 判决预测
```bash
POST http://localhost:8000/api/judgement
Content-Type: application/json

{
  "instruction": "根据案件事实，预测可能的判决结果",
  "input": "案件事实描述...",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### 摘要提取
```bash
POST http://localhost:8000/api/summary
Content-Type: application/json

{
  "instruction": "提取法律文书的核心内容摘要",
  "input": "法律文书全文...",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### 阅读理解
```bash
POST http://localhost:8000/api/reading
Content-Type: application/json

{
  "instruction": "问题内容",
  "input": "法律文本..."
}
```

---

## 性能优化

### GPU vs CPU
- **CPU 模式**: 每次推理 1-5 分钟
- **GPU 模式** (RTX 3060 Ti): 每次推理 5-30 秒
- **推荐**: 使用 GPU 模式（运行 `install_gpu_version.sh`）

### 显存占用
- 基础模型加载: ~4GB
- LoRA adapter: ~500MB
- 推理时: ~4-6GB
- **要求**: 至少 6GB 显存

---

## 开发命令

### 训练新模型
```bash
python train_unified.py
```

### 测试模型
```bash
python test_unified.py
```

### 数据集分割
```bash
python data_split.py
```

---

## 故障排除

### Q: API 服务器启动失败？
A: 检查依赖是否安装完整：
```bash
source venv_wsl/bin/activate
pip install -r requirements_real_model.txt
```

### Q: 前端无法连接后端？
A: 确认后端运行在 `http://localhost:8000`，检查 CORS 配置。

### Q: 推理速度慢？
A: 安装 GPU 版本：`./install_gpu_version.sh`

### Q: 显存不足？
A: 减少 `max_new_tokens` 参数，或使用 CPU 模式。

---

## 开发团队

- 项目负责人：整体架构设计与项目管理
- 算法工程师：模型训练、优化与部署
- 全栈开发：前后端开发与系统集成

---

## 许可证

本项目仅供学习研究使用。
