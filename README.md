# MiniFold

MiniFold 是一个基于 Python 的轻量级蛋白结构分析与建模流程，专注于“最小可用预测器”思路：
- 使用 Qwen 生成若干条符合长度约束的二级结构候选（仅 H/E/C，支持多链，链间用 `|` 分隔）
- 使用 DeepSeek 评估每个候选的可靠性（返回 0–1 概率并按阈值筛选）
- 使用自研 Backbone Predictor 将二级结构映射为理想化主链扭转角，重建 N/CA/C 骨架并导出 PDB
- 使用 py3Dmol 生成交互式 HTML，可在浏览器直接查看结构

该流程完全在本地 Python 环境运行，不依赖 PyRosetta 或 ESMFold 等第三方折叠工具。

## 安装

1. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量（推荐 `.env` 文件）
   在项目根目录创建 `.env` 文件（请勿上传到 GitHub），内容示例：
   ```
   ARK_API_KEY=your_deepseek_api_key_here
   ARK_MODEL=deepseek-v3-2-251201
   ARK_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions

   DASHSCOPE_API_KEY=your_qwen_api_key_here
   QWEN_MODEL=qwen3-max
   QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```
   也可在 PowerShell 临时设置：
   ```powershell
   $env:ARK_API_KEY="..."
   $env:DASHSCOPE_API_KEY="..."
   ```

## Usage

Prepare a FASTA file (e.g., `test.fasta`) with your protein sequence.

运行示例：
```bash
python minifold.py test.fasta --env "membrane protein" --ssn 5 --threshold 0.5
```

结果目录结构（以 `test.fasta` 为例）：
- `output/test/requirements.txt`: 保存环境与筛选阈值等要求参数
- `output/test/test_ss_candidates.json`: DeepSeek 生成的二级结构候选
- `output/test/test_2_1.txt`, `test_2_2.txt`, ...: 每条候选单独保存
- `output/test/test_ss_kept.json`: 千问筛选保留的候选（含概率）
- `output/test/3d_structures/*.pdb`: 生成的 3D 骨架结构
- `output/test/3d_structures/*.html`: 交互式可视化页面
- `output/test/test_report.md`: 汇总报告
- `output/test/process_report.log`: 流程统计（候选数、保留数、生成模型数）

## 目录结构
- `minifold.py`：主流程入口（读取、生成、评估、建模、可视化、报告）
- `modules/input_handler.py`：FASTA 读取
- `modules/env_loader.py`：加载 `.env` 环境变量
- `modules/llm_module.py`：DeepSeek 注释与候选评估（返回概率）
- `modules/qwen_module.py`：Qwen 候选生成与解析（长度归一/多样性过滤）
- `modules/backbone_predictor.py`：骨架生成（N/CA/C），导出 PDB
- `modules/visualization.py`：py3Dmol HTML 可视化

## 注意事项
- `.env` 中包含密钥，请加入 `.gitignore`，切勿提交到仓库
- 骨架是理想化主链，不包含侧链与 O 原子，适合快速预览与轮廓分析
- 若网络环境限制，浏览器可能无法加载 3Dmol 的 CDN，可自行替换脚本源地址
- 当 Qwen 返回纯 H 或长度不符时，内部会进行裁剪/补齐与多样性过滤；严重不合规的候选会被丢弃
- 若所有候选均被丢弃，流程会生成一个回退模型用于占位与验证

## 依赖
`requirements.txt`：
- `requests`
- `py3Dmol`
- `biopython`
- `openai`
- `numpy`

## 许可协议
本项目遵循 MIT License。请在使用或分发时保留许可声明。
