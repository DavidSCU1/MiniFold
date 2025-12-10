# MiniFold

MiniFold 是一个基于 Python 的轻量级蛋白结构分析与建模流程。当前版本采用：
- DeepSeek 生成多条二级结构（仅 H/E/C，长度等于序列）
- 千问对每条二级结构进行可能性评估并筛选（阈值可配置）
- PyRosetta 基于筛选后的二级结构进行 3D 结构折叠与 Relax（已移除）
- Backbone Predictor 基于二级结构重建 N-CA-C 骨架并输出 PDB
- py3Dmol 生成交互式可视化 HTML

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. 设置环境变量（推荐 `.env` 文件）
   在项目根目录创建 `.env` 文件（**请勿上传此文件到 GitHub**），内容格式如下：
   ```
   ARK_API_KEY=your_deepseek_api_key_here
   ARK_MODEL=deepseek-v3-2-251201
   ARK_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
   DASHSCOPE_API_KEY=your_qwen_api_key_here
   QWEN_MODEL=qwen3-max
   QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```
   或在 PowerShell 临时设置：`$env:ARK_API_KEY="..."`、`$env:DASHSCOPE_API_KEY="..."`

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
- `minifold.py`: 主流程入口，包含目录创建与执行报告输出
- `modules/input_handler.py`: FASTA 读取
- `modules/env_loader.py`: 加载 `.env` 环境变量
- `modules/llm_module.py`: DeepSeek 生成二级结构与注释
- `modules/qwen_module.py`: 千问评估与筛选、单条候选评估
- `modules/backbone_predictor.py`: 自研骨架生成模块
- `modules/visualization.py`: 3D 可视化 HTML 生成

## 常见问题
- **DeepSeek**：若提示未授权，请检查 `ARK_API_KEY`、`ARK_MODEL` 与 `ARK_API_URL` 是否正确
- **千问**：若返回格式不为纯数字，筛选将失败；请确认 `DASHSCOPE_API_KEY` 与 `QWEN_API_URL`（兼容模式）
