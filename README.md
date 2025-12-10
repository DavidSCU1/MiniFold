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

## 使用示例

准备 `test.fasta`（单条或多条序列均可）。运行：
```bash
python minifold.py test.fasta --env "cytosolic protein" --ssn 5 --threshold 0.3
```
=======
### GUI 方式

提供了简洁的桌面界面，便于选择文件并运行流程：
```bash
python gui.py
```
在界面中选择 FASTA、输出目录，设置 `环境/ssn/阈值`，点击“开始运行”即可。右侧滑杆可调界面缩放（0.9–1.5），提升不同分辨率下的清晰度。

>>>>>>> Stashed changes
说明：
- `--env` 为环境描述（例如膜蛋白、嗜热环境等），用于引导候选生成与评估
- `--ssn` 为候选条数上限
- `--threshold` 为 DeepSeek 评估保留阈值（0–1）

## 输出内容
以 `test.fasta` 为例，输出目录 `output/test/` 包含：
- `requirements.txt`：记录运行参数（环境、候选数、阈值）
- `raw_qwen.txt`：Qwen 的原始候选文本（可能包含超长行，后续会归一到序列长度）
- `test_ss_candidates.json`：经解析与长度归一的候选（含多链）
- `case_*/`：每个候选的分链文件与元数据（若评估低于阈值会被清理）
- `cases_kept.json`：保留候选清单（含概率与文件名）
- `3d_structures/*.pdb`：Backbone Predictor 生成的骨架 PDB
- `3d_structures/*.html`：对应的交互式可视化页面
- `test_report.md`：汇总报告（序列、候选、评估结果、模型链接）
- `process_report.log`：流程统计（候选数、保留数、生成模型数等）

## 工作流程
1. 从 FASTA 读取氨基酸序列
2. 调用 Qwen 生成恰好 `ssn` 条候选（仅 H/E/C；多链用 `|`；总长度严格等于序列长度）
3. 对候选进行长度归一与多样性检查（至少包含两种字符），并落盘为分链文件
4. 调用 DeepSeek 对每个候选进行概率评估，按阈值保留高分候选
5. 对保留候选调用 Backbone Predictor：
   - H/E/C 分别映射到理想化的 `phi/psi` 扭转角（H:-62/-41；E:-135/135；C 为带噪的环区角）
   - 使用固定键长/夹角构建 N/CA/C 坐标并导出 PDB
6. 使用 py3Dmol 生成 HTML，可在浏览器直接查看结构（使用 cdnjs 加载 3Dmol）
7. 生成报告与统计信息

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
