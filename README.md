# MiniFold

MiniFold 是一个基于 Python 的轻量级蛋白结构分析与建模流程，专注于“最小可用预测器”思路。它利用大语言模型（LLM）进行二级结构预测与评估，结合经典的几何算法进行全原子 3D 建模与复合物组装。

**核心特性：**
- **LLM 驱动预测**：
  - 使用 **Qwen** 生成符合长度约束的二级结构候选（H/E/C）。
  - 使用 **DeepSeek** 评估每个候选的可靠性（概率打分）。
- **全原子 3D 建模 (Full-Atom)**：
  - 不仅重建 N-CA-C 骨架，还利用 **NeRF (Natural Extension Reference Frame)** 算法和几何模板自动构建所有侧链原子及骨架氧原子，提供完整的化学细节。
- **复合物组装 (Complex Assembly)**：
  - 支持多链预测任务，使用 **L-BFGS-B** 优化算法，基于回转半径（Rg）和链间碰撞（Clash）最小化原则，自动将分散的链组装成紧凑的复合物结构。
- **现代化交互界面**：
  - 提供 CLI、GUI 和 Web UI 三种使用方式。
  - **Web UI** 支持实时进度条显示、任务节点追踪、ETA 预估以及交互式 3D 可视化（Cartoon + Sticks 模式）。

该流程完全在本地 Python 环境运行（LLM 通过 API 调用），不依赖 PyRosetta 或 ESMFold 等庞大的第三方折叠工具。

## 安装

1. **克隆项目**
   ```bash
   git clone https://github.com/DavidSCU1/MiniFold.git
   cd MiniFold
   ```

2. **安装依赖**
   建议使用 Conda 创建独立环境：
   ```bash
   conda create -n minifold python=3.9
   conda activate minifold
   pip install -r requirements.txt
   ```
**iGPU 加速（DirectML）提示：**
- Windows 上建议安装 `torch-directml` 以启用 Intel/AMD/NVIDIA 等设备的通用加速。
- 默认 `requirements.txt` 同时包含 `torch` 与 `torch-directml`，如遇到冲突，可在独立 Conda 环境中安装并在 Web UI 的 iGPU 设置里填写该环境名称。
- 运行时在 Web UI 勾选 “iGPU Acceleration” 或 CLI 使用 `--igpu` 即可启用；如需隔离依赖，可再配合 `--igpu-env <env_name>`。

3. **配置环境变量**
   在项目根目录创建 `.env` 文件（请勿上传到 GitHub），填写你的 API Key：
   ```ini
   # DeepSeek (Volcengine)
   ARK_API_KEY=your_deepseek_api_key_here
   ARK_MODEL=deepseek-v3-2-251201
   ARK_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions

   # Qwen (DashScope)
   DASHSCOPE_API_KEY=your_qwen_api_key_here
   QWEN_MODEL=qwen3-max
   QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

## 使用指南

### 1. Web 界面 (推荐)
提供最完整的体验，包括进度追踪和结果管理。

**启动：**
```bash
python web_ui/server.py
```
或者直接双击运行 `MiniFold_Web.bat`。

**访问：**
打开浏览器访问 `http://localhost:9000`。

**功能亮点：**
- **实时进度监控**：精确显示任务百分比、当前处理步骤（如 "Predicting SS", "Generating 3D Structures"）和预计剩余时间。
- **环境配置**：支持指定全局运行环境和 iGPU 独立环境（实现环境隔离）。
- **Product Manager**：浏览历史任务与 3D 结构。模型在预测阶段自动完成复合体装配与侧链补全（Refined），无需额外 Assemble 操作。

### 2. GUI 桌面版
简洁的图形界面，适合快速单任务运行。

**启动：**
```bash
python gui.py
```
在界面中选择 FASTA 文件、设置参数并运行。支持界面缩放以适应高分屏。

### 3. 命令行 (CLI)
适合批量处理或服务器环境。

**运行：**
```bash
python minifold.py test.fasta --env "cytosolic protein" --ssn 5 --threshold 0.3 --outdir output
```

**参数说明：**
- `--env`: 蛋白环境描述（如 "membrane protein", "cytosolic"），辅助 LLM 判断。
- `--ssn`: 生成的二级结构候选数量（默认 5）。
- `--threshold`: DeepSeek 评估的保留阈值（0-1，默认 0.5）。
- `--igpu`: 启用 iGPU 加速（如支持）。
- `--igpu-env`: 指定 iGPU 模块运行的 Conda 环境名称。

## 输出结果

结果默认保存在 `output/<job_name>/` 目录下：

- **3d_structures/**: 包含生成的 PDB 文件和 HTML 可视化文件。
  - `*_model_*.pdb`: 主链与基本侧链模型（作为中间产物保留）。
  - `*_refined.pdb`: 精修复合体（自动装配并补全侧链），作为最终 PDB 输出。
  - `*_model_*.html`: 交互式网页，默认使用精修版 PDB 展示（Cartoon, Sphere, Surface）。
- **case_*/**: 中间过程文件（分链序列等）。
- **\*_ss_candidates.json**: 二级结构候选数据。
- **\*_results.json**: 最终生成的模型清单。
- **\*_annotation.txt**: LLM 对序列的功能注释。

## 技术细节

- **侧链构建**：基于几何模板和 NeRF 算法，从主链坐标推导侧链原子位置。目前使用最常见的转子构象（Rotamer）。
- **刚体组装**：通过 scipy 的 `minimize(method='L-BFGS-B')` 优化旋转和平移矩阵，目标函数结合了回转半径（紧凑性）和原子间距离（避免碰撞）。
- **进度通信**：核心流程通过标准输出打印 `[PROGRESS]` 标签，Web 服务器实时捕获并推送至前端，解决了传统 CLI 工具在 Web 上进度不可见的问题。

## 许可证

MIT License
