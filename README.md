# MiniFold：轻量级蛋白结构预测与多硬件加速工作流

MiniFold 是一个将 LLM 生成的二级结构“直觉”和基于 PyTorch 的物理约束优化结合在一起的轻量级蛋白结构预测系统。它从 FASTA 序列出发，经过 LLM 多模型投票与环境驱动的重审、物理约束主链折叠与装配、可选的 NPU Head 评分与 MD 后处理，输出全原子 PDB 结构和 HTML 3D 可视化结果。

## 功能概览
- LLM + 多模型投票：
  - 通过 Ark 等多模型对二级结构候选进行打分与投票。
  - 支持根据环境描述进行“主考官优化-重审”，反复 refine 二级结构。
- 物理一致的主链折叠：
  - 使用可微物理损失（Ramachandran 先验、β 片层配对约束、疏水核心与排斥项等）进行优化。
  - 自动构建主链和侧链，严格控制肽键几何与链连续性。
- 多硬件加速：
  - GPU / iGPU：通过 `torch`、`torch-directml`、IPEX 等后端承担主力折叠与装配。
  - NPU / 轻量设备：使用固定窗口 Head 网络对输出结构进行评分与轻量精炼。
  - CPU：负责数据加载、调度、日志等非重计算任务。
- 多种交互入口：
  - 命令行工作流（适合批量任务与自动化）。
  - 桌面 GUI（Tkinter）一键运行，实时日志。
  - Web UI（内置 HTTP Server + HTML 前端）支持浏览器 3D 预览。

## 目录结构
- `minifold.py`：命令行入口，串联完整工作流（FASTA → LLM → 折叠 → 评估 → 输出）。
- `gui.py`：桌面图形界面入口（Tkinter），封装常用参数与日志显示。
- `web_ui/`
  - `index.html`：Web 前端界面（日志 + 表单 + 3D 预览）。
  - `server.py`：简单 HTTP API + 静态资源服务，负责调用主流程并管理输出。
- `modules/`：核心算法与工具模块
  - `input_handler.py`：读取并解析 FASTA，回退到原始序列解析。
  - `ss_generator.py`：基于 PyBioMed AAIndex 以及统计/随机策略生成二级结构候选。
  - `ark_module.py`：封装 Ark Chat Completions 接口的投票、审核、二级结构 refine。
  - `backbone_predictor.py`：基于物理先验的主链折叠与约束优化。
  - `sidechain_builder.py`：侧链构建与简单 packing。
  - `assembler.py`：多链 PDB 装配、链拆分与重组。
  - `refine.py`：Ramachandran 微调、疏水核心分析与可选局部优化。
  - `quality.py`：Rama 通过率、接触能量、简单 TM-score 代理等质量指标。
  - `env_loader.py`：从多种 `.env` 文件自动注入环境变量。
  - `igpu_predictor.py` / `igpu_runner.py`：iGPU / IPEX 加速相关逻辑。
  - `npu_runner.py`：固定窗口 Head 网络，对折叠结果进行评分并写入 PDB `REMARK`。
  - `esm_runner.py`：封装 ESM-2 / ESMFold 模型调用，并与本地 3D 模型权重衔接。
  - `visualization.py`：生成基于 3Dmol.js 的 HTML 3D 视图。
- `scripts/`
  - `post_assemble.py`：对已经生成的链结构做后装配和可视化。
  - `dml_probe.py`：用于 DirectML / 设备探测的小工具。
- `3d_model/best_model_gpu.pt`：ESMFold 相关 3D 主干模型权重（已随仓库提供）。
- `MiniFold_Web.bat`：Windows 下启动 Web 界面的批处理脚本。

## 环境与依赖

### Python 版本
- 推荐：`Python >= 3.9`（64 位）

### 必需依赖
核心运行时依赖在 `requirements.txt` 中声明，主要包括：
- `numpy`、`scipy`
- `torch`
- `requests`
- `pybiomed`（提供 AAIndex1）  
- `biopython`（解析 FASTA，`Bio.SeqIO`）
- `tqdm`（进度条工具，部分脚本/扩展可使用）
- `py-cpuinfo`（CPU / iGPU 能力探测）

在项目根目录执行：

```bash
pip install -r requirements.txt
```

ESM 相关说明：
- 默认情况下，MiniFold 使用仓库内置的 `3d_model/best_model_gpu.pt` 与本地 `model.py` 中的模型定义，无需额外安装 `esm` 包。
- 如需集成官方 `facebookresearch/esm` / `fair-esm` 推理代码，可在独立环境中安装相应包，并通过命令行参数 `--esm-env` 或 Web UI 中的 ESM 环境配置指向该环境。

### 可选加速依赖
- `torch-directml`：在 Windows 上为 Intel / AMD / NVIDIA 显卡提供统一 iGPU 加速后端。
- Intel IPEX：`intel-extension-for-pytorch`，用于 Intel XPU / Arc / Ultra 等设备。
- `openmm`：用于可选的能量最小化 / 简单 MD 精炼（建议通过 `conda` 安装）。
- `volcengine`：可选的 Ark 官方 Python SDK，本项目默认通过 HTTP API 调用 Ark 服务，并不强制依赖 SDK。

可选安装示例：

```bash
# Windows DirectML
pip install torch-directml

# Intel IPEX（需额外索引）
pip install intel-extension-for-pytorch \
  --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# OpenMM（建议使用 conda 安装）
conda install -c conda-forge openmm
```

### API Key 与环境变量

Ark 等大模型配置通过环境变量注入。推荐在项目根目录新建 `.env`（或 `.env.local`）：

```ini
ARK_API_KEY=你的Key
ARK_MODELS=doubao-seed-1-8-251215,deepseek-v3-2-251201,doubao-seed-1-6-251015,kimi-k2-thinking-251104,deepseek-r1-250528
ARK_API_URL=https://ark.cn-beijing.volces.com/api/v3/chat/completions
```

模块 `modules/env_loader.py` 会自动在当前工作目录与项目根目录查找 `.env` / `.env.oneapi` / `.env.local` 文件并注入变量。

## 使用方式

### 1. 命令行工作流

基础调用示例（CPU 或默认后端）：

```bash
python minifold.py input.fasta \
  --outdir output
```

启用 LLM 多模型投票 + 环境约束：

```bash
python minifold.py input.fasta \
  --outdir output \
  --env "膜蛋白，需跨膜螺旋" \
  --ssn 5 \
  --threshold 0.5
```

使用 iGPU / GPU 加速与 NPU Head：

```bash
python minifold.py input.fasta \
  --outdir output \
  --env "胞内可溶性酶" \
  --ssn 5 \
  --threshold 0.5 \
  --igpu --backend directml --igpu-env MiniFold_iGPU \
  --npu --npu-env MiniFold_NPU
```

常用参数（详见 `minifold.py`）：
- `--ssn`：从 LLM 侧生成的二级结构候选数量。
- `--threshold`：Ark 投票平均得分阈值，低于阈值的候选会被过滤。
- `--target-chains`：强制目标链数量（如 1 或 2），用于多聚体建模。
- `--igpu` / `--igpu-env`：启用 iGPU 折叠并指定外部 conda 环境或 Python 路径。
- `--backend`：选择加速后端：`auto|directml|ipex|cuda|cpu|oneapi_cpu`。
- `--npu` / `--npu-env`：启用 NPU Head 评分 / 轻量精炼。
- `--refine-ramachandran`：在折叠后对主链 φ/ψ 角做局部平滑。
- `--refine-hydrophobic`：分析疏水核心并尝试压实。
- `--repack-long-sides`：对长侧链（Lys/Gln/Glu 等）进行重新 packing。
- `--md` / `--md-steps`：选择后处理引擎（`openmm|amber|gromacs`）及步数，需用户自行准备环境。
- `--esm-backbone` / `--esm-env`：使用 `3d_model/best_model_gpu.pt` 提供的 ESM 主干模型，可在独立环境中运行。

运行结束后，默认在 `output/<输入文件前缀>/` 下生成：
- 中间 JSON 与文本：SS 候选、Ark 投票结果、refine 记录等。
- `3d_structures/`：折叠生成的 PDB 文件以及对应 HTML 可视化。

### 2. 桌面 GUI

```bash
python gui.py
```

主要功能：
- 选择 FASTA 文件与输出目录。
- 填写环境描述、候选数量、阈值等参数。
- 勾选“🚀 启用 iGPU 加速”和“⚡ 启用 NPU 加速”，以及各自使用独立环境的名称或 Python 路径。
- 实时查看运行日志，并在窗口下方展示状态。

适合桌面使用与交互调参。

### 3. Web 界面

直接使用 Python 启动：

```bash
python web_ui/server.py
```

或在 Windows 上执行批处理脚本：

```bash
MiniFold_Web.bat
```

然后在浏览器访问：

```text
http://localhost:9000
```

Web 界面支持：
- 在左侧表单中选择 FASTA、输出目录与运行环境。
- 启用/关闭 iGPU 与 NPU，以及对应的外部环境名称。
- 实时查看日志与进度条，并在结果区加载 3Dmol.js 的结构预览。

如需修改 Web 服务端口，可编辑 `web_ui/server.py` 中的 `PORT` 变量。

## NPU Head 评分与精炼说明

NPU 部分由 `modules/npu_runner.py` 实现，核心思想是：
- 从 PDB 中抽取 CA 坐标，构造固定长度（默认 128×128）的距离矩阵补丁。
- 使用一个小型前馈网络对补丁进行打分，输出 0–1 之间的自然性评分。
- 在 PDB 文件中插入一行 `REMARK NPU_HEAD Applied FixedPatch128 Score x.xxxx`，用于记录评分。

设备自动检测顺序：
- 优先使用 MPS（如 macOS 上的 Apple GPU）。
- 尝试使用 DirectML 设备（如 Windows 上的 iGPU）。
- 回退到 CPU。

在命令行、GUI 与 Web 中启用 `--npu`（或勾选对应选项）即可触发此流程。

## 典型工作流

1. 从 FASTA 读取序列（支持多序列）并进行预处理。
2. 使用 PyBioMed AAIndex 与统计模型生成二级结构候选。
3. 调用 Ark 多模型对各个候选进行概率评估与投票（可选环境文本约束）。
4. 对最佳候选执行环境驱动的二级结构 refine，并在必要时反复投票。
5. 基于最终二级结构，以物理约束优化方式折叠主链，并构建侧链。
6. 对折叠结果进行质量评估（Ramachandran、β 片层配对、简单 TM-score 代理等）。
7. 可选：运行 NPU Head 评分与 remark 注入。
8. 可选：调用 OpenMM / 其它 MD 引擎进行短程能量最小化或微型 MD。
9. 生成最终 PDB 与 3D HTML 可视化，以及中间日志与 JSON 报告。

## 常见问题

- `torch-directml` 安装失败  
  - 检查 Python 与 PyTorch 版本是否兼容，并确保在 Windows 环境下安装。
- IPEX 加速不可用  
  - 需要匹配的 Intel GPU 驱动与 IPEX 版本，可参考 Intel 官方文档。
- Web 端口占用或访问失败  
  - 默认端口为 `9000`，如被占用可修改 `web_ui/server.py` 中的 `PORT` 并重新启动。
- OpenMM 后处理失败  
  - 通常与环境或依赖不完整相关，可暂时禁用 `--md openmm` 或根据报错补全 conda 依赖。
- 无法从 Ark 获取有效得分  
  - 检查 `ARK_API_KEY`、`ARK_API_URL` 是否配置正确，或查看网络连通性。

## 许可证

本项目使用 MIT License。
