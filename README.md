# MiniFold：轻量级蛋白结构预测与加速工作流

MiniFold 是一个将生物大模型（LLM）与物理约束优化结合的轻量级蛋白结构预测系统。它通过 LLM 生成与环境一致的二级结构候选，再用高效的物理引擎进行主链折叠和装配，并支持 iGPU/NPU 等设备进行推理加速。

**核心理念**：LLM 给“直觉”，物理做“裁决”。

## 特性概览
- 多模型投票与重审：整合多个生物模型的 SS 候选，并根据环境描述进行“主考官优化-重审”。
- 物理有效的主链折叠：基于 PyTorch 的可微物理引擎，保证正确的肽键几何、Ramachandran 合理性、排斥碰撞与链连续性。
- 硬件加速策略：
  - GPU/iGPU：承担主力计算（多链折叠/装配），支持 DirectML、IPEX、CUDA、MPS。
  - NPU：承担固定形状、小型、重复计算密集的 Head/输出层评分与精炼。
  - CPU：数据加载、日志、非计算密集的流程。
- 交互式 Web 界面与桌面 GUI：实时日志与 3D 预览（HTML/3Dmol），支持外部 conda 环境运行。

## 安装与环境
### 依赖
- 必选：`python>=3.9`、`numpy`、`torch`、`requests`、`volcengine`、`pybiomed`、`tqdm`、`colorama`、`py-cpuinfo`
- 可选（按需）：
  - `torch-directml`（Windows 跨厂 iGPU 加速）
  - Intel IPEX（Intel XPU/Arc/Ultra 加速）
  - `openmm`（可选的短程能量最小化/微型 MD 精炼，推荐用 conda 安装）

在项目根目录执行：
```bash
pip install -r requirements.txt
```
可选安装示例：
- Windows DirectML：
```bash
pip install torch-directml
```
- Intel IPEX（需额外索引）：
```bash
pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
- OpenMM（建议 conda）：
```bash
conda install -c conda-forge openmm
```

### API Key 与环境变量
在项目根目录新建 `.env`（可选），配置 Ark 等模型的 Key：
```ini
ARK_API_KEY=你的Key
ARK_MODELS=doubao-seed-1-6-251015,deepseek-v3-2-251201
```

## 快速开始
### 命令行
```bash
python minifold.py <input.fasta> \
  --outdir output \
  --ssn 5 \
  --threshold 0.5 \
  --igpu --backend directml --igpu-env MiniFold_iGPU \
  --npu --npu-env MiniFold_NPU
```
- `--igpu`/`--igpu-env`：启用 iGPU 并指定 iGPU 的 conda 环境（或 Python 路径）
- `--backend`：选择加速后端（`auto|directml|ipex|cuda|cpu|oneapi_cpu`）
- `--npu`/`--npu-env`：启用 NPU 固定 head 评分，并指定 NPU 的 conda 环境（或 Python 路径）
- `--target-chains`：限定链数（如 1/2）

### 桌面 GUI
```bash
python gui.py
```
- 勾选“🚀 启用 iGPU 加速”与“⚡ 启用 NPU 加速”
- 如需环境隔离，分别填写对应的环境名或 Python 路径（例如 `MiniFold_NPU`）

### Web 界面
```bash
python web_ui/server.py
```
浏览器访问 `http://localhost:9000`（默认端口）。左侧面板可：
- 选择全局运行环境（外部 conda 环境）
- 勾选 iGPU/NPU，并填写各自环境名
- 运行后实时查看日志与 3D 预览

## NPU 加速说明
**适配原则**：NPU 擅长固定形状、小型、重复计算密集的全连接/卷积。MiniFold 将 NPU 用于“输出 Head 的固定窗口评分与轻量精炼”，避免进入序列长度动态的 Evoformer/Attention 大模块。
- 固定窗口评分由 `modules/npu_runner.py` 进行：
  - 解析生成的 PDB，取 CA 坐标构造固定大小的距离补丁（默认 128×128）
  - 使用小型前馈网络（FP16）计算评分，并写入 PDB 的 `REMARK`：`REMARK NPU_HEAD Applied FixedPatch128 Score x.xxx`
  - 支持 DirectML/MPS/CPU 自动探测
- 开启方式：
  - CLI：`--npu`（当前进程）或同时加 `--npu-env <env>`（外部环境）
  - GUI/Web：勾选 NPU，并填入环境名

## 典型流程
1. 读取 FASTA，生成二级结构候选（PyBioMed + LLM 多模型投票）
2. 如提供环境描述，执行“主考官优化-重审”，并提取物理约束
3. 主链折叠与装配（iGPU/GPU/CPU），生成 PDB 与 HTML 可视化
4. NPU 输出 Head 评分（可选，固定窗口），写入 `REMARK`
5. 可选后处理（OpenMM 简单最小化/微型 MD），更新 PDB
6. 输出报告与 3D 预览

## 常见问题
- `torch-directml` 安装失败
  - 请确认 Python 与 PyTorch 版本兼容，并在 Windows 环境下安装
- IPEX 加速不可用
  - Intel GPU 驱动与 IPEX 版本需匹配，参考官方索引安装
- Web 端口
  - 默认端口为 `9000`，如需调整可编辑 `web_ui/server.py` 的 `PORT`
- OpenMM 后处理失败
  - 可跳过或使用 conda-forge 安装 OpenMM

## 许可证
MIT License
