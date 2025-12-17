# 目标概述
- 引入更全面的转子库（Dunbrack 2010），基于邻近残基空间位阻与简化能量评估动态择优构象。
- 增加侧链优化（保持骨架不动、仅优化χ角），用可微分能量项做快速最小化以修正碰撞。
- 提升多链组装的全局搜索：多起点采样 + 局部 L-BFGS-B；可选 PSO/GA 进行更全局的刚体参数探索。
- 增设结构质量评估：Ramachandran 合格率、原子接触能（MJ 势与范德华项）、TM-score 代理分数；在结果中附加质量报告。

# 代码定位与改动点
- `modules/igpu_predictor.py:365` 主优化入口 `optimize_from_ss_gpu`：添加“多起点采样”和“质量报告生成”集成；保留现有 Adam→LBFGS 两阶段。
- `modules/igpu_predictor.py:948` `write_pdb`：在输出 PDB 附加 `REMARK` 质量分数；侧链打包时传入局部环境以做动态选择。
- `modules/sidechain_builder.py:50` `pack_sidechain`：改为“基于环境的多候选评分”，支持 Dunbrack 2010 全库；评分包含冲突、LJ 近程、氢键/芳香简化项。
- `modules/sidechain_builder.py:146` `build_sidechain`：保持拓扑生成；允许以 χ 角数组驱动，便于后续局部优化。
- 新增 `modules/quality.py`：统一质量指标计算（Ramachandran 合格率、接触能、TM-score 代理）。
- 新增 `modules/global_search.py`：全局采样与可选 PSO/GA 的简单实现，专注于多链刚体参数。

# 侧链构建精细化
- **Dunbrack 2010 集成**：
  - 引入精简版库文件（JSON/NPY），包含每种氨基酸的主峰 χ 角集合与概率；加载到 `ROTAMER_LIBRARY`。
  - 若库缺失则回退到当前近似表（确保鲁棒性）。
- **动态构象选择**（更新 `pack_sidechain`）：
  - 对每个候选 rotamer：构建侧链坐标并与局部环境原子集合做评分。
  - 评分包含：
    - 冲突计数（阈值 1.5Å）与重原子近程惩罚（如 <2.6Å）。
    - 简化 LJ（σ≈4.0、ε≈0.05，与当前 `igpu_predictor` 保持一致）。
    - 潜在氢键几何权重与芳香/阳离子-π近似（沿用现有指标的简版）。
  - 取总分最低的 rotamer；若并列，按库概率择优。
- **侧链局部优化**：
  - 针对出现高冲突的残基，固定骨架，优化 χ 角（1–3 次 LBFGS/Adam，小步长）。
  - 能量项沿用上面的简化物理项；每残基独立或小邻域批次优化，控制耗时。

# 全局搜索能力提升
- **多样化起点采样**（集成到 `optimize_from_ss_gpu`）：
  - 对 `rb_params`（多链刚体平移/旋转）在“初始温度”下随机生成 K 个起点（例如 K=8–16）。
  - 以少量 Adam 步收敛后，分别用 L-BFGS-B 精修，保留最优。
- **可选模拟退火**：
  - 在起点生成阶段对 `rb_params` 执行温度逐步下降与随机扰动，提升逃逸局部极值能力。
- **PSO/GA 全局优化（可选模块）**：
  - `global_search.py` 实现轻量 PSO：粒子状态为各链的 `t(3)+r(3)`，适配现有 `calc_loss` 作为适应度，迭代 30–50 代。
  - 备选 GA：简单交叉（平均/拼接）+ 变异（高斯噪声），种群规模小以控时。
  - 默认关闭，通过参数启用，避免常规任务时间膨胀。

# 结构质量评估模块
- **Ramachandran 合格率**：
  - 新增 `quality.rama_pass_rate(phi,psi,aa_seq)`：统计落入允许区域比例（General/GLY/PRO 分区）。
- **原子接触能**：
  - `quality.contact_energy(CA,CB,seq)`：结合 MJ 矩阵与接触权重、近程 LJ，输出接触能统计。
- **TM-score 代理**：
  - 无参考结构时，给出“全局紧致度/接触网络一致性”的归一化分数，作为 TM-score 的简化代理（明确为估计值）。
- **输出集成**：
  - 在 `optimize_from_ss_gpu` 完成后调用质量模块，生成 `results.json`（与 PDB 同目录），并向 PDB 写 `REMARK` 行。

# 集成要点与数据流
- 在 `write_pdb` 中维护一个邻域环境列表（滚动加入已写入原子；并从下一残基骨架预测补充近邻），把该列表传入 `pack_sidechain` 以做本地判定。
- 在 `optimize_from_ss_gpu` 的“多起点”阶段缓存每个起点的最优损失与质量分数，最终选择 `损失+质量` 综合最优；提供可选权重。

# 性能与风险控制
- 保持默认路径与现有性能：
  - 关闭 PSO/GA 时，耗时增幅主要来自多起点（K 可配）与少量侧链优化（仅在高冲突残基触发）。
- 明确“TM-score 代理”为启发式指标，不等同真实 TM-score。
- Dunbrack 2010 库体积控制在可接受范围；若不可用则平滑回退。

# 交付物
- 新文件：`modules/quality.py`、`modules/global_search.py`、`data/dunbrack2010.json`（或 `.npy`）。
- 修改：`modules/igpu_predictor.py`、`modules/sidechain_builder.py`、必要的初始化/参数开关。
- 输出：PDB + `results.json`（含各项分数与参数），PDB `REMARK` 附带质量摘要。

# 配置参数（示例缺省值）
- `global_search.num_starts=8`、`global_search.enable_pso=False`、`quality.enable=True`、`sidechain.optimize_clashy=True`、`dunbrack.path=data/dunbrack2010.json`。

请确认以上方案，我将按此在代码中落地实现、并进行验证。