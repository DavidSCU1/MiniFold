## 目标
- 替换当前基于阿里云千问(Qwen)的二级结构生成，改为：
  1) 由 FASTA 序列经 ProtParam/ProteinAnalysis 结合残基倾向启发式生成多个二级结构候选；
  2) 通过火山 Ark API 同时调用多个模型进行概率评估与投票聚合，选出“最可信”的一个候选；
- 移除 Qwen 的依赖与调用路径，仅保留 Ark 方案；

## 现状定位
- 候选生成：`modules/qwen_module.py` 的 `qwen_ss_candidates` 在 `modules/pipeline.py:62` 被调用；其结果用于后续评估与筛选；
- 评估：`modules/llm_module.py:161` 提供 `deepseek_eval_case`（Ark API），在 `modules/pipeline.py:85` 使用；

## 设计与实现
1. 候选生成（ProtParam + 残基倾向）
- 新增 `modules/ss_generator.py`：`protparam_ss_candidates(sequence, env_text, num)`
  - 使用 `Bio.SeqUtils.ProtParam.ProteinAnalysis` 的 `secondary_structure_fraction()` 获取 helix/turn/sheet 的全局比例约束；
  - 基于经典残基二级结构倾向（示例：`ALMEKQR` 偏 H，`VIGYFW` 偏 E，`PG`/带破坏螺旋的残基偏 C），结合滑窗与多阈值，生成 `num` 个多样化候选；
  - 保证：仅含 `H/E/C`；总长度严格等于序列长度；按线圈聚集处优先切分成多链（`|`），至少 2 链；
  - 输出结构与现有一致：`{"cases": [{"chains": [str,...]}], "raw": "...", "attempts": n, "lines": m}`。

2. Ark 评估与投票
- 新增 `modules/ark_module.py`：
  - `ark_eval_case(model, sequence, environment, chains, req_text=None, api_key=None)`：参照 `llm_module.deepseek_eval_case`，将 `model` 作为参数，Ark 端点 `https://ark.cn-beijing.volces.com/api/v3/chat/completions`；
  - `ark_vote_cases(models, sequence, environment, chains_list, req_text=None)`：对每个候选 `chains_list`，并行/串行调用 `models`，收集每模型的概率 `p`，聚合为 `p_avg`（均值）+ `p_med`（中位）+ 可选加权（默认等权，若环境设定权重则应用）；返回 `best_case_idx` 与详细评分；
- 模型清单（默认等权）：
  - `doubao-seed-1-6-251015`
  - `deepseek-v3-2-251201`
  - `doubao-1-5-pro-256k-250115`
  - `kimi-k2-thinking-251104`
  - `deepseek-r1-250528`
- 失败容错：单模型失败不致命；聚合时仅统计成功项；若全部失败，回退到旧的阈值流程或最终回退模式。

3. 管道集成
- 修改 `modules/pipeline.py`：
  - 删除 `from modules.qwen_module import qwen_ss_candidates`（`modules/pipeline.py:8`）；改为 `from modules.ss_generator import protparam_ss_candidates`；
  - 在 `run_pipeline` 中用 `protparam_ss_candidates` 替代 `qwen_ss_candidates`（`modules/pipeline.py:62`）；
  - 评估阶段改为：对所有候选调用 `ark_vote_cases(models, ...)`，选出 `best_case_idx`，仅保留这一案例并继续 3D 结构生成；
  - 日志与产物：
    - 写入 `*_ss_candidates.json`（保留所有生成候选）与 `*_votes.json`（各模型对每候选的评分与聚合分）；
    - 明确打印“最可信案例”与聚合分；
- 保留既有回退逻辑：当无有效候选或全部评估失败时，沿用当前 `fallback` 生成与后续流程。

4. 配置与环境
- 使用已有 `.env` 加载：`ARK_API_KEY`、`ARK_API_URL`（已有于 `modules/llm_module.py`）；
- 可选：支持 `ARK_MODELS` 逗号列表覆盖默认模型集；支持 `ARK_MODEL_WEIGHTS`（逗号权重，与模型一一对应）。

## 验证
- 生成器单元校验：对多组序列（短/中/长、富含疏水/亲水）验证输出仅含 `H/E/C`、总长度一致、多链切分合理；
- Ark 调用冒烟测试：对每模型调用一次 `ark_eval_case`，确认鉴权与返回结构正常；
- 管道端到端：用示例 FASTA 跑通，产出 `*_votes.json`、报告与 3D 文件；
- 失败场景：API Key 缺失、网络异常、模型个别失败时能正常降级。

## 代码改动清单
- 新增：`modules/ss_generator.py`、`modules/ark_module.py`
- 修改：`modules/pipeline.py`（导入与候选/评估替换，产物与日志更新）
- 保留但不再使用：`modules/qwen_module.py`（后续可移除）

## 交付与文档
- 在报告中新增“投票与聚合”段，记录每模型评分与最终聚合分；
- 在 README 增补使用 Ark 多模型的环境变量说明与示例；

请确认以上方案，收到确认后我将按此实现并进行验证。