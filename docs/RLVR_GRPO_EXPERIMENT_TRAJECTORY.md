# EduSkillBench Rubric-RL / GRPO 实验轨迹

> 状态：环境、基线生成与首轮 reward feasibility pilot 已完成；reward gate 尚未通过，未开始训练
> 首次记录：2026-09-14
> 仓库固定提交：`98cb2977f068600d0ae66747b3521597a7a38dc0`

## 0. 先说结论：给老师汇报的重点

### 一句话结论

本项目当前最有价值、归因最干净的研究问题，不是“再做一个教育垂域大模型”，而是：

> **EduSkillBench 的 task-specific rubric 能否成为有效的强化学习信号，使一个可训练的小模型在完全隔离的教育任务上获得可验证、可迁移的能力提升？**

我们采用“先验证 reward，再决定训练路线”的四步实验：

1. 先测小模型原始能力，以及 Skill 在推理阶段能补多少能力；
2. 用独立训练题和多次 rollout 验证 Rubric Judge 是否稳定、有区分度、与人工排序一致；
3. reward 可用时直接做 GRPO，形成最干净的 `Base → Direct GRPO` 因果对照；
4. 只有出现探索不足或格式崩坏时，才用少量高质量示范做 cold-start SFT，再接 GRPO。

### 为什么这条路线值得做

- EduSkillBench 已经拥有 14 个单轮 Skill、42 个正式测试任务和逐题 rubric，但现有实验只证明 Skill prompting 的描述性增益，没有证明 rubric 具有训练效用。
- 直接比较 `Base` 与 `Direct GRPO`，只改变“是否接受 rubric-guided RL”一个主要变量，研究归因最清楚。
- 教育开放题没有数学 exact match 或代码单测式 hard verifier，因此最大风险不是训练框架，而是 LLM Judge 的噪声、偏差和 reward hacking。
- 不预设必须 SFT。SFT 只在实验确认 cold-start/exploration failure 后加入，避免把 SFT 和 RL 的收益混在一起。

### 术语边界

本项目可以放在 RLVR 的大方向下讨论，但主奖励来自 LLM 对开放式教育 rubric 的判断，并非完全确定性的 verifiable reward。方法名称应优先写成：

- `rubric-guided RL`
- `verifier-guided GRPO`
- `RLAIF with structured educational rubrics`

实现上采用“确定性检查 + 冻结 LLM Judge”的混合 verifier，避免把主观 Judge 包装成完全客观的验证器。

### 预期论文级贡献

1. 验证 EduSkillBench rubric 不只能做评价，还能提供 learning signal；
2. 比较 Direct GRPO 与 cold-start SFT→GRPO，回答开放教育任务是否需要示范启动；
3. 通过独立 Judge、人工盲评和 held-out 任务，测量并约束 reward hacking；
4. 分析 RL 后 Skill prompting 是否仍有剩余增益，区分“写入权重的程序性能力”和“推理时 Skill 增强”。

## 1. 当前事实与已完成工作

### 1.1 上游项目

- 主仓库：<https://github.com/Airlivy/EduSkillBench>
- 本地路径：`/home/gpuuser/csl/EduSkillBench`
- 固定提交：`98cb2977f068600d0ae66747b3521597a7a38dc0`
- 数据：42 个单轮任务、14 个单轮 Skills；另有 12 个多轮任务，暂不进入第一阶段训练。
- 官方评测栈：OpenCode + BenchFlow 0.6.7 + Docker。
- 本轮不修改官方 42 题，训练和 checkpoint 选择不得使用它们。

### 1.2 现有结果的局限

- 官方结果主要是单次运行点估计；
- 每个 Skill 只有 3 个单轮任务；
- task 与 rubric 明确围绕 matched Skill 编写，可能放大 Skill adherence 的收益；
- 既有不同模型行使用同族 Judge，不能直接做严格的跨模型因果比较；
- 42 题样本过小，最终结果必须报告配对重采样/置信区间和逐 Skill 异质性。

这些不是项目缺陷清单，而是本轮 RL 实验应主动解决的研究机会。

### 1.3 计算环境快照

2026-09-14 检测结果：

- 2 × NVIDIA GeForce RTX 4090，每张报告约 48 GB 可见显存；
- GPU 0 当时仅约 3.5 GB 空闲，GPU 1 约 48.6 GB 空闲；
- 系统内存约 31 GiB，Swap 8 GiB；
- `/home` 可用磁盘约 844 GB；
- Python 3.12.3 已安装；
- 仓库独立 `.venv` 已建立，`uv`、PyTorch 2.6.0+cu124、TRL 0.29.1、
  Transformers 4.57.6、PEFT 0.20.0、BenchFlow 0.6.7 均已安装并验证；
- Qwen3-4B 权重已完整下载并校验；
- 当前没有 Docker，因此正式 BenchFlow sandbox 评测仍不可运行。

因此当前可直接做单卡 4B LoRA/QLoRA pilot。要做双卡训练或独立 vLLM
rollout server，需要先释放 GPU 0；要复现原 BenchFlow 流程，还需安装 Docker。

## 2. TokenPlan API 联调记录

### 2.1 节点与协议

- OpenAI-compatible Base URL：`https://discovery-api.intern-ai.org.cn/v1`
- 模型列表：`GET /v1/models`
- Chat Completions：`POST /v1/chat/completions`
- API Key 只通过 `TOKENPLAN_API_KEY` 环境变量注入，不写入仓库、文档或日志。

### 2.2 模型列表测试

`GET /v1/models`：

- HTTP 200；
- 延迟约 0.16 秒；
- 首次返回 5 个当时可见模型：
  - `deepseek-v4-flash-0731`
  - `Agents-A1`
  - `Intern-S2-Preview-397B`
  - `glm-5.2`
  - `Atria-Dawn-Preview`

### 2.3 Chat Completions 最小测试

测试提示要求只返回 `OK`。以下延迟只是单请求连通性快照，不代表正式吞吐排名。

| 模型 | HTTP | 首次预算 | 结果 | 观察 |
|---|---:|---:|---|---|
| deepseek-v4-flash-0731 | 200 | 16 | `OK`，约 1.44 s | 正常；usage 提供 reasoning token 明细 |
| Agents-A1 | 200 | 16/128 | 16 时截断；128 时 `OK`，约 2.24 s | 绝大部分预算先用于 reasoning |
| Intern-S2-Preview-397B | 200 | 16/128/512 | 512 时 `OK`，约 4.33 s | 128 仍可能全部消耗在 reasoning，Judge 必须留足预算 |
| glm-5.2 | 200 | 16/128 | 128 时 `OK`，约 0.90 s | 16 时正文为空；128 正常 |
| Atria-Dawn-Preview | 200 | 16 | `OK`，约 0.39 s | 最小请求正常 |

工程结论：五个模型节点均可用，但不能用“正文为空”简单判定请求失败。必须同时检查 HTTP 状态、`finish_reason`、`reasoning_content`、`usage` 与正文；Judge 的 `max_tokens` 应按 rubric 输出和隐藏 reasoning 一起预算。

### 2.4 尚不能从 smoke test 得出的结论

- 不能仅凭最小请求延迟选择 Judge；
- 不能推断哪个模型最符合人工教育评价；
- 不能用一次返回判断稳定性；
- 正式选择必须经过同一 calibration set 的排序一致性、重复性、JSON 合规率和成本测试。

### 2.5 节点漂移记录

2026-09-14 后续联调时，`GET /v1/models` 的返回从最初 5 个节点变化为
10 个节点；`glm-5.2` 与 `Intern-S2-Preview-397B` 不再可用，出现了
`glm-5.3`、`intern-s2`、`qwen3.8-27b`、`deepseek-v4-pro-0813`、
`minimax-m3`、`kimi-k2.6` 等新 ID。旧 `glm-5.2` 请求返回 HTTP 404，
错误码为 `model_not_available`。

工程要求：每次正式运行前调用 `/v1/models` 校验 Judge ID，并在 manifest
中保存当次可见模型列表；不能假设 provider alias 长期稳定。

## 3. 四步实验与 Go/No-Go 门槛

## Step 1：Small-model baseline

### 目标

测量小模型初始能力、Skill prompting 增益和失败类型。

### 实验单元

在官方 42 题上运行：

1. `Base / No Skill`
2. `Base / With Skill`

固定 system prompt、chat template、生成预算和采样参数。保存 response、seed、token 数、截断状态、格式错误与延迟。

### 主要输出

- 总 rubric reward；
- 14 个 Skill 的分项成绩；
- rubric criterion 分布；
- 格式合规率、空答/截断率；
- Skill gain：`R(Base+Skill) - R(Base)`。

官方 42 题只用于最初 baseline 与最终锁定模型的评测，不能用于选择训练 checkpoint。

## Step 2：Reward feasibility test

### 数据

先构造 140–280 条与 benchmark 同能力分布、但题面与主题隔离的训练/开发 prompts：

- 14 Skills × 每 Skill 10–20 条；
- 每条只需 prompt、context 和 rubric，不需要人工 gold response；
- 按主题/知识点/情境分组切分，不能只做字符串去重；
- Train / Dev / Judge calibration 相互隔离。

### Rollout

- smoke 阶段每题 4 个 response；
- feasibility 正式统计每题 8 个 response；
- 统一保存模型、版本、seed 和 decoding 配置。

### 混合 reward

建议：

\[
R = w_hR_{hard} + \sum_k w_kR_{rubric,k}
    - \lambda_fP_{failure} - \lambda_lP_{extreme\ length}
\]

- `R_hard`：必需字段、结构可解析、选项数量等确定性约束；
- `R_rubric,k`：冻结 LLM Judge 给出的分项得分；
- failure penalty：空答、截断、拒答、复制题目；
- 只惩罚极端长度，不鼓励模型为了拿分机械缩短。

### Judge 校准

人工盲标 60–100 个 response pairs，覆盖 14 Skills 和好/中/差答案。建议第一版 Go 条件：

- Judge—人工 pairwise agreement ≥ 75%；
- Spearman/Kendall 等级相关 ≥ 0.60；
- 重复评分排序一致率 ≥ 85%；
- 至少 70% rollout groups 存在非平凡分差；
- JSON 合规率 ≥ 99%；
- 没有显著的长度偏好、格式关键词刷分或 Skill 文本复述偏好。

这些阈值是预注册用的工程门槛，不宣称为通用统计定律。

### Judge firewall

- 训练 Judge：冻结模型、prompt、rubric 和解析器；
- Audit Judge：使用另一模型，不向 policy 回传分数；
- 人工盲评：复核 Judge 分歧最大和提升最大的样本；
- 最终评价不能只复用训练 Judge。

## Step 3：Direct GRPO

只有 Step 2 通过才开始。

### 对照

- `Base`
- `Base → Direct GRPO`
- 两者分别做 No-Skill / With-Skill 评测

### 训练监控

- 训练 reward 与 audit reward；
- rollout group 内方差和 ties 比例；
- KL、clip ratio、梯度范数；
- 输出长度和格式合规率；
- 逐 Skill 学习曲线与负迁移；
- API 失败率、429、超时和缓存命中率。

### 止损条件

以下任一持续出现即暂停并审计：

- 训练 Judge 分数上升而 Audit Judge/人工分数不升；
- 输出长度无控制增长；
- 大量复制 rubric 或堆砌教学术语；
- 组内 reward 塌缩为常数；
- KL 激增或通用指令能力明显退化；
- API 错误被误当成 0 reward；
- 少量 Skill 上升但其他 Skill 大幅下降。

## Step 4：条件式 cold-start SFT → GRPO

仅在以下情况触发：rollout 普遍低质、组内无方差、格式失败严重，或 Direct GRPO 在独立 dev 上无稳定提升。

流程：

1. Teacher + Skill 生成 300–1000 条候选高质量 response；
2. 训练 Judge 初筛，Audit Judge 复核，人工抽检；
3. 训练 `SFT-only` 并先独立评测；
4. 从同一 SFT checkpoint 继续 GRPO；
5. 比较 Base / SFT / Direct GRPO / SFT→GRPO。

必须保留 SFT-only 对照，否则不能判断最终收益来自 imitation 还是 reward optimization。

## 4. 第一阶段模型与框架决策

### 4.1 Policy 模型原则

- Policy 必须是本地有权重、可训练的小模型；TokenPlan API 模型只担任 Teacher/Judge/Audit Judge。
- 首轮使用 instruction-tuned 3B 级模型做端到端 pilot；在数据、reward 和恢复机制跑通后再评估 7B。
- 先 LoRA/QLoRA，避免全参数训练把工程成本置于研究问题之前。
- 具体 checkpoint 要在模型许可、chat template、中文/英文教育任务表现和本地可下载性确认后冻结。

### 4.2 推荐主框架：TRL + PEFT

第一阶段选 Hugging Face TRL 的 `GRPOTrainer`：

- 原生支持自定义 reward callable；
- 支持异步 reward 函数，适合并发调用远程 TokenPlan Judge；
- 可组合多个 reward，并设置权重；
- 与 Transformers、Datasets、PEFT、Accelerate/DeepSpeed 兼容；
- 支持 vLLM rollout；
- 对 3B/7B、单机 LoRA pilot 的代码量和调试成本最低。

建议初始拓扑：

```text
Independent prompt dataset
          ↓
TRL GRPOTrainer + PEFT policy
          ↓ rollouts
Reward Gateway
  ├─ deterministic checks
  ├─ TokenPlan async rubric judge
  ├─ schema validation
  └─ cache / retry / audit log
          ↓ scalar + criterion rewards
GRPO update / checkpoints
          ↓
Independent dev evaluation
          ↓
BenchFlow official-42 final evaluation
```

当只有 GPU 1 空闲时，先让 TRL 完成单卡 3B pilot；GPU 0 释放后，再考虑 GPU 0 训练、GPU 1 vLLM rollout server，或双卡 DeepSpeed。

### 4.3 为什么现在不把 veRL 作为首选

veRL 适合更大的 actor/rollout/reward 资源池和多机扩展，但当前最大的未知是 Judge 是否可靠，而不是 GPU 调度。过早引入 Ray/Hydra、分布式 worker 和更复杂的自定义 reward worker，会增加定位成本。等数据和 reward contract 固定后，若 TRL 的 rollout 吞吐成为明确瓶颈，再做迁移更合理。

### 4.4 第二阶段备选

| 框架 | 适用时机 | 当前判断 |
|---|---|---|
| OpenRLHF | 需要 HTTP remote reward、多 GPU hybrid engine 或多轮 agent 训练 | 强备选；reward gateway 固定后评估 |
| AReaL | Judge 延迟高，需要 rollout 与训练异步重叠，或未来多节点扩展 | 有吸引力，但第一版工程成本较高 |
| veRL | actor/rollout/reward 分池、大模型多节点训练 | 规模扩大后再迁移 |
| BenchFlow | 复现官方 Skill benchmark 与最终隔离评测 | 保留为 evaluator，不作为 RL trainer |

### 4.5 工程接口必须框架无关

为了未来可从 TRL 迁移，以下数据契约单独实现：

- `PromptRecord`：task_id、skill_id、split、messages、rubric、provenance；
- `RolloutRecord`：policy version、seed、completion、token/finish metadata；
- `JudgeRecord`：judge model、prompt version、criterion scores、raw response hash；
- `RewardRecord`：hard checks、rubric vector、penalties、final reward；
- `ExperimentManifest`：代码提交、数据版本、模型版本、超参数和环境信息。

训练框架只消费标准化 prompt 和 reward，不直接耦合 TokenPlan 响应格式。

## 5. 下一阶段执行清单

### P0：环境与复现底座

- 安装 `uv`，创建锁定的 Python 环境；
- 检查 CUDA/PyTorch 与实际 48 GB 显存映射；
- 安装最小 TRL/Transformers/PEFT/Datasets 栈；
- Docker 与 BenchFlow 单独安装，不阻塞 reward feasibility；
- 增加实验 manifest、日志脱敏和 checkpoint 恢复测试。

### P1：数据与 Judge feasibility

- 冻结首个 3B policy checkpoint；
- 为 14 Skills 构造首批 140 条 train/dev/calibration prompts；
- 完成近重复检测和 provenance 记录；
- 每题 rollout ×4，随后对关键样本扩展到 ×8；
- 比较至少两个 TokenPlan Judge；
- 产出 histogram、group variance、人工一致性和失败案例报告。

### P2：训练

- 100-step smoke GRPO；
- 验证断点续训、缓存、超时和无效 reward 隔离；
- Go 后再启动正式 Direct GRPO；
- 用 dev 选 checkpoint，最后一次性运行官方 42 题。

### P3：条件分支

- Direct GRPO 成功：补重复种子、SFT baseline 和独立人工评测；
- Direct GRPO cold-start 失败：构造 300–1000 条 cold-start demonstrations，再做 SFT→GRPO。

## 6. 实验主表

| Policy | Post-training | No Skill | With Skill | 是否必做 |
|---|---|---:|---:|---|
| Small Base | none | R1 | R2 | 是 |
| Small Direct-RL | rubric GRPO | R3 | R4 | 是，Step 2 通过后 |
| Small SFT | high-reward imitation | R5 | R6 | 论文归因建议补充 |
| Small SFT-RL | cold-start + GRPO | R7 | R8 | 仅 cold-start 失败时 |

关键量：

- `R3 − R1`：Rubric RL gain；
- `R2 − R1`：Base Skill gain；
- `R4 − R3`：RL 后 residual Skill gain；
- `R7 − R5`：cold-start 后的额外 GRPO gain。

## 7. 风险登记

| 风险 | 影响 | 控制措施 |
|---|---|---|
| Test leakage | 结果失效 | 官方 42 隔离；主题/模板级去重 |
| Same-judge overfitting | 虚假提升 | 训练 Judge、Audit Judge、人工盲评分离 |
| Reward hacking | 学会讨好 rubric | adversarial probes、长度控制、criterion 日志 |
| Judge nondeterminism | advantage 噪声 | temperature 0、重复评分、缓存和稳定性门槛 |
| API failure as low reward | 错误梯度 | 无效请求标记为 missing，整组重试或丢弃 |
| Reasoning token 截断 | 空正文/JSON 失败 | 足够输出预算，检查 finish_reason 与 reasoning 字段 |
| Small official test | 不确定性大 | 配对 bootstrap、逐 Skill 报告、人工 audit |
| SFT/RL 混杂 | 无法归因 | Direct GRPO、SFT-only、SFT→GRPO 分开 |
| API key 泄漏 | 额度与实验污染 | 环境变量、日志脱敏、正式实验前轮换 key |

## 8. 决策日志

### 2026-09-14

- 读取共享讨论并确认四步路线；
- 克隆 EduSkillBench 主仓库并固定提交；
- 审计数据、评测脚本和现有环境；
- 确认 TokenPlan OpenAI-compatible endpoint；
- `GET /v1/models` 与五个 Chat Completions 节点联调成功；
- 发现 reasoning 模型需要显著大于 16 tokens 的输出预算；
- 第一阶段框架决定为 TRL + PEFT，自定义异步 rubric reward；
- BenchFlow 保留为最终官方评测层；
- 下一决策点：完成 Judge feasibility 后，Go Direct GRPO 或进入 cold-start 分支。
- 建立仓库内独立 Python 3.12 环境定义；训练依赖与原 BenchFlow 0.6.7
  评测依赖拆为 `train`、`eval` extras；Torch 固定使用 CUDA 12.4 wheel。
- vLLM 暂不与训练环境混装；先完成单卡 TRL pilot，再建立独立 rollout
  serving 环境，避免 Torch/CUDA 依赖被隐式替换。
- `.venv` 已通过 `uv sync --extra train --extra dev --extra eval` 完成安装；
  `torch.cuda.is_available()`、GPU 1 FP16 矩阵运算、TRL/GRPOTrainer 导入、
  BenchFlow 0.6.7 导入均通过。当前 Docker 尚未安装，因此官方 BenchFlow
  运行仍待补齐主机依赖。
- 真实数据校验发现 rubric criterion 数量并非固定 5 项：36 题为 5 项、
  3 题为 6 项、2 题为 7 项、1 题为 9 项；代码已按逐题 rubric 动态处理，
  不再把“固定五项”写成错误不变量。
- 新增 `code/rlvr/data.py`、`prompt_pool.py`、`judge.py` 和
  `run_local_baseline.py`；现有 10 个数据/Judge 单元测试，`pytest` 与
  `ruff` 均通过。首个 Qwen3-4B policy 权重已下载并校验到仓库外的
  `/home/gpuuser/csl/models/Qwen3-4B-Instruct-2507`。

## 9. Pilot-001：Hinge Question 三难度端到端实验

### 配置

- Policy：`Qwen/Qwen3-4B-Instruct-2507`，revision `cdbee75f...`；
- 解码：temperature 0，seed 20260914，max new tokens 2048；
- 任务：`hinge-question-designer` 的 easy / medium / hard 三题；
- 条件：No-Skill 与完整 `SKILL.md` system-context injection；
- 训练 Judge 候选：`deepseek-v4-pro-0813`；
- Audit Judge 候选：`Atria-Dawn-Preview`；
- 混合 reward 对截断回答施加确定性惩罚，因此同时保留原始 rubric reward。

### 生成行为

| Task | No-Skill prompt/output | Skill prompt/output | 截断 |
|---|---:|---:|---|
| easy | 203 / 1107 | 4964 / 1580 | 均否 |
| medium | 200 / 1231 | 4961 / 2048 | Skill 截断 |
| hard | 199 / 1333 | 4960 / 2048 | Skill 截断 |

Skill 注入使输入增加约 4760 tokens，并显著增加输出长度。medium/hard 的
Skill 回答达到 2048-token 上限，因此正式协议需要同时控制质量和 token 成本。

### 两位 Judge 的混合 reward

| Task | DeepSeek No | DeepSeek Skill | Δ | Atria No | Atria Skill | Δ |
|---|---:|---:|---:|---:|---:|---:|
| easy | 0.949 | 0.983 | +0.034 | 0.652 | 0.949 | +0.298 |
| medium | 0.456 | 0.298 | -0.158 | 0.558 | 0.332 | -0.226 |
| hard | 0.847 | 0.332 | -0.515 | 0.754 | 0.578 | -0.175 |
| mean | 0.751 | 0.538 | -0.213 | 0.654 | 0.620 | -0.035 |

### Rubric-only 结果与解释

- DeepSeek rubric mean：No-Skill 0.707，Skill 0.593，Δ -0.113；
- Atria rubric mean：No-Skill 0.593，Skill 0.690，Δ +0.097；
- 两位 Judge 对 easy 的 Skill 改进、medium 的弱/负改进基本同向；
- 对 hard 题，DeepSeek 给 Skill 显著更低分，Atria 给两者相同 rubric 分；
- 两位 Judge 对三题总体 Skill 方向发生反转，证明当前 reward 尚未通过
  feasibility gate，不能直接启动 GRPO。

Pilot-001 的关键结论不是“Skill 无效”，而是：

1. Skill 注入存在显著长度成本和截断风险；
2. 同一回答在不同 Judge 下分数尺度和总体方向会变化；
3. reward 合约、长度策略和人工校准必须在训练前冻结；
4. 这一 Skill 在上游结果中本就出现过模型相关的负迁移，因此应作为
   reward calibration 的高价值压力测试样本。

### Judge 协议稳定性复测

同一批 6 条回答上的复测进一步暴露了 provider 与 reasoning-budget 风险：

- `deepseek-v4-pro-0813` 首轮在 2048 预算下 6/6 成功，但同配置复测为
  0/6 有效正文；这说明一次成功不能视为稳定 Judge；
- `deepseek-v4-flash-0731` 在 2048 下 0/6，`kimi-k2.6` 抽测 0/2；
- `Atria-Dawn-Preview` 在 4096 下总体可用，但仍发生过需要重试的样本；
- `minimax-m3` 固定 2048 时部分失败，个别样本需要 4096 或 8192 才能
  返回完整结构化正文。

为此 Reward Gateway 已实现以下防线：

1. 每次运行先读取 `/models`，不存在的 Judge ID 直接停止；
2. HTTP/429/5xx/网络故障重试，但失败不会转换成低 reward；
3. `finish_reason=length` 且正文为空、JSON 不完整或 rubric 缺项时，自动按
   2048→4096→8192 扩容；
4. 每条分数保存实际 Judge ID、预算、finish reason 与 usage；
5. 缺失 criterion 不允许静默按 0 分处理。

### Minimax 自适应预算复验（6 条）

| Task | No-Skill mixed/rubric | Skill mixed/rubric | Skill 实际 Judge 预算 |
|---|---:|---:|---:|
| easy | 0.813 / 0.780 | 0.932 / 0.920 | 2048 |
| medium | 0.541 / 0.460 | 0.307 / 0.390 | 4096 |
| hard | 0.813 / 0.780 | 0.697 / 0.850 | 4096 |
| mean | 0.722 / 0.673 | 0.645 / 0.720 | — |

- 6/6 最终通过 JSON 与 rubric schema 校验，2/6 经 2048→4096 自动扩容；
- rubric-only 的 Skill 均值提升 `+0.047`；
- 混合 reward 的 Skill 均值变化为 `-0.077`，主要来自 medium/hard 的
  policy 输出在 2048 被截断，而非 rubric 内容质量全面下降；
- 同一 Minimax 配置两轮评分也有可见数值变化，说明 temperature 0 并不等于
  后端确定性。缓存、重复评分与人工校准仍不可省略。

### 当前 Go/No-Go

**No-Go for GRPO。** 工程评分链已经可以可靠区分“远端无效响应”和“有效低分”，
但 reward 的跨 Judge 方向一致性与同 Judge 重复性尚未达到预注册门槛。下一步是
构建独立 calibration prompts、每题多 rollout，并进行人工 pairwise 盲标；不能用
这 3 道官方测试题继续调 Judge 或选 checkpoint。

## 10. Calibration-001：独立题与长度归因实验

### 数据生成与准入

- 使用 released Skill 作为能力定义，由 `Atria-Dawn-Preview` 生成 3 条新的
  `hinge-question-designer` 候选；
- 生成器强制 `judge_calibration` split、rubric 总分 100、禁止 Skill/benchmark
  名称泄漏，并与全部官方题做 exact fingerprint 与字符 3–5 gram TF-IDF 筛查；
- 三条候选对官方题的最大相似度为 0.373、0.207、0.275，均低于 0.72
  自动拦截阈值；
- 内部审计拒绝 chemistry 候选，因为“用电负性阈值定义键型”的表述可能过度
  绝对化；fractions 候选待学科复核；elapsed-time 候选只获准做 pipeline smoke；
- 自动筛查只是风险预警，不能替代学科专家审查，也不构成人工 Judge 校准。

### 独立 elapsed-time 题结果

固定 policy、seed 和 greedy decoding，只改变 Skill rollout 的最大输出预算：

| Condition | Policy output | 截断 | Minimax rubric/mixed | Atria rubric/mixed |
|---|---:|---:|---:|---:|
| No-Skill, 2048 cap | 1516 | 否 | 0.595 / 0.656 | 0.595 / 0.656 |
| Skill, 2048 cap | 2048 | 是 | 0.410 / 0.324 | 0.478 / 0.381 |
| Skill, 4096 cap | 3447 | 否 | 0.730 / 0.771 | 0.625 / 0.681 |

同一 Skill 回答解除截断后，相对 No-Skill 的 rubric 增益变为：Minimax
`+0.135`、Atria `+0.030`；混合 reward 也都转正。2048 截断版缺失后段的
diagnostic key/decision guide，导致两个 Judge 都给出低分。

这提供了当前最强的机制证据：**完整 Skill procedure 对内容有正向信号，但会诱导
更长回答；固定过短的 rollout cap 会把“执行得更完整”错误地转化为 failure penalty。**

后续协议因此必须：

1. 训练 prompt 明确要求紧凑输出，减少无效展开；
2. rollout 上限至少覆盖 pilot 的 95% 完整回答，再单独惩罚极端冗长；
3. 同时报告 rubric-only、hard check、长度与 mixed reward，避免单一标量掩盖机制；
4. GRPO group 内不能混用不同 token cap；截断样本需标记并进入 failure audit；
5. 在多个独立题和多 seed 上复现后，才能把该现象写成研究结论。

## 11. Overnight-001：DeepSeek V4 Flash 论文口径基线

2026-09-14 16:42 UTC 已通过独立 `tmux` 会话 `eduskill-dsflash` 启动后台任务，
PID 与日志分别保存在：

- `logs/overnight_dsflash_20260914.pid`
- `logs/overnight_dsflash_20260914.log`

协议固定为：官方 42 个单轮任务 × No-Skill/Skill 两条件，Qwen3-4B policy，
temperature 0.7、top-p 0.9、seed 20260914，两组统一 4096 completion 上限。
Judge 固定为 TokenPlan 当前 ID `deepseek-v4-flash-0731`。

评分 prompt 与 BenchFlow 0.6.7 released verifier 对齐：逐 rubric item 判定
PASS/FAIL，主分数为通过项比例，而不是先前 continuous criterion score；expected
output、task prompt 和全部 expected behaviors 均传给 Judge。DS Flash 的论文协议预检
得到有效的 `3/5 = 0.6`，但需要将 Judge 输出预算自动扩到 8192，其中绝大部分为
reasoning tokens。

输出包括：

- `artifacts/overnight_dsflash_20260914/rollouts.jsonl`
- `artifacts/overnight_dsflash_20260914/scores.jsonl`
- `artifacts/overnight_dsflash_20260914/summary/runs.csv`
- `artifacts/overnight_dsflash_20260914/summary/skills.csv`
- `artifacts/overnight_dsflash_20260914/summary/overall.json`

对齐论文的部分包括官方任务、配对条件、task-specific rubrics、PASS/FAIL 比例、
单次点估计和逐 Skill 宏平均。不同之处必须保留在报告中：论文使用 OpenCode +
Docker 且 policy 是 API 模型；本实验是 Qwen3-4B 本地权重与 Transformers chat
template。因此它是 RL policy baseline，不是论文 agent-harness 的精确复现。

### 完成状态与结果

2026-09-16 07:26:10 UTC 完成 84/84 个唯一 cell。JSONL 中另外保留 4 条历史
失败记录供审计；聚合器按 `(task, condition, seed, judge, protocol)` 去重，并优先
选取最终成功记录，因此失败重试不会污染均值。

| 指标 | 结果 |
|---|---:|
| No-Skill rubric reward | 0.5762 |
| With-Skill rubric reward | 0.8815 |
| Skill lift | +0.3053 |
| Normalized gain | 0.7204 |
| 正向 / 持平 / 负向 Skills | 13 / 1 / 0 |
| No-Skill / Skill 截断 | 3 / 1 |

最后一个缺失 cell 连续返回 4/5 条 item 明细，但包含合法的顶层 `score`。released
BenchFlow 0.6.7 verifier 本身直接使用该顶层 score，并不校验 item 数量；gateway
已改为与该行为一致，同时保存返回明细供审计。修改后该 cell 成功，完整汇总正常生成。

实际活跃耗时约 1 小时 42 分钟：本地生成约 1 小时 29 分钟，DS Flash 评分与
重试约 14 分钟。中间的 API 不可达和人工恢复间隔不计入活跃计算时间。相同规模的
后续单次基线建议预留 2 小时，考虑 provider 排队和重试时使用 3 小时安全窗口。

## 12. Reward-Feasibility-001：独立多样本评分实验

2026-09-16 07:38 UTC 已通过 `tmux` 会话 `eduskill-feasibility` 启动 Step 2
第一轮。实验为 14 个 Skill 各生成 3 条与官方题隔离的 calibration prompt，共
42 条；Qwen3-4B 对每题用 4 个 seed 采样，共 168 个 No-Skill rollout；随后由
`deepseek-v4-flash-0731` 按 BenchFlow PASS/FAIL 协议评分。

自动报告包括 JSON/协议成功率、每题 group reward 方差与 range、全平局比例、
截断率和逐 Skill 分布。自动门槛为有效评分率至少 99%、至少 70% group 有不小于
0.2 的 reward range。学科内容审查、相同回答重复评分、人工 pairwise agreement
和长度/风格偏差仍是人工或后续实验门槛，自动任务不会伪造这些结论。

状态与日志：

- `bash code/rlvr/check_reward_feasibility_001.sh`
- `logs/reward_feasibility_001.log`
- `artifacts/reward_feasibility_001/`

预计活跃耗时约 2.5 小时，provider 排队或重试情况下预留 4 小时。

## 13. 参考入口

- EduSkillBench：<https://github.com/Airlivy/EduSkillBench>
- EduSkillBench 预印本：<https://www.preprints.org/manuscript/202608.2214>
- TokenPlan 门户：<https://discovery.intern-ai.org.cn/token-plan/home?tabIndex=1>
- TRL GRPOTrainer：<https://huggingface.co/docs/trl/grpo_trainer>
- veRL：<https://verl.readthedocs.io/>
- OpenRLHF：<https://openrlhf.readthedocs.io/>
- AReaL：<https://areal-ai.io/AReaL/>
