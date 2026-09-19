# Reward Feasibility 001 实验日志与结论

## 结论先行

本轮工程链路已完整跑通，但当前离散 PASS/FAIL reward **未通过 Direct GRPO 的
组内方差门槛**。因此现在不应直接启动正式 GRPO。

- 42/42 独立 calibration prompts；
- 168/168 Qwen3-4B rollouts；
- 168/168 DeepSeek V4 Flash 有效评分；
- JSON/协议成功率 100%；
- 只有 50% 的四样本 group 出现至少 0.2 reward range，低于预注册的 70%；
- 30.95% 的 group 四个回答完全同分；
- mean group standard deviation 为 0.0869；
- 4/168 rollout 截断，截断率 2.38%。

当前决策是 `No-Go Direct GRPO`。这不是训练失败，而是训练前检测发现 reward
过于离散、部分任务存在 ceiling/floor ties；若直接训练，大量 group 的 advantage
会为零，浪费 rollout 和 Judge 成本。

## 实验配置

| 项目 | 配置 |
|---|---|
| Policy | Qwen/Qwen3-4B-Instruct-2507 |
| Prompt 数 | 14 Skills × 3 = 42 |
| 每题 rollout | 4 seeds |
| 总 rollout | 168 |
| Sampling | temperature 0.7, top-p 0.9 |
| Completion cap | 4096 |
| Judge | deepseek-v4-flash-0731 |
| Judge protocol | BenchFlow 0.6.7 PASS/FAIL top-level score |
| Judge budget | 2048→4096→8192 adaptive |

所有 prompts 使用 `judge_calibration` split，未使用官方 42 题。与官方题的最高
字符 n-gram TF-IDF 相似度为 0.4882，低于 0.72 自动拦截阈值。自动相似度检查
不能代替学科专家审查。

## 时间日志

| 阶段 | UTC | 耗时 |
|---|---|---:|
| 启动 | 2026-09-16 07:38:28 | — |
| Prompt pool 完成 | 08:15:05 | 36.6 min |
| 168 rollouts 完成 | 11:06:16 | 171.2 min |
| Judge pass 1 完成 | 12:07:17 | 61.0 min |
| 缺失项补跑并完成 | 12:10:04 | 2.8 min |
| 总计 | 07:38:28–12:10:04 | 4 h 31 min 36 s |

后续同规模实验应按约 5 小时估计，并预留 6 小时安全窗口。

## 逐 Skill 信号概览

| Skill | Mean reward | 非平凡 group 比例 |
|---|---:|---:|
| adaptive-hint-sequence-designer | 0.778 | 0.667 |
| backwards-design-unit-planner | 0.933 | 0.333 |
| differentiation-adapter | 0.400 | 0.667 |
| emergent-project-design-scaffold | 0.967 | 0.000 |
| hinge-question-designer | 0.600 | 0.667 |
| lesson-builder | 0.367 | 0.333 |
| motivation-diagnostic-task-redesign | 0.150 | 0.667 |
| project-brief-designer | 0.650 | 0.333 |
| retrieval-practice-generator | 0.600 | 0.667 |
| ruler-emotional-literacy-sequence | 0.783 | 0.333 |
| self-efficacy-builder-sequence | 0.819 | 0.333 |
| self-explanation-prompt-designer | 0.717 | 1.000 |
| socratic-questioning-sequence-generator | 0.467 | 0.333 |
| spaced-practice-scheduler | 0.600 | 0.667 |

`emergent-project-design-scaffold` 接近 ceiling 且三个 group 均无非平凡方差；
`motivation-diagnostic-task-redesign` 接近 floor。两类任务都不适合作为当前形态的
GRPO learning signal，需要提高任务难度分辨率或使用更密集的 criterion reward。

## 下一步实验

在启动 GRPO 前先做 Reward Feasibility 002：

1. 保持 DS V4 Flash 为 Judge，但训练 reward 改为 criterion-level continuous score；
2. Paper PASS/FAIL score 继续作为独立 audit 指标，不直接替代 dense training reward；
3. 对 ceiling/floor Skill 重写或加难 prompts；
4. temperature 从 0.7 提升到 0.9 做小规模多样性对照；
5. 对相同 response 重复评分，测量 Judge repeatability；
6. 完成学科审查、人工 pairwise agreement 和长度偏差审计。

只有 dense reward 的非平凡 group 比例达到 70%，且 Judge/人工门槛通过后，才启动
10-step GRPO smoke。当前 calibration prompts 不进入训练集。

## 产物

- 原始日志：`logs/reward_feasibility_001.log`
- Prompts：`artifacts/reward_feasibility_001/prompts.jsonl`
- Prompt audit：`artifacts/reward_feasibility_001/prompt_audit.json`
- Rollouts：`artifacts/reward_feasibility_001/rollouts.jsonl`
- Scores：`artifacts/reward_feasibility_001/scores.jsonl`
- Overall：`artifacts/reward_feasibility_001/summary/overall.json`
- Groups：`artifacts/reward_feasibility_001/summary/groups.csv`
- Skills：`artifacts/reward_feasibility_001/summary/skills.csv`
