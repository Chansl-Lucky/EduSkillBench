# Qwen3-4B Skill-GRPO 实验汇报

## 一、先说结论

本轮实验已经完整跑通“基础模型 → 注入 Skill → Skill 条件下 GRPO”三阶段闭环。在同一批
42 道官方测试题、同一 Qwen3-4B 基座、同一生成配置和同一 DS V4 Flash Judge 下：

| 实验条件 | 平均 reward | 相对前一阶段 |
|---|---:|---:|
| Qwen3-4B Base / No-Skill | 0.6435 | — |
| Qwen3-4B Base / With-Skill | 0.9063 | +0.2628 |
| Qwen3-4B GRPO / With-Skill | 0.9388 | +0.0325 |

结果说明：匹配 Skill 的上下文注入对 4B 小模型有明显帮助；在相同 Skill 条件下，GRPO
训练后的模型又取得了 3.25 个百分点的正向点估计。但 GRPO 增量的配对 bootstrap 95%
置信区间为 `[-0.0048, 0.0714]`，跨过 0，因此目前不能声称已经得到统计稳定的 RL
提升。更准确的表述是：**工程闭环与正向趋势已经验证，稳定增益仍需更多种子或更强的
训练 reward 信号确认。**

一句话汇报：**在 14 个教育 Skill、42 道隔离测试题上，Qwen3-4B 从 No-Skill 的
0.6435 提升到 With-Skill 的 0.9063，70 步 Skill-conditioned GRPO 后进一步达到
0.9388；GRPO 增量为 +0.0325，但单次实验的置信区间仍跨 0。**

## 二、研究问题与三阶段对照

本实验回答的问题是：当推理阶段始终提供相同的匹配 `SKILL.md` 时，经过 rubric-guided
GRPO 的 Qwen3-4B，是否能比未经训练的 Qwen3-4B 更有效地使用 Skill？

三个条件分别为：

1. **Base / No-Skill**：只提供教育任务，用来测量小模型原始能力；
2. **Base / With-Skill**：向同一个基础模型注入完整匹配 Skill，用来测量推理时 Skill
   augmentation 的贡献；
3. **GRPO / With-Skill**：训练和测试均提供匹配 Skill，用来测量权重更新在 Skill
   注入之上的额外贡献。

核心因果对照是第 3 组减第 2 组；第 1 组主要用于刻画原始论文所研究的 Skill lift。

## 三、冻结实验配置

| 项目 | 配置 |
|---|---|
| Policy | `Qwen/Qwen3-4B-Instruct-2507` |
| 固定 revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Skill 范围 | 全部 14 个 single-turn Skills |
| 训练/开发数据 | 70 条 train、28 条 dev，与 42 道官方测试题隔离 |
| 训练方法 | GRPO，70 optimizer steps，每步 4 个 completions |
| 参数更新 | LoRA rank 8，作用于 attention `q/k/v/o` projections |
| 训练采样 | temperature 0.9，top-p 0.9，最多 2,048 tokens |
| 正式评测 | 42 题，temperature 0.7，top-p 0.9，seed 20260914 |
| Judge | Volcengine Ark `deepseek-v4-flash-260425` |
| 评分协议 | 与 BenchFlow 对齐的逐 rubric item PASS/FAIL 比例 |
| 不确定性 | 20,000 次 paired task bootstrap |

官方 42 题没有参与训练 prompt 生成、训练 reward 或 checkpoint 选择。生成训练题时还使用
character n-gram 相似度阈值 0.72 排除与测试题过近的样本。

## 四、结果细分与解释

- 42 个任务中，GRPO 相对 Base / With-Skill：9 个提升、29 个持平、4 个下降；
- 14 个 Skill 中：6 个提升、6 个持平、2 个下降；
- Base 与 GRPO 的平均生成长度分别为 2,596 和 2,606 tokens；两组各有 1 条截断，因而
  结果不是通过显著增加回答长度获得的；
- 70 个训练 group 中有 55 个存在非零 reward range，但只有 8 个达到
  `range >= 0.2`，另有 15 个完全同分。

最后一项说明训练信号虽然可用，但组内区分度仍偏弱，这可能是 GRPO 增量较小且置信区间
跨 0 的主要原因。下一轮若要强化论文结论，优先级应是重复随机种子并改进 reward 的组内
排序能力，而不是只增加训练步数。

## 五、与原论文实验的边界

这是一组本地 weight-level 扩展实验，不是原论文 OpenCode + Docker agent trajectory
harness 的严格复现。当前实验已经对齐三组内部比较所需的任务、Skill、rubric、生成预算
和 Judge，因此可以解释同一套本地框架内的相对变化；但不能把这里的绝对分数直接用于
宣称 Qwen3-4B 优于论文中的 hosted 模型。

本轮没有加入人工评审，结论限定在冻结的自动 Judge 协议下。

## 六、仓库阅读与复现入口

- 完整英文实验说明：[`docs/QWEN3_4B_SKILL_GRPO.md`](QWEN3_4B_SKILL_GRPO.md)
- 正式运行配置：[`experiments/runs/skill_grpo_formal_001/manifest.yaml`](../experiments/runs/skill_grpo_formal_001/manifest.yaml)
- 三阶段一键流程：[`code/rlvr/run_skill_grpo_formal_001.sh`](../code/rlvr/run_skill_grpo_formal_001.sh)
- GRPO 核心实现：[`code/rlvr/train_skill_grpo.py`](../code/rlvr/train_skill_grpo.py)
- 汇总结果：[`results/qwen3_4b_skill_grpo/summary.json`](../results/qwen3_4b_skill_grpo/summary.json)
- 逐 Skill 结果：[`results/qwen3_4b_skill_grpo/skills.csv`](../results/qwen3_4b_skill_grpo/skills.csv)
- 逐任务结果：[`results/qwen3_4b_skill_grpo/tasks.csv`](../results/qwen3_4b_skill_grpo/tasks.csv)

仓库不包含真实 API Key、模型权重、私有运行日志或大体积 artifacts。环境变量模板见
`.env.example`，安装与复现命令见英文实验说明。
