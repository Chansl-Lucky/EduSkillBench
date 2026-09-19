# Qwen3-4B Verifier Calibration 001

## 结论

本地 Qwen3-4B 能稳定输出结构化 rubric 判断，但不适合作为当前 GRPO 的单一训练
reward。主要原因不是运行失败，而是严重的评分天花板和组内同分塌缩。

## 数据与结果

- 输入：独立 `judge_calibration` split，42 prompts × 4 frozen rollouts；
- 有效：166/168（98.81%）；
- 平均 reward：0.9613；
- 满分率：80.12%；
- 完整四采样组：40；
- `range >= 0.2` 非平凡组比例：2.5%（预注册目标 70%）；
- 全同分组比例：77.5%；
- 平均组内标准差：0.0172。

与冻结 DS V4 Flash 的 120 个共享有效 cell 比较：

- Qwen 均值：0.9642；DS 均值：0.7873；
- Qwen 相对 DS 平均偏高：0.1769；
- Pearson：0.2960；Spearman：0.3114；
- Qwen 在 74.17% 的共享 cell 上评分更高；
- Qwen 共享 cell 满分率 81.67%，DS 为 18.33%。

## 解释

Qwen verifier 并非完全常数：可产生 0.49、0.66、0.825 等低分。但绝大多数答案
被判为 1.0，导致同一 prompt 的四个 rollout 无法形成 GRPO 所需的相对优势。
因此它可以用于验证结论对 Judge 选择是否敏感，但不能直接替代 DS reward。

## 后续用途

1. 对已经冻结的 84 条官方 Skill baseline 做诊断性复评；
2. 比较 DS Judge 与 Qwen Judge 下的 No-Skill、With-Skill 和 lift；
3. 该结果单列为 Judge sensitivity analysis，不替换 DS 主结果；
4. 若要将 Qwen 用于训练，需要重新设计更严格的 verifier prompt，并从头重新校准。
