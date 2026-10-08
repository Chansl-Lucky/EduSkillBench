# 263 题 Pro 重评分：运行与同门版本逐项核对

范围：Base、Base+Skill、SFT 各 263 题，789 单元；不评 42 题，不重新生成回答、不重训。GRPO 仍在独立进程训练，本任务不改其奖励。完成后本任务仅输出上述三个已有版本的 Pro 成绩。

注意覆盖范围：上游 305 题合计涵盖 14 Skill，但这里单独选出的 263 题实际上覆盖 9 个 Skill、30 个来源文档。Skill 分项只报告这 9 个有测试题的组，另外 5 个不能伪造分数或算作零分 Skill；不能称“263 题验证了全部 14 Skill”。

## 已对齐与未对齐

| 项目 | 本次实际设置 | 与固定上游版本的关系 |
| --- | --- | --- |
| 来源版本 | `6270c70e793ad625297dcce8074bc5cd601aa5c1` | 固定代码、文档与模型配置，非动态 main |
| 数据 | 263 个 advisory case，完整对象逐项相同 | 题目、背景、scope_note、criteria、applicable_ids、原等级/权重不改 |
| Judge 请求 | Ark Plan Chat，`deepseek-v4-pro` | 请求别名一致；本机响应解析为 `deepseek-v4-pro-ga-260813`，逐响应检查；官方历史解析版本未知 |
| 请求参数 | 关闭 thinking，10000 token，流式 Chat；不传 temperature/seed | 复用上游 `judge.call_judge` 实际组包，不自制相似 prompt |
| 超时 | headers=45、idle=45、content_idle=60、total=120 秒 | 上游真实流式进度 deadline；不是仅设 socket 120 秒 |
| Prompt | 直接调用固定上游 `source_protocol.prompt(case,answer)` | 字节逻辑相同，无新增质量标准、参考答案或自创反证规则 |
| 输出 schema | 固定上游 `source_protocol.schema` | 保留原生等级，逐项 reason/evidence/counterevidence，不做任意等级转分 |
| 证据与校验 | 上游按 160 字符分 E 片段；`validate(...require_counterevidence=True)` | 数量/ID/等级/证据范围/最高档证据/反证最低档规则均相同 |
| 评分对象 | 用户任务主会话最后一条已终止 assistant 的正文 | 不把子代理、工具结果、早期拒答、重复 ACP 文本当成最终回答 |
| 生成方式 | 复用之前 Docker/OpenCode 输出 | **不同**：同门本次正式结果直接 Chat 生成；不能声称本次所有生成细节也一致 |
| Skill 输入 | 复用旧实验已生成的 With-Skill 答案 | **未重做**：旧实验 Skill 注入/资源访问与同门直接 Chat 资源拼接的差异仍保留 |
| 单题主指标 | 适用维度中取最高原生等级的数量 / 适用维度数量 | 上游 `build_results.py` 同一公式；原始 `native.score=None` 仍保留 |
| 总表 | 三个版本各自题目等权均值，固定分母 263 | 不按维度总数或原权重做全局池化，不合入 42 题 |
| 失败处理 | 18 个原执行超时（Base 7、Skill 3、SFT 8）计零，保留源错误 | 不重 roll 直到成功、不用超时中间产物冒充完成；Judge 失败另记，可重试失败评分 |
| 重试 | 每次最多 2 次请求，共享网络/格式纠错次数；后续只恢复未成功单元 | 复用上游 validation-feedback 原则，已成功评分不再请求、也不取最高 |
| 报告中间量 | 原始等级、证据和反证，strict-pass、review flags、有效子集次要均值、失败类别、Skill 分项 | 中间量不替代主表；存在反证不代表已人工审核其正确性 |
| 配对统计 | 5000 次配对来源簇 bootstrap，30 来源，seed=20261005 | 同一上游统计方法；比较 Base→Skill、Base→SFT、Skill→SFT；条件于本次缓存结果，不是多种随机种子实验 |
| 并发 | 8→32→64 实际评分梯度压测；实际 429/传输故障才降并发 | **吞吐改动**，不是评分规则改动；无需本地 GPU；与 GRPO 共用 API Key，限流时降载 |

不能从新版与旧版分差单独推导“Pro比Flash更严格/更宽松”：本次同时更换了 Judge 和评分输入对象。旧 Flash 读取完整多代理轨迹，本次读取最终答案，这两项因素必须一起披露。

## 运行与进度

后台进程和当前 PID 见：

- `artifacts/rescore_advisory263_pro_20261007/launcher.json`
- `artifacts/rescore_advisory263_pro_20261007/worker.log`
- `artifacts/rescore_advisory263_pro_20261007/status.json`
- `artifacts/rescore_advisory263_pro_20261007/pressure_tests.jsonl`

服务器查看命令：

```bash
tail -f ${EDUSKILL_ROOT}/artifacts/rescore_advisory263_pro_20261007/worker.log
```

每个完成单元立即独立持久化。重启按绑定核对缓存，只继续未成功评分；任务完全脱离对话会话运行。最多两轮失败单元恢复，之后仍失败则带类别计零并报告，不无限阻塞。

## 验收产物

- `manifest.json`：来源、原数据/旧分数/代码 SHA256、环境差异与实际配置。
- `inputs.json` 和 `cells/<arm>/<task_id>/input.json`：冻结题面、最终答案、主会话/消息/part 身份、来源数据库路径和哈希。
- 每题 `answer.md`、`raw_*.txt`、`trace_*.json`、`response_*.json`、`failure_*.json`：可回查的原始评分证据；不保存凭据。
- 每题 `result.json`：首次校验通过结果及原生 verdict；失败单元明确标明计零由失败政策产生，不是伪造 Judge verdict。
- `paper_rows.json`：789 单元逐题主指标、失败、strict-pass、review flags。
- `paper_report.json`：三组主分数、三组配对差、来源簇区间、有效配对次要比较、263 题实际覆盖的 9 个 Skill 分项。
- `REPORT.md`：可直接汇报的简表。

6 个单测覆盖父子代理隔离、未完成主会话不回退到早期计划、多父会话拦截、适用维度完整性、非权重最高档占比与最高档证据要求。原始 Flash 文件不覆盖；完成时再次核对旧评分哈希。
