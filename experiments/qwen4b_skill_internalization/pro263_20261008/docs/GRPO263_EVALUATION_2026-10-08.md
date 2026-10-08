# GRPO 263 题测试与自动整理发布

本轮只测试完成于 2026-10-07 01:42（北京时间）的 SFT246→GRPO123 模型，不重训、不重跑 Base/Base+Skill/SFT。

## 固定配置

- Qwen3-4B-Instruct-2507 + rank32/all-seven-module LoRA，最终 adapter SHA256：`bf175d09ef5394eb2f3bce41b1cc8903753383582d8c461569f17332b22fa5a5`。
- 无 Skill 输入。使用旧 263 题的 no-skill `instruction.md`、`task.toml`、Dockerfile，逐文件哈希相同。
- Docker/OpenCode/BenchFlow 使用旧 handoff 环境、教育单轮 system、600 秒任务和 idle 超时、Hermes tool parser。vLLM 服务配置除 adapter/alias 和端口外不变。
- 单个 GPU1 服务、环境并发 8；不启动三套模型服务。Pro 评分 CPU 进程容量 64，随答案完成逐个评分，不等待全部生成。
- 先验证 2 个真实测试题的模型加载、完整主会话最终答案及评分校验，不能用分数高低挑选是否继续。通过后自动执行剩余261题。
- Pro评分复用固定上游 prompt/schema/validate，最终主会话答案、原生等级和反证、每题最高档维度占比。未交付/超时不改成成功，失败计零并单独说明。
- 原三组 Pro 分数不改写；其残留 14 个 Judge 校验失败仍标明，不隐去。
- 网络路由例外：10 月 8 日本机直连 Ark 发生 ConnectTimeout，代理实测 HTTP 200 且返回同一 Pro 版本。因此本轮显式使用已测试代理，只改变 transport，不改上游 prompt、payload、schema、SSE解析或超时。若代理失效会记录失败，不伪造成绩。
- GPU 服务在生成结束或出错时退出；失败评分可以从原答案继续，不重新 roll。

## 输出与监控

运行目录：`artifacts/eval_grpo263_pro_20261008/`。

```bash
tail -f ${EDUSKILL_ROOT}/artifacts/eval_grpo263_pro_20261008/worker.log
```

`status.json` 显示已生成数、成功评分数、错误类别；`server.log` 为模型加载日志；`cells/grpo/<题号>/` 保存最终答案、父会话/消息身份、Judge原始响应和校验结果。

完成后生成 `comparison/REPORT.md`、`paper_report.json`、`paper_rows.json`：四版本主表、GRPO相对三个版本的配对差、来源簇 bootstrap 区间、失败情况和 9 个 Skill 分项。263 题包含 9 Skill、30 来源文档；不是全部 14 个训练 Skill 的完整检验。

本轮并未消除 Docker/OpenCode 与同门直接 Chat 的生成差异。不能把评分对齐解释为所有生成环境完全相同。

## GitHub发布

目标为用户自己的 `Chansl-Lucky/EduSkillBench`，不是直接改师门 `Airlivy/EduSkillBench`。

评测完成后自动打包报告、逐题分数、评分与评测核心代码、配置和说明，API Key/账号密码/模型权重/原始Docker日志不上传。独立临时clone只暂存这次实验包，不提交当前工作树的其他修改。成功记录在 `PUBLISHED.json`；认证/推送失败记录 `publish_status.json` 和 `publish.log`，不能声称已上传。

目录：`experiments/qwen4b_skill_internalization/pro263_20261008/`。
