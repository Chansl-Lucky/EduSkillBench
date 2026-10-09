# 正确性敏感 Judge：审核与复用

这是新增评分协议的消融实现，不是原生评分未改动复现，也不是通用科学真值验证器。生成端 Docker/OpenCode 仍见 [native_skill](../native_skill/README.md)。本目录不重新生成任何回答、不训练模型。

先读 [变化审核说明](../../docs/JUDGE_CHANGE_REVIEW_2026-10-09.md)，再读 [实际新增提示词](../../docs/judge_correctness_20261009/prompts/) 与 [同答卷评分协议](../../docs/GLM305_CORRECTNESS_REJUDGE_2026-10-09.md)。原生 Judge 源码保留在 [upstream/judge.py](../native_skill/upstream/judge.py) 和 [source_protocol.py](../native_skill/upstream/source_protocol.py)，没有覆盖它们。

## 离线验收

从仓库根目录运行，Python 3.10+，无需 API Key、GPU、Docker 或下载模型：

```bash
python3 repro/judge_correctness/bootstrap.py
python3 -m unittest discover -s code/evaluation -p 'test_glm305_*.py'
python3 -m unittest discover -s code/evaluation -p 'test_ark_standard_resume_20261009.py'
python3 repro/judge_correctness/export_review.py
git diff -- docs/judge_correctness_20261009
```

bootstrap 只把仓库中已冻结、SHA256一致的上游文件安装到旧脚本兼容路径，及放置一条公开诊断样本。已有文件若不同则拒绝覆盖。r2/r3脚本仅保留给回归测试和审核演进；**正式执行为r4及其Ark恢复适配器**。

## 正式数据接续不是空仓库直接启动

全部610份生成答卷、各源目录、旧判定、r1缓存、r4输入/manifest以及已接受缓存属于服务器运行材料，不随本次代码提交上传。单条诊断fixture仅用于离线测试，不是全量数据集。

接续该次冻结运行需要从原服务器受控迁移以下目录并保留路径/hash绑定：

```text
artifacts/glm_native_skill_recovery_20261009_r1/
artifacts/glm305_correctness_pro_20261009_r1/
artifacts/glm305_comprehensive_pro_20261009_r4/
artifacts/sft246_grpo_20261005_r2/api_switch_20261007/upstream_scoring/
以及聚合rows.json/source_cell和manifest/protected_hashes指向的全部来源目录
```

代码对已冻结绝对路径及hash严格校验，**仅克隆Git不能复原服务器答卷或自动迁移这些绑定**。保持同样`/home/gpuuser/csl/EduSkillBench`目录布局，或者另行显式、审计式地重新绑定，不能直接替换路径来冒充原实验续跑。

具备完整运行材料之后，凭证只从环境传入，不把真实Key写入配置、命令示例、日志或Git：

```bash
# 在可信会话中设置 LLM_API_KEY，不将真实值粘贴进此文件。
export ARK_SCORING_PROXY=http://127.0.0.1:38790
python3 code/evaluation/resume_glm305_ark_standard_20261009.py --launch
tail -f artifacts/glm305_comprehensive_pro_20261009_r4/worker.log
```

同一输出目录有pipeline锁，禁止重复启动同一批评分。模型固定`deepseek-v4-pro-ga-260813`，thinking关闭，10,000输出tokens，最多64并发。每条答卷内部扫描/必要复审顺序执行，JSON结构/证据验证失败的有限恢复另降至16、1；网络离线只等待，不填0，不重采样已接受评分。

普通Ark完整endpoint为`https://ark.cn-beijing.volces.com/api/v3/chat/completions`。本服务器直连外网当前不可用，仍依赖38790的SSH反向代理：后台进程独立于终端，不等于网络独立于电脑。`NETWORK_STATUS.json`为connected或waiting_for_network，离线后需重建代理才能自动续跑。

## 审核输出

- `manifest.json` / `ARK_STANDARD_RESUME_PROFILE.json`：冻结输入、代码/缓存hash、模型和传输变更。
- `cells/<arm>/<task>/scan_prompt.txt`、`scan_raw_*`、`scan.json`：实际完整请求与首次合法扫描。
- 同目录`merged_proposals.json`、`review_prompt.txt`、`review.json`：合并疑点、核验及维度归因。
- `result.json`：旧新分数、确认/未决错误、局部证据、Judge与程序分歧。
- `task_rows.json`、`skill_rows.json`、`confirmed_error_cases.json`、`REPORT.md`：逐题/分Skill/总体与bad case。
- `FINISHED.json`或`FINISHED_WITH_UNRESOLVED.json`：完整结束或仍有未决，不把后者当全量结论。

Git中的进度快照以文件时间为准，不是实时日志。最终汇总须确认607份有效答卷全部评分完成，并加上3份原执行失败；不以均分下降多少决定是否接受这套协议。
