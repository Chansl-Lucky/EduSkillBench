# Qwen3-4B 论文对齐复现实验协议

> 状态：已准备，未启动。这里的“Qwen 4B”特指固定 revision
> `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`。

## 结论先行

严格对齐论文 Qwen 路线时，policy 和主 Judge 使用同一份 Qwen3-4B 权重、同一
served model alias。这里的“同样”不是只要求同属 Qwen 家族，而是要求模型 ID、
revision、服务模板均一致。

这会保留论文的同模型自评设定，也会保留它固有的 self-preference 风险。因此同时
保存一条完全隔离的 DeepSeek-V4-Flash 审计分数：审计分不回传、不替换主分，也
不能与主分混成一个指标。

## 测量链路

```text
同一 Qwen3-4B vLLM 服务
  ├─ policy：BenchFlow → OpenCode → 完整 agent trajectory
  └─ primary judge：BenchFlow PASS/FAIL rubric → reward

冻结 trajectory
  └─ audit judge：DeepSeek-V4-Flash → audit reward（只用于稳健性分析）
```

两组条件均由 BenchFlow 自动运行：`No Skill` 与 `With Skill`。Skill 是 OpenCode
sandbox 中可调用能力，不再把完整 SKILL.md 注入 system prompt。Judge 读取完整
trajectory，并按每条 rubric 的 PASS/FAIL 比例计算分数。

## 与论文一致和不一致的部分

一致项：官方 42 题、14 个 Skill、OpenCode、Docker、BenchFlow、完整 trajectory、
逐项 PASS/FAIL、Qwen policy 配 Qwen 主 Judge。

唯一核心替换：论文的 hosted Qwen3.7-Plus 替换为可训练的本地
Qwen3-4B-Instruct-2507。因此该结果是“框架对齐的小模型复现”，不是论文模型行的
逐数值复现。

预注册改进：No-Skill 与 With-Skill 都使用 600 秒 timeout，避免不对称补跑造成
额外混杂。以前的 Transformers 直接生成 + system 注入结果只作为 bridge baseline，
不得和本次 OpenCode 结果冒充完全同条件比较。

## 已准备的执行顺序

1. 等当前 `reward_feasibility_002` 结束并确认 GPU 1 空闲。
2. 安装 Docker 并确认当前用户可访问 daemon。
3. 建立独立 `.venv-qwen4b-serve`，不污染训练 `.venv`。
4. 生成 42 题只读式副本；原始 released evals 不修改。
5. 启动 vLLM，先做普通回答和 tool-call 两类 smoke test。
6. 以 concurrency=1 跑 14 个 Skill 的两组条件。
7. 汇总主 Judge paired lift、逐 Skill 结果、失败/超时/截断率和置信区间。
8. 冻结 trajectory 后用 DS V4 Flash 做独立复评，报告 Judge agreement 与结论是否反转。

## 命令（现在不要执行启动步骤）

```bash
uv run python code/evaluation/prepare_qwen4b_paper_aligned.py
code/evaluation/setup_qwen4b_serving_env.sh
code/utils/patch_benchflow.sh --apply

# 阶段结束后由守卫脚本启动两个 tmux 会话：
code/evaluation/launch_qwen4b_paper_aligned.sh
```

运行前检查：

```bash
code/evaluation/preflight_qwen4b_paper_aligned.sh
```

服务只绑定 Docker bridge gateway，不绑定公网网卡；policy 与容器内 Judge 都通过
该 bridge 地址访问同一进程。

当前机器的 Docker 安装需要一次交互式 sudo（自动化进程不能代填密码）：

```bash
code/evaluation/install_docker_ubuntu.sh
```

脚本完成后重新登录一次，使 `docker` 用户组生效，再运行 preflight。

主日志固定为 `logs/paper_aligned_qwen3_4b_001.log`；模型服务日志应在启动 tmux
时重定向到 `logs/qwen3_4b_vllm.log`。所有脚本只从环境变量读取可替换的 API key，
不把真实 key 写入仓库。

主运行结束后会自动生成：

- `artifacts/paper_aligned_qwen3_4b_001/summary/overall.json`
- `artifacts/paper_aligned_qwen3_4b_001/summary/skills.csv`
- `artifacts/paper_aligned_qwen3_4b_001/summary/runs.csv`

其中 overall 包含 42 个任务配对 lift 的 20,000 次 bootstrap 95% CI。

## 启动门槛

- 当前阶段任务已结束；
- GPU 1 空闲且显存满足 32K KV cache；
- Docker、vLLM、BenchFlow patch、42 题 manifest 全部通过 preflight；
- `/v1/models` 返回且 model alias 精确匹配；
- tool-call smoke test 确认 OpenCode 能发现并调用 Skill；
- manifest 在启动前写入实际包版本、GPU、模型 revision 和时间戳。

任何一项失败都不进入 84-cell 正式运行，避免再次得到“答案生成了，但不是论文同一
测量过程”的不可比结果。
