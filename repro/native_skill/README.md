# 随题候选技能包：原生自主使用评测

此入口采用同一Docker/OpenCode环境比较No-Skill与With-Skill。With-Skill只注册随题候选包，不把Skill全文写进题面，也不要求必须调用某Skill。Agent自行决定读取、选择模式及使用哪些操作。

默认候选包只有该题预映射的一个Skill，不是全库搜索。`expected_skill`只是保留的映射标签，不应未经适用性验收就当作唯一正确路由金标。可用`--candidate-map`为每题配置多个候选；仍不能据此声称完成全库检索。

## 目录与安装

在服务器的`csl/`下克隆此分支，仓库内相对路径保持统一：

```text
csl/EduSkillBench/
  data/exports/eduskillbench-305-20261003/cases.json
  skills/single_turn/<skill>/SKILL.md
  repro/native_skill/{run.py,harness.py,runtime.py,setup.sh}
  repro/native_skill/upstream/{judge.py,source_protocol.py,profiles.py}
  .venv-native-skill/
  artifacts/<独立运行名>/{tasks,cells,jobs,manifest.json,status.json,summary.json}
```

要求Python3.12、可用Docker及Docker组权限；已有Docker时无需为每次实验重新安装。运行`bash repro/native_skill/setup.sh`安装独立虚拟环境、冻结的107项依赖，恢复八文件BenchFlow补丁，构建预装镜像并校验268个源码文件。已有匹配环境可以直接运行`python -m repro.native_skill.verify`，不覆盖训练环境。

Docker重建会得到本机自己的镜像ID，因此各运行记录实际ID；若需逐字节迁移现有镜像，使用Docker save/load后再校验。不能把同版本号说成跨服务器镜像逐字节一致。

## 冻结配置

| 项目 | 设置 |
| --- | --- |
| 框架 | BenchFlow0.6.7，268个源码hash校验 |
| 容器 | Node22.20.0、OpenCode1.18.11，预装运行时+ripgrep |
| system | 教育单轮任务；缺非关键材料标假设，不编造课堂效果/教师批准 |
| 题面 | `context + user_prompt`；不注入rubric或Skill全文 |
| 候选包 | 原生注册到`.config/opencode/skills`；文件对agent可读 |
| 环境 | 每题1CPU/1024MB，600秒；启动容量2，运行容量默认16 |
| 生成 | 同一工具接口、max_tokens上限10000、单请求480秒；Provider必要选项显式记录 |
| Judge | DeepSeek V4 Pro；必须返回`deepseek-v4-pro-ga-260813`；thinking disabled |
| 评分并发 | 默认64，可独立调整，不与环境并发混淆 |
| 证据 | 用户任务父会话最后一个正常结束的完整最终答案；不拼工具日志或文件内容 |
| 263题 | 适用维度最高原生等级占比，再按题平均 |
| 42题 | 所有rubric子项PASS才算整题1，否则0；与263分开报告 |

没有独立全量科学正确性验证器。Judge评分不能自动当作完整事实正确性认证。

## 使用

密钥只放环境变量，不进入仓库、容器或命令行参数。`LLM_API_KEY`供生成，`JUDGE_API_KEY`可单独供Judge；后者未设则复用前者。需要网络代理时分别设`POLICY_PROXY`和`JUDGE_PROXY`。本地Docker/健康检查不经过外部代理。

先做不调用API的准备：

```sh
.venv-native-skill/bin/python -m repro.native_skill.run \
  --out artifacts/glm_native_smoke --model glm-5.3 \
  --policy-base-url https://ark.cn-beijing.volces.com/api/plan/v1 \
  --policy-options '{"thinking":{"type":"enabled"},"reasoning_effort":"low"}' \
  --task-id lesson-builder__cn01_01 --conditions no_skill with_skill
```

加`--execute`执行该题端到端试跑。确认完整回答、Skill注册和评分身份后，新目录去掉`--task-id`执行305题×两条件：

```sh
nohup .venv-native-skill/bin/python -u -m repro.native_skill.run \
  --out artifacts/glm_native305 --model glm-5.3 \
  --policy-base-url https://ark.cn-beijing.volces.com/api/plan/v1 \
  --policy-options '{"thinking":{"type":"enabled"},"reasoning_effort":"low"}' \
  --concurrency 16 --judge-concurrency 64 --execute \
  > native305.log 2>&1 < /dev/null &
```

GLM-5.3必须使用enabled/low：实际接口拒绝disabled。Qwen4B Base/SFT/GRPO沿用相同题库、镜像、system、Skill暴露与Pro评分，只替换本地OpenAI兼容服务的`--model`和`--policy-base-url`，不向Qwen盲传GLM专属options。不同权重各用独立`--out`，服务若在另一机器需保证宿主bridge可达。

纯远端模型不占本地GPU。本地Qwen服务应由其启动器在所属任务推理结束后卸载；本入口不会擅自杀死共享服务器上的其他模型进程。

候选表格式示例：`{"lesson-builder__cn01_01":["lesson-builder"]}`。不指定时默认按task_id前缀选一个。不要为强行增加调用率在题面提示预期Skill名称。

## 留存与恢复

`tasks/`保存原题面、候选资源及清单；`manifest.json`冻结全部hash、模型、选项和镜像；`jobs/`保留原始数据库、ACP和清理前捕获；`cells/<条件>/<题号>/`保存最终答案、评分原文、逐项判定和`routing.json`。

重复执行同一命令只恢复未完成阶段：已有最终答案用于评分，不因评分失败重新rollout；已有合法评分不重复抽样。改变配置必须使用新目录。执行/交付终态失败记0并保留原因；Judge未决保留null，不能伪装成0或宣称全量完成。`summary.json`不生成混淆两套协议的305总分。

`routing.json`分开记录预期Skill、候选列表、实际skill工具名称及状态、SKILL.md文件访问。读取命中率不是功能采用率，更不是全库路由准确率。多候选实验另行固定候选集与正确路由定义。

## 代码与输入来源

题库/Skill及教育system源自Airlivy/EduSkillBench冻结提交`5a9c6c2`；Pro提示词与判定器源自`6270c70`。第三方Skill许可见根目录`THIRD_PARTY_NOTICES.md`。环境说明与实验结果清单分开保存，逐题来源和配置不可省略。

移植后的离线源码/hash检查与启动准备，不等于新服务器已经完成全量实验。每台服务器应先完成端到端试跑再放行批量。
