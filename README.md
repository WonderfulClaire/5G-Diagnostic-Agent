# 5G Diagnostic Agent

> 2026-09-15审计更新：早期合成案例编号携带根因ID并进入提示词，存在标签泄漏风险。现已移除模型输入中的编号，并建立[修复协议后的数据对照](docs/VERIFIED_CURRICULUM.md)。此前开发集成绩保留供调试追溯，不作为无泄漏的能力证据。

面向 5G 下行吞吐异常的多轮根因诊断 Agent。模型按需查询无线指标、邻区关系、移动性、天线、切换和资源分配信息，形成有证据支持的诊断与修复建议。

## 项目重点

- **查询—观察—提交协议**：完整案例在环境侧保存；模型只看到问题和工具返回。至少观察两个非空视角才能提交，查询与最终提交必须分轮执行。
- **证据增量与调用成本**：记录内容哈希和 evidence ID；奖励新增记录，惩罚重复查询、非法调用，并对每次查询收取成本。相同工具返回新记录仍可获得信息增量奖励。
- **LoRA + GRPO 训练配置**：环境将分步交互交给 rollout/trainer，配置组采样、KL、LoRA 与训练集/开发集。完整 GPU 训练需要预训练模型及兼容运行时。
- **分层评测与可审计轨迹**：保存动作、观察、奖励分解和失败类型，报告根因 Exact Match、逐样本集合 F1、Micro F1、工具失败率与平均调用次数；新增 reward-alignment audit，可离线重算每个 GRPO group 的学习路由并检查是否发生不该发生的 optimizer update。
- **Harness generalization**：评测端支持 canonical / compact / alternate 三套语义等价的查询工具 schema；alias 在进入环境前映射回 canonical，并同时保存 raw/canonical action，用于区分任务能力与工具接口过拟合。

## 真实模型学习闭环

新增可执行的单卡 LoRA-SFT 与多轮 GRPO 训练器：真实 rollout、回答 token 掩码、冻结参考策略、组内学习信号检查，以及逐组奖励/梯度/KL 日志。数据飞轮读取这些轨迹后，可决定补示范、继续 RL 或审查奖励，再用旧样本回放与 KL 保护做增量回训。

[训练、工具恢复与飞轮联调](docs/LEARNING_LOOP.md) · [实验记录](reports/experiments/20260915/REPORT.md)

最新[无编号泄漏课程对照](reports/experiments/20260915-verified-curriculum/REPORT.md)：固定1.7B模型和2048步，混合单/双故障课程在首次32条冻结合成测试上达到27/32，单故障课程对照为16/32；两者单故障均16/16、工具错误均0。收益来自课程SFT，不能归因于RL或视为真实网络效果。完整原始预测、来源分组bootstrap及版本门槛可复核。

追加两个种子的配对复核均提高11/32；三个种子平均准确率44.79%→79.17%。其中seed123虽提高总分，仍因C7细分类别回退被门槛拒绝；不挑选最好种子作为普遍结论。

旧协议调试对照：[根因字段加权](reports/experiments/20260915-retention/REPORT.md) · [0.6B/1.7B容量与回训](reports/experiments/20260915-capacity/REPORT.md)。这些记录受上方编号泄漏问题影响，保留用于追溯。

[成对难负例与决策学习](reports/experiments/20260915-decision/REPORT.md)进一步对照决策SFT与SFT+DPO：DPO组合保留了8/8单故障，但双故障仍0/4，总正确数未超过初始模型，尚未通过版本验收。

## 本地验证

环境、数据预处理和评测测试可以独立于 GPU 训练依赖运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
python -m pytest tests/recipes/telelogs -q
```

测试覆盖：证据隔离、批量查询、原子拒绝、空查询、重复记录、新增记录、参数校验、状态重置、步数上限和可计算的集合指标。

## 模型推理

接入 OpenAI-compatible 模型服务，例如本机运行的推理服务。JSONL 每行包含 `id`、`split`、`symptom`、`case` 和仅供环境评分的 `ground_truth`；`case.sections` 是六类工具可检索的记录。

```bash
python -m recipes.telelogs.evaluation.run_agent cases.jsonl \
  --base-url http://localhost:8000/v1 --model MODEL_NAME \
  --max-steps 8 --harness-variant canonical \
  --output runs/predictions.jsonl
python -m recipes.telelogs.evaluation.evaluate_predictions runs/predictions.jsonl
```

需要认证时通过 `MODEL_API_KEY` 环境变量提供。运行器有工具轮数和 HTTP 超时限制，模型请求不携带完整 case 或标签；输出包含逐轮轨迹。模型服务由使用方启动。

## GPU 训练

先按 [训练依赖说明](docs/INSTALLATION.md) 准备环境。检查硬件及依赖后，使用训练集派生的开发集；官方测试集只用于最终评价。

```bash
python scripts/preflight.py --backend cuda
DATA_DIR=/path/to/data MODEL_PATH=/path/to/model CUDA_VISIBLE_DEVICES=0 \
  bash examples/telelogs/run_grpo_interview.sh
```

启动器提供单卡小规模 LoRA-GRPO 配置，参数可用 Hydra 覆盖。当前训练路径面向 CUDA；Ascend/NPU 的 trainer 与 rollout 适配尚未验证，不能把安装 torch-npu 等同于完成移植。

## 评测边界

工具协议通过不等于诊断能力提升。已完成小模型的合成数据 GPU 实验，详情见实验记录；目前没有经过独立业务测试集确认的 GRPO 提升数字。奖励中的证据/修复匹配含启发式成分，仍可能被模型迎合；请同时检查 evidence ID、原始观察和独立判分。`macro_f1` 保留为旧字段别名，准确名称是 `mean_sample_set_f1`，不是按类别平均的 Macro F1。

Reward 对齐审计：

```bash
python -m scripts.telelogs.audit_reward_alignment runs/grpo/metrics.jsonl --strict
```

它会用保存的 reward / correctness / efficiency cost 重新计算 learning route，并检查 route mismatch、reward-quality conflict 下误更新、以及本该更新却跳过的 group。这个工具不证明 reward 本身正确，但能防止“日志里已经显示冲突，训练器却还是更新了”这类可审计错误。

[实验协议](docs/EXPERIMENT_PROTOCOL.md) · [Harness generalization](docs/HARNESS_GENERALIZATION.md) · [本次验证记录](reports/release_validation.md) · [代码来源与许可](THIRD_PARTY_NOTICES.md)

仓库不分发受限 TeleLogs 样本、私有日志或模型权重。项目工作集中在 5G 环境、证据协议、奖励和评测适配；通用训练框架的来源见许可说明。
