# 5G Diagnostic Agent

面向 5G 下行吞吐异常的多轮根因诊断 Agent。模型按需查询无线指标、邻区关系、移动性、天线、切换和资源分配信息，形成有证据支持的诊断与修复建议。

## 项目重点

- **查询—观察—提交协议**：完整案例在环境侧保存；模型只看到问题和工具返回。至少观察两个非空视角才能提交，查询与最终提交必须分轮执行。
- **证据增量与调用成本**：记录内容哈希和 evidence ID；奖励新增记录，惩罚重复查询、非法调用，并对每次查询收取成本。相同工具返回新记录仍可获得信息增量奖励。
- **LoRA + GRPO 训练配置**：环境将分步交互交给 rollout/trainer，配置组采样、KL、LoRA 与训练集/开发集。完整 GPU 训练需要预训练模型及兼容运行时。
- **分层评测与可审计轨迹**：保存动作、观察、奖励分解和失败类型，报告根因 Exact Match、逐样本集合 F1、Micro F1、工具失败率与平均调用次数。

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
  --max-steps 8 --output runs/predictions.jsonl
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

工具协议通过不等于诊断能力提升。当前公开版新增的测试验证实现正确性，没有新增经过独立测试集确认的 GRPO 提升数字。奖励中的证据/修复匹配含启发式成分，仍可能被模型迎合；请同时检查 evidence ID、原始观察和独立判分。`macro_f1` 保留为旧字段别名，准确名称是 `mean_sample_set_f1`，不是按类别平均的 Macro F1。

[实验协议](docs/EXPERIMENT_PROTOCOL.md) · [本次验证记录](reports/release_validation.md) · [代码来源与许可](THIRD_PARTY_NOTICES.md)

仓库不分发受限 TeleLogs 样本、私有日志或模型权重。项目工作集中在 5G 环境、证据协议、奖励和评测适配；通用训练框架的来源见许可说明。
