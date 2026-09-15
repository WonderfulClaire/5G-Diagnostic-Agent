# 真实模型、工具交互与数据飞轮

项目提供一个可检查的单卡训练链路。它直接使用 Transformers 与 PEFT，适合逐条审计训练信号；通用分布式 trainer 仍保留原入口。

## 最小运行链路

```bash
pip install -r requirements-local-model.txt
python -m scripts.telelogs.make_training_cases runs/train.jsonl
python -m scripts.telelogs.make_eval_cases runs/dev.jsonl
python -m recipes.telelogs.training.supervised runs/train.jsonl \
  --model /path/to/model --output runs/sft --steps 40
python -m scripts.telelogs.make_compound_cases runs/compound_train.jsonl
python -m recipes.telelogs.training.online_grpo runs/compound_train.jsonl \
  --model /path/to/model --adapter runs/sft/adapter --output runs/grpo \
  --steps 8 --group-size 4 --max-tokens 512
python -m recipes.telelogs.evaluation.run_agent runs/dev.jsonl --local \
  --model /path/to/model --adapter runs/grpo/adapter --output runs/predictions.jsonl
```

示例数据是显式合成的简单 5G 场景，训练/开发使用不同读数、措辞和部分不同根因组合；不等同于官方 TeleLogs，更不能用于声称生产网络泛化。

## 强化学习实现

- 模型实际生成多轮工具调用，每一步保存精确的输入前缀、生成 token、环境反馈和奖励。
- 旧策略概率在参数更新前计算，参考策略是训练开始时的冻结副本；从 SFT adapter 初始化时也保留该 adapter 的参考分布。
- 只对模型生成的 token 计算目标；工具观察作为下一步上下文，不进入目标 mask。
- 组内优势、clip、KL、梯度范数与更新状态均有记录。每组至多做一次参数更新，因此初始 importance ratio 为 1，clip 计数通常为 0；不把它当作已经充分验证了多 epoch PPO 的证据。
- 实验性组筛选将诊断质量与奖励分开：质量有差异的组用于 RL；只有辅助奖励变化的错误组转入审查；全正确但成本不同的组可用于效率 RL。这改变了训练样本选择，效果需要单独消融。

## 工具调用与恢复

完整案例只在环境侧；模型只看到症状、工具说明和已经返回的观察。模型必须等观察返回后再提交。重复证据、空查询和非法调用分别记录。

推理上下文直接采用环境返回的完整消息历史，避免把整个历史反复嵌入新消息。畸形或截断的 XML 工具调用整批拒绝，返回修复提示，不执行其中的部分调用；普通未提交的最终回答仍按失败结束。无论是否恢复，工具轮数上限都生效。

## 飞轮衔接

配套 [BLM 数据飞轮](https://github.com/WonderfulClaire/BLM-Multimodal-Audit/tree/main/data_flywheel) 可直接读取本项目训练轨迹：

```bash
# 在 BLM-Multimodal-Audit 仓库中执行
python -m data_flywheel.rl_feedback /path/to/grpo/trajectories.jsonl --output runs/routes.jsonl
```

真实训练中，稳定漏根因的样本与只有奖励变化的样本会走不同的后续处理。候选数据需保留来源、内容哈希、独立审核与规则检查；只创建新的数据版本，并保留旧样本回放。合成联调使用可执行规则核验标签，不宣称调用了真实大教师或人工 Judge。

回训可从当前 adapter 增量开始，使用四次旧样本、一次新样本的确定性调度；`--replay-kl` 在旧样本的回答位置计算冻结旧模型到新模型的完整词表 KL，约束能力漂移：

```bash
python -m recipes.telelogs.training.supervised curated_train.jsonl \
  --model /path/to/model --adapter runs/current/adapter \
  --output runs/flywheel-next --steps 100 --lr 0.00002 --replay-kl 0.2
```

数据混合比例本身不能保证旧能力不退化。是否接受回训版本，要看固定评测里的新问题修复、旧类别退化与统计不确定性，不能只看训练 loss。

[本次实验记录](../reports/experiments/20260915/REPORT.md) 保留失败结果、实现版本边界和后续决策。

后续增加奖励与正确性排序检查：若更差回答反而获得更高奖励，返回 `audit_reward_quality_conflict` 并跳过该组更新。它是保守的分流规则，质量标签不可靠时仍可能误判，不能替代奖励审核。当前小型轨迹回放没有触发新增分支，尚无训练收益证据。

[`--diagnosis-weight` 单变量实验](../reports/experiments/20260915-retention/REPORT.md)只加权生成回答中的根因数组，保留其余字段监督。权重8与权重1均为8/12、单根因均5/8，未胜过对照且未保住原模型6/8的单根因能力，因此未采用；32条新测试保持封存。
