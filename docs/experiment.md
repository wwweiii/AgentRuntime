# 实验设计

## 实验 1：上下文复用收益

对比对象：

- Baseline：每个 Agent 都拼接完整历史
- Agent Runtime：使用 ContextStore 的 segment、snapshot 和 context_ref

指标：

- `model.context_tokens_est`
- `context.reuse_ratio`
- `context.dedupe_hits`
- `context.segments`
- `benchmark.estimated_context_token_saving`

运行：

```powershell
python -m agent_runtime.cli benchmark --mock
```

## 实验 2：调度机制

对比策略：

- FIFO，作为后续扩展 baseline
- DAG-aware
- Resource-aware + Context-aware，本项目默认策略

指标：

- 总耗时
- completed/failed 数量
- worker 并发数
- 模型调用 token 估计

## 实验 3：容错机制

故障类型：

- Ollama 未启动
- AgentTask timeout
- Worker 进程异常退出
- Agent 不存在

观察：

- Runtime 是否继续运行
- 是否产生 `task.failed_attempt`
- 是否进入 `task.retrying`
- fallback task 是否创建

## 实验 4：通信机制

对比：

- 传递完整上下文文本
- 传递 `context_ref`

本项目的消息事件中只记录 `context_ref`，上下文正文由 ContextStore 管理。

