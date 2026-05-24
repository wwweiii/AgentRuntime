# Agent Runtime

面向多智能体系统的 Python 用户态执行时原型。它把 Agent 建模为类似 OS 进程的一等执行实体，提供任务调度、上下文复用、消息通信、故障隔离、资源账本和系统级可观测能力。

本项目优先支持 Windows 本地开发，并通过 `OSAdapter` 保留 Linux/openEuler/openKylin 上接入 cgroups、namespace、seccomp 的扩展点。

## 快速运行

先确认 Ollama 已启动，并且本地已有模型：

```powershell
ollama run qwen3.5:9b
```

运行示例：

```powershell
python -m agent_runtime.cli submit examples/software_dev_team/task.json
```

如果不想调用模型，可使用 mock 模式验证运行时机制：

```powershell
python -m agent_runtime.cli submit examples/software_dev_team/task.json --mock
```

查看运行结果：

```text
runs/<run_id>/events.jsonl
runs/<run_id>/metrics.json
runs/<run_id>/contexts.json
```

## 核心能力

- Agent 生命周期：Created、Ready、Scheduled、Running、Waiting、Completed、Failed、Retrying、Killed
- DAG 调度：支持任务依赖、优先级、资源预算、上下文亲和性
- 动态任务：Agent 输出可触发 Runtime 生成后续任务
- 上下文管理：segment hash 去重、snapshot、delta、压缩、隔离
- 通信机制：mailbox 与 pub/sub，消息传递 `context_id` 而不是完整上下文
- 容错机制：进程隔离、超时、重试、fallback、故障注入
- 可观测性：事件日志、指标、trace、上下文复用率
- OS 适配：Windows 本地进程隔离；Linux 预留 cgroups adapter

## 推荐演示

```powershell
python -m agent_runtime.cli submit examples/software_dev_team/task.json --mock
python -m agent_runtime.cli inspect runs
python -m agent_runtime.cli benchmark --mock
```

