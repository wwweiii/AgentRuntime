# Agent Runtime 设计说明

## 目标

Agent Runtime 是一个 Python 用户态多智能体执行时原型。系统从操作系统视角把 Agent 提升为一等执行实体，围绕 AgentTask 的生命周期、调度、上下文内存、通信、容错和可观测性提供统一管理能力。

## 核心抽象

### Agent-as-Process

`AgentSpec` 描述 Agent 的身份、角色、系统提示词、工具权限和资源画像。`AgentTask` 是一次可调度执行实例，包含依赖、优先级、资源请求、失败策略、上下文引用和状态。

生命周期：

```text
created -> ready -> scheduled -> running -> completed
                                  -> retrying -> running
                                  -> failed
                                  -> killed
```

### Context-as-Memory

上下文由 `ContextStore` 管理，不在 Agent 之间复制完整文本。上下文被拆成 `ContextSegment`，再组合为 `ContextSnapshot`。AgentTask 只携带 `context_id`，执行前由 Runtime 按权限和 token 预算物化。

机制：

- hash 去重：相同 segment 只存一份
- snapshot：不同任务共享底层 segment
- delta：Agent 输出作为新增 segment
- compression：超阈值后生成 summary segment
- isolation：支持 public、shared、private 与 allowed_agents

### Model/Tool-as-Device

LLM 被抽象为模型设备，目前实现 `OllamaClient`，默认使用本地 `qwen3.5:9b`。工具系统预留在 AgentSpec 的 `tools` 字段中，后续可以把代码执行器、文件系统、网络访问抽象为受控设备。

## 调度机制

`ContextAwareDAGScheduler` 支持：

- DAG 依赖满足后进入 ready
- priority 优先级
- ResourceManager 资源配额
- context locality，上下文复用越多得分越高
- retry penalty，失败重试任务降低调度得分

调度得分示意：

```text
score = priority * 10
      + context_reuse_score
      - dependency_penalty
      - memory_penalty
      - retry_penalty
```

## 容错机制

每个 AgentTask 在独立 `multiprocessing.Process` 中执行，Runtime 负责：

- timeout 后终止 Worker
- worker 崩溃后记录 failed attempt
- 按 FailurePolicy 自动 retry
- retry 耗尽后可创建 fallback task
- 单任务失败不直接导致 Runtime 崩溃

## 通信机制

`MessageBus` 支持：

- mailbox 点对点消息
- topic pub/sub
- 消息携带 `context_ref`，不携带完整上下文

## OS Adapter

当前 Windows 后端使用：

- 独立进程 Worker
- task 工作目录隔离
- timeout 和 Runtime 资源账本
- 可选 psutil 监控进程资源

Linux/openEuler/openKylin 后续增强：

- cgroups v2：CPU、内存、进程数限制
- namespace：文件系统、网络、进程隔离
- seccomp：系统调用过滤
- eBPF：资源事件采集

## 可观测性

每次运行会生成：

```text
runs/<run_id>/events.jsonl
runs/<run_id>/metrics.json
runs/<run_id>/contexts.json
runs/<run_id>/tasks.json
```

核心事件：

- runtime.created
- task.scheduled
- task.started
- task.completed
- task.failed_attempt
- task.retrying
- dynamic_task.created
- message.published

