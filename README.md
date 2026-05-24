# Agent Runtime

面向多智能体系统的 Python 用户态执行时原型。把 Agent 建模为类 OS 进程的一等执行实体，提供任务调度、上下文复用、消息通信、故障隔离、资源账本和系统级可观测能力。

## 环境要求

- Python ≥ 3.10
- [Ollama](https://ollama.com)（可选，关闭后使用 mock 模型验证调度/容错机制）

推荐模型：

```powershell
ollama pull qwen3.5:4b          # 生成模型（4B 参数，消费级 GPU 可用）
ollama pull qwen3-embedding:4b  # 嵌入模型，用于语义级上下文去重
```

## 快速开始

```powershell
# 安装依赖
pip install -e .

# mock 模式 — 验证调度、上下文复用、故障恢复
arun submit examples/software_dev_team/task.json --mock

# 真实模型 — 需要先启动 Ollama
arun submit examples/software_dev_team/task.json

# 增大超时（真实模型响应较慢时使用）
arun submit examples/software_dev_team/task.json --timeout 300
```

## 查看运行结果

```powershell
# 列出所有 run 及概要
arun inspect runs

# 查看单个 run 的完整指标
arun inspect runs/<run-id>/metrics.json
```

每个 run 目录的结构：

```
runs/<run-id>/
├── metrics.json      # 任务状态分布、上下文复用率、token 消耗
├── events.jsonl      # 完整事件时间线
├── contexts.json     # 上下文段详情
├── tasks.json        # 各任务最终状态
└── workers/          # 每个 worker 的 input / result / log
```

## 性能对比

```powershell
# 默认 3 轮，所有场景
arun benchmark --mock

# 单场景，5 轮
arun benchmark --scenario examples/code_review_pipeline/task.json --rounds 5

# 输出 JSON 供分析
arun benchmark --json > report.json
```

## 项目结构

```
agent_runtime/
├── cli.py                 # 命令行入口（submit / inspect / benchmark / serve）
├── core/
│   ├── runtime.py         # AgentRuntime 主循环：collect → schedule → start
│   ├── models.py          # 数据模型：AgentSpec, AgentTask, RuntimeConfig, WorkerResult
│   ├── state.py           # 9 状态任务生命周期
│   └── loader.py          # 场景文件加载
├── context/
│   └── store.py           # 上下文存储：三段式去重（SHA256 → 归一化 → 语义嵌入）
├── scheduler/
│   └── dag_scheduler.py   # 上下文感知 DAG 调度器
├── worker/
│   ├── process_worker.py  # 进程级隔离 worker 管理
│   └── runner.py          # 子进程执行体：模型调用 + 动态任务提取
├── model/
│   ├── ollama_client.py   # Ollama /chat API 封装（含 30+ 关键词 mock 表）
│   └── embedding_client.py # Ollama /api/embed 封装 + SHA256 缓存
├── fault/
│   └── recovery.py        # 故障管理器：重试/退避/fallback agent
├── message/
│   └── bus.py             # 消息总线：pub/sub + mailbox
├── resource/
│   └── quota.py           # CPU/内存/token 资源配额管理
├── osadapter/             # OS 适配层（Windows 进程隔离，Linux cgroups 扩展点）
└── observability/
    └── event_log.py       # 事件日志 + Metrics 计数器
examples/                  # 多 Agent 场景
tests/                     # 单测
benchmark/                 # 性能对比框架
```

## 核心能力

| 维度 | 实现 |
|------|------|
| 任务生命周期 | Created → Ready → Scheduled → Running → Waiting → Completed / Failed / Retrying / Killed |
| DAG 调度 | 依赖解析、优先级、资源预算、上下文亲和性 |
| 动态任务 | Agent 输出 `RUNTIME_DYNAMIC_TASK` 触发 Runtime 生成后续任务，含深度/去重控制 |
| 上下文去重 | 三级流水线：SHA256 精确 → 归一化 text → 语义 embedding（cosine 相似度） |
| 上下文隔离 | Segment 所有者为 Agent，支持 public / shared / private 可见性 |
| 进程隔离 | 每任务独立子进程，文件通道 IPC |
| 容错 | 超时终止、自动重试（退避）、fallback agent 级联、故障注入 |
| 消息通信 | pub/sub 模式，传递 context_id 引用而非完整上下文 |
| 可观测 | 结构化事件日志、Metrics 计数器、上下文复用率/snapshot 数 |

## 创建自定义场景

场景文件为 JSON，包含 `runtime` 配置、`agents` 列表、`tasks` 列表和可选的 `root_context`。参考 `examples/` 目录下的示例。

```json
{
  "runtime": { "max_workers": 2, "token_budget_per_task": 8000 },
  "agents": [
    { "agent_id": "analyst", "role": "分析师", "system_prompt": "你是数据分析师..." }
  ],
  "tasks": [
    { "task_id": "analyze", "agent_id": "analyst", "objective": "分析数据" }
  ],
  "root_context": { "goal": "完成数据报告" }
}
```
