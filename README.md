# Agent Runtime

面向多智能体系统的 Python 用户态执行时原型。把 Agent 建模为类 OS 进程的一等执行实体，提供任务调度、上下文复用、消息通信、故障隔离、工具调用、资源账本和系统级可观测能力。

## 环境要求

- Python ≥ 3.10
- [Ollama](https://ollama.com)（可选，关闭后使用 mock 模型验证调度/容错机制）

推荐模型：

```powershell
ollama pull qwen3.5:4b          # 生成模型（4B 参数，消费级 GPU 可用）
ollama pull qwen3-embedding:4b  # 嵌入模型，用于语义级上下文去重
```

可选依赖：

```powershell
pip install faiss-cpu           # 语义去重向量索引加速（≥200 segments 自动启用）
```

## 快速开始

```powershell
# 安装依赖
pip install -e .

# mock 模式 — 验证调度、上下文复用、故障恢复（毫秒级完成）
arun submit examples/software_dev_team/task.json --mock

# 真实模型 — 需要先启动 Ollama
arun submit examples/software_dev_team/task.json

# 增大超时（真实模型响应较慢时使用）
arun submit examples/software_dev_team/task.json --timeout 300
```

## CLI 命令

```powershell
# 提交多 Agent 场景
arun submit <scenario.json> [--mock] [--model <name>] [--timeout <seconds>]

# 查看运行结果
arun inspect runs                    # 列出所有 run 及概要
arun inspect runs/<run-id>/metrics.json  # 单个 run 的完整指标

# 性能基准对比（多场景多轮）
arun benchmark [--mock] [--scenario <path>] [--rounds <n>] [--json]

# A/B 对照实验（上下文复用 vs 无复用 vs 随机调度）
arun experiment [--mock|--no-mock] [--scenario <path>] [--rounds <n>] [--json]

# 启动本地 Dashboard
arun serve --port 8765
```

## 实验数据

在 `software_dev_team` 场景 (5 agents, 5 tasks DAG) 上使用 qwen3.5:4b 真实模型，2 轮对照实验：

| 指标 | BASELINE | NO-REUSE | RANDOM-SCHED |
|------|:---:|:---:|:---:|
| 执行时间 (s) | **190.97** | 292.40 | 245.67 |
| Segments 数 | **14** | 22 | 20 |
| Token 消耗 | **16,640** | 25,308 | 21,854 |
| 复用率 | **16.78%** | 0% | 4.35% |
| 共享率 | **97.75%** | 96.77% | 96.55% |

| 对比维度 | 提升 |
|----------|------|
| 上下文复用 vs 无复用 — 时间节省 | **34.7%** |
| 上下文复用 vs 无复用 — token 节省 | **34.3%** |
| 上下文复用 vs 无复用 — segment 节省 | **36.4%** |
| 上下文感知调度 vs 随机调度 — 时间节省 | **22.3%** |

详细分析见 [docs/experiment_report.md](docs/experiment_report.md)。

## 运行结果

每个 run 目录的结构：

```
runs/<run-id>/
├── metrics.json      # 任务状态分布、上下文复用率、token 消耗
├── events.jsonl      # 完整事件时间线
├── contexts.json     # 上下文段详情
├── tasks.json        # 各任务最终状态
└── workers/          # 每个 worker 的 input / result / log
```

## 项目结构

```
agent_runtime/
├── cli.py                 # 命令行入口（submit / inspect / benchmark / experiment / serve）
├── core/
│   ├── runtime.py         # AgentRuntime 主循环：collect → schedule → start
│   ├── models.py          # 数据模型：AgentSpec, AgentTask, RuntimeConfig, WorkerResult
│   ├── state.py           # 9 状态任务生命周期
│   └── loader.py          # 场景文件加载
├── context/
│   └── store.py           # 上下文存储：三级去重（SHA256 → 归一化 → 语义嵌入 + FAISS）
├── scheduler/
│   └── dag_scheduler.py   # DAG 调度器：依赖 + 优先级 + 资源感知 + 上下文亲和性
├── worker/
│   ├── process_worker.py  # 进程级隔离 worker 管理
│   └── runner.py          # 子进程执行体：模型调用 + 动态任务 + 消息指令 + 工具调用
├── model/
│   ├── ollama_client.py   # Ollama /chat API 封装（含 30+ 关键词 mock 表）
│   └── embedding_client.py # Ollama /api/embed 封装 + SHA256 缓存
├── tool/
│   ├── spec.py            # ToolSpec / ToolResult 数据模型
│   ├── registry.py        # 工具注册表
│   └── executors.py       # 内置工具：shell_cmd / read_file / write_file
├── fault/
│   └── recovery.py        # 故障管理器：重试/退避/fallback agent/熔断
├── message/
│   └── bus.py             # 消息总线：pub/sub + mailbox
├── resource/
│   └── quota.py           # CPU/内存/token 资源配额管理
├── osadapter/             # OS 适配层（Windows 进程隔离，Linux cgroups 扩展点）
└── observability/
    └── event_log.py       # 事件日志 + Metrics 计数器
benchmark/
    └── runner.py          # 性能基准框架
tools/
    └── comparison_report.py  # 框架对比分析（vs LangGraph / AutoGen）
examples/                  # 4 个多 Agent 场景
tests/                     # 24 个测试用例
docs/                      # 设计文档 + 实验报告
```

## 核心能力

| 维度 | 实现 |
|------|------|
| 任务生命周期 | Created → Ready → Scheduled → Running → Waiting → Completed / Failed / Retrying / Killed |
| DAG 调度 | 依赖解析、优先级、资源预算、上下文亲和性评分、随机调度对照 |
| 动态任务 | Agent 输出 `RUNTIME_DYNAMIC_TASK:<agent_id>:<objective>` 触发 Runtime 生成后续任务，含深度/去重控制 |
| 上下文去重 | 三级流水线：SHA256 精确 → 归一化 text → 语义 embedding（FAISS 向量索引，≥200 segments 自动切换） |
| 上下文隔离 | Segment 所有者为 Agent，支持 public / shared / private 可见性 + allowed_agents 白名单 |
| 上下文压缩 | Token 预算截断 + extractive summary（超阈值压缩） |
| 进程隔离 | 每任务独立子进程，文件通道 IPC |
| 容错 | 超时终止 → 自动重试（退避）→ fallback agent 级联 → 深度限制 → 失败率熔断 |
| 消息通信 | pub/sub 主题广播 + mailbox 点对点 + `RUNTIME_MESSAGE` 指令 + context_ref 引用传递 |
| 工具调用 | `RUNTIME_TOOL` 指令，内置 shell_cmd / read_file / write_file，支持扩展 |
| 资源账本 | CPU 槽位 + 内存 + token 预算 + 并行模型调用数 |
| 可观测 | 结构化 JSONL 事件日志、Metrics 计数器、per-run 导出、Dashboard |

## Agent 通信协议

Agent 可在输出中使用以下 Runtime 指令：

```
RUNTIME_DYNAMIC_TASK:<agent_id>:<objective>   # 请求创建新任务
RUNTIME_MESSAGE:<agent_id>:<content>          # 向其他 Agent 发送消息
RUNTIME_TOOL:<tool_name>                      # 调用工具（参数下行为 JSON）
{"param": "value"}
```

## 创建自定义场景

场景文件为 JSON，包含 `runtime` 配置、`agents` 列表、`tasks` 列表和可选的 `root_context`。参考 `examples/` 目录下的 4 个示例。

```json
{
  "runtime": {
    "max_workers": 2,
    "token_budget_per_task": 8000,
    "max_fallback_depth": 3,
    "failure_rate_threshold": 0.5
  },
  "agents": [
    { "agent_id": "analyst", "role": "分析师", "system_prompt": "你是数据分析师..." }
  ],
  "tasks": [
    {
      "task_id": "analyze",
      "agent_id": "analyst",
      "objective": "分析数据",
      "dependencies": [],
      "priority": 8,
      "failure_policy": {
        "retry": 1,
        "timeout_s": 120,
        "fallback_agent": "summarizer"
      }
    }
  ],
  "root_context": { "goal": "完成数据报告" }
}
```

### Runtime 配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_workers` | 2 | 最大并行 worker 数 |
| `max_parallel_model_calls` | 1 | 最大并行模型调用数 |
| `memory_budget_mb` | 4096 | 总内存预算 |
| `token_budget_per_task` | 12000 | 每任务 token 上限 |
| `context_compression_threshold` | 5000 | 上下文压缩触发阈值 |
| `max_dynamic_depth` | 2 | 动态任务最大嵌套深度 |
| `max_fallback_depth` | 3 | fallback 级联最大深度 |
| `failure_rate_threshold` | 0.5 | 失败率熔断阈值 |
| `semantic_threshold` | 0.90 | 语义去重 cosine 相似度阈值 |
| `disable_context_reuse` | false | 关闭上下文复用（用于 A/B 实验） |
| `random_scheduling` | false | 随机调度模式（用于 A/B 实验） |

## 运行测试

```powershell
pip install -e ".[dev]"
pytest tests/ -v    # 24 tests
```

## 框架对比

Agent Runtime 在 13 个关键维度上与 LangGraph / AutoGen 的对比：

| 框架 | 完全支持维度 |
|------|:---:|
| Agent Runtime | **13/13** |
| LangGraph | 6/13 |
| AutoGen | 5/13 |

Agent Runtime 独有优势：上下文去重、上下文可见性隔离、进程级故障隔离、资源配额管理、OS 内核集成、语义级上下文复用。

详细对比运行 `python tools/comparison_report.py`。
