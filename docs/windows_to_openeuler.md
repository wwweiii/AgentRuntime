# 从 Windows 原型迁移到 openEuler/openKylin

## 当前 Windows 能力

- Python Runtime Core
- multiprocessing Worker 隔离
- timeout / retry / fallback
- ContextStore
- MessageBus
- Ollama 本地模型调用
- JSONL trace

## Linux 增强路径

### cgroups v2

在 `agent_runtime/osadapter/linux.py` 中为每个 AgentTask 创建 cgroup：

```text
/sys/fs/cgroup/agent-runtime/<run_id>/<task_id>
```

限制：

- `cpu.max`
- `memory.max`
- `pids.max`

### namespace

使用 `unshare` 或容器运行时为高风险工具创建隔离环境：

- mount namespace
- network namespace
- pid namespace

### seccomp

对代码执行类 Tool 限制危险 syscall，例如 mount、ptrace、reboot。

### eBPF

采集：

- 进程启动/退出
- CPU 时间
- 内存峰值
- 文件和网络访问事件

