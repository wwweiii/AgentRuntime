from enum import Enum


class TaskState(str, Enum):
    CREATED = "created"
    READY = "ready"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    WAITING = "waiting"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    KILLED = "killed"


TERMINAL_STATES = {
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.KILLED,
}

