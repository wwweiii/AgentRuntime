from __future__ import annotations

import platform

from agent_runtime.osadapter.base import OSAdapter
from agent_runtime.osadapter.linux import LinuxAdapter
from agent_runtime.osadapter.windows import WindowsAdapter


def get_os_adapter() -> OSAdapter:
    if platform.system().lower().startswith("win"):
        return WindowsAdapter()
    if platform.system().lower() == "linux":
        return LinuxAdapter()
    return OSAdapter()

