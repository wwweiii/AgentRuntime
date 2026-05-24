from pathlib import Path

from agent_runtime.core.loader import load_scenario


def test_load_example() -> None:
    config, agents, tasks, root = load_scenario(Path("examples/software_dev_team/task.json"))

    assert config.model == "qwen3.5:4b"
    assert "Todo CLI" in root["goal"]
    assert any(agent.agent_id == "planner" for agent in agents)
    assert any(task.task_id == "plan" for task in tasks)

