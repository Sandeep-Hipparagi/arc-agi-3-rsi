import json
import time
from pathlib import Path
from typing import Dict, Any

class DeterministicEvaluator:
    """
    Runs a single game evaluation with a fixed SKILL.md and outputs
    normalized scores and full trajectory traces for the Outer Loop.
    """
    def __init__(self, skill_path: Path, trace_dir: Path, max_env_steps: int = 400):
        self.skill_path = skill_path
        self.skill_markdown = skill_path.read_text(encoding="utf-8")
        self.trace_dir = trace_dir
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.max_env_steps = max_env_steps

    @staticmethod
    def compute_level_score(agent_steps: int, human_baseline_steps: int, completed: bool) -> float:
        """
        ARC-AGI-3 objective scoring:
        100% (1.0) if agent_steps <= human_baseline_steps.
        Scales down proportionally if more steps are taken; 0.0 if incomplete.
        """
        if not completed:
            return 0.0
        if agent_steps <= human_baseline_steps:
            return 1.0
        return round(human_baseline_steps / float(agent_steps), 4)

    def log_event(self, trace_file: Path, payload: Dict[str, Any]) -> None:
        payload["timestamp"] = time.time()
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
