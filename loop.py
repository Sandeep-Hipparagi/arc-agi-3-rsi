import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

from src.repl_session import RLMReplSession
from src.task_memory import TwoStageWorldModel
from src.env_bridge import ArcEnvBridge, MockArcEnvBridge
from src.task_agent import TaskAgent


def compute_weighted_game_score(level_steps: Dict[int, int], human_baselines: Dict[int, int]) -> float:
    """
    ARC-AGI-3 Weighted Level Scoring:
    Level i (1-indexed) carries weight i.
    Each completed level scores min(1.0, human_steps / agent_steps).
    """
    total_weight = sum(human_baselines.keys())
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    for lvl_idx, baseline in human_baselines.items():
        if lvl_idx in level_steps:
            steps = max(1, level_steps[lvl_idx])
            lvl_score = min(1.0, baseline / float(steps))
            weighted_sum += lvl_idx * lvl_score
    return round(weighted_sum / total_weight, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Milestone 1 Inner-Loop Evaluator")
    parser.add_argument("--game", type=str, default="ls20", help="ARC-AGI-3 game ID (e.g., ls20, ft09, vc33)")
    parser.add_argument("--skill", type=Path, default=Path("skills/SKILL.md"), help="Path to SKILL.md")
    parser.add_argument("--model", type=str, default=os.getenv("INNER_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--mode", type=str, default="OFFLINE", choices=["OFFLINE", "ONLINE", "NORMAL"])
    parser.add_argument("--no-vision", action="store_true", help="Disable image attachment on render_frame_strip()")
    parser.add_argument("--mock", action="store_true", help="Run with deterministic MockArcEnvBridge instead of real arc_agi environment")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without external LLM calls")
    args = parser.parse_args()

    has_api_key = bool(args.api_key or os.getenv("OPENAI_API_KEY"))
    if not has_api_key and not args.base_url and not args.dry_run:
        print("[!] Note: No OPENAI_API_KEY or OPENAI_BASE_URL detected. Defaulting to zero-cost DRY-RUN mode.")
        print("[!] To execute with live LLMs: export OPENAI_API_KEY='...' (or pass --api-key).")
    is_dry_run = args.dry_run or (not has_api_key and not args.base_url)

    run_id = f"{'mock' if args.mock else args.game}_{int(time.time())}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_file = run_dir / "trajectory.jsonl"

    skill_markdown = args.skill.read_text(encoding="utf-8") if args.skill.exists() else "# ARC-AGI-3 Domain Skill"
    repl = RLMReplSession(artifact_dir=run_dir / "frames")
    world_model = TwoStageWorldModel()

    if args.mock:
        print("[*] Initializing MockArcEnvBridge...")
        env_bridge = MockArcEnvBridge(game_id=args.game)
    else:
        print(f"[*] Initializing ArcEnvBridge (game={args.game}, mode={args.mode})...")
        env_bridge = ArcEnvBridge(game_id=args.game, operation_mode=args.mode)

    def trace_logger(event: Dict[str, Any]) -> None:
        event["timestamp"] = time.time()
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    agent = TaskAgent(
        skill_markdown=skill_markdown,
        repl=repl,
        world_model=world_model,
        env_bridge=env_bridge,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        enable_vision=not args.no_vision,
        trace_callback=trace_logger,
        dry_run=is_dry_run,
    )

    if is_dry_run:
        print(f"[*] Running in DRY-RUN mode on '{args.game}' (max_env_steps={args.max_steps})...")
    else:
        print(f"[*] Starting Inner Loop on game='{args.game}' with model='{args.model}'...")

    final_outcome = agent.run_episode(max_env_steps=args.max_steps)

    official_scorecard = env_bridge.get_official_scorecard()

    summary = {
        "run_id": run_id,
        "game_id": args.game,
        "model": args.model if not is_dry_run else f"{args.model} (dry-run)",
        "skill_path": str(args.skill),
        "final_status": final_outcome.state_status,
        "levels_completed": final_outcome.levels_completed,
        "total_env_steps": final_outcome.env_steps_total,
        "final_world_model": {
            "proven_mechanics": world_model.proven_mechanics,
            "unverified_hypotheses": world_model.unverified_hypotheses,
            "fatal_transitions": world_model.fatal_transitions,
        },
        "official_scorecard": official_scorecard,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[+] Run complete! Summary written to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
