import os
import json
import time
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

from src.repl_session import RLMReplSession
from src.task_memory import TwoStageWorldModel
from src.env_bridge import ArcEnvBridge, MockArcEnvBridge
from src.task_agent import TaskAgent
from src.trace_digest import TraceDigester
from src.learner_agent import LearnerAgent, SkillCandidate
from src.evolution_tree import EvolutionTree, SkillNode


def _evaluate_single_game(
    game: str,
    skill_markdown: str,
    skill_path: Path,
    eval_id: str,
    max_steps: int,
    inner_model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    is_mock: bool,
    is_dry_run: bool,
    artifact_root: Path,
) -> Dict[str, Any]:
    run_name = f"{eval_id}_{game}_{int(time.time() * 1000) % 1000000}"
    run_dir = artifact_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_file = run_dir / "trajectory.jsonl"

    repl = RLMReplSession(artifact_dir=run_dir / "frames")
    world_model = TwoStageWorldModel()

    if is_mock:
        env_bridge: Any = MockArcEnvBridge(game_id=game)
    else:
        env_bridge = ArcEnvBridge(game_id=game, operation_mode="OFFLINE")

    def trace_logger(event: Dict[str, Any]) -> None:
        event["timestamp"] = time.time()
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    agent = TaskAgent(
        skill_markdown=skill_markdown,
        repl=repl,
        world_model=world_model,
        env_bridge=env_bridge,
        model=inner_model,
        base_url=base_url,
        api_key=api_key,
        trace_callback=trace_logger,
        dry_run=is_dry_run,
    )

    final_outcome = agent.run_episode(max_env_steps=max_steps)
    official_scorecard = env_bridge.get_official_scorecard()

    summary = {
        "run_id": run_name,
        "game_id": game,
        "model": inner_model,
        "skill_path": str(skill_path),
        "final_status": final_outcome.state_status,
        "levels_completed": final_outcome.levels_completed,
        "total_env_steps": final_outcome.env_steps_total,
        "final_world_model": {
            "proven_mechanics": dict(world_model.proven_mechanics),
            "unverified_hypotheses": dict(world_model.unverified_hypotheses),
            "fatal_transitions": list(world_model.fatal_transitions),
        },
        "official_scorecard": official_scorecard,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Digest run
    digest = TraceDigester.digest_run(run_dir)
    digest_md = TraceDigester.render_digest_markdown(digest)

    score = digest.get("overall_score", 0.0)
    sc = digest.get("scorecard", {})
    lvl_scores = sc.get("level_scores", [])
    if not lvl_scores:
        lvl_scores = [score]

    actions = digest.get("total_env_steps", 0)
    hygiene = digest.get("hygiene", {})
    noops = hygiene.get("noop_steps", 0)
    resets = sc.get("resets", 0)
    proven_count = len(digest.get("final_world_model", {}).get("proven_mechanics", {}))
    game_tiebreaker = (proven_count * 10.0) - (resets * 5.0) - (noops * 1.0)

    return {
        "game": game,
        "score": score,
        "lvl_scores": lvl_scores,
        "actions": actions,
        "tiebreaker": game_tiebreaker,
        "run_dir": str(run_dir),
        "digest_md": digest_md,
    }


def evaluate_skill_on_batch(
    skill_path: Path,
    games: List[str],
    eval_id: str,
    max_steps: int,
    inner_model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    is_mock: bool,
    is_dry_run: bool,
    artifact_root: Path,
    workers: int = 2,
) -> Tuple[Dict[str, float], Dict[str, List[float]], float, float, int, Dict[str, str], str]:
    """
    Evaluates a single SKILL.md across a batch of games and digests telemetry.
    Supports concurrent evaluation across games using ThreadPoolExecutor.
    """
    skill_markdown = skill_path.read_text(encoding="utf-8")
    game_scores: Dict[str, float] = {}
    level_scores_by_game: Dict[str, List[float]] = {}
    total_actions_sum = 0
    total_tiebreaker = 0.0
    run_dirs: Dict[str, str] = {}
    digest_mds: List[str] = []

    actual_workers = max(1, min(workers, len(games)))
    if actual_workers > 1:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_single_game,
                    game=g,
                    skill_markdown=skill_markdown,
                    skill_path=skill_path,
                    eval_id=eval_id,
                    max_steps=max_steps,
                    inner_model=inner_model,
                    base_url=base_url,
                    api_key=api_key,
                    is_mock=is_mock,
                    is_dry_run=is_dry_run,
                    artifact_root=artifact_root,
                )
                for g in games
            ]
            results = [f.result() for f in futures]
    else:
        results = [
            _evaluate_single_game(
                game=g,
                skill_markdown=skill_markdown,
                skill_path=skill_path,
                eval_id=eval_id,
                max_steps=max_steps,
                inner_model=inner_model,
                base_url=base_url,
                api_key=api_key,
                is_mock=is_mock,
                is_dry_run=is_dry_run,
                artifact_root=artifact_root,
            )
            for g in games
        ]

    for r in results:
        g = r["game"]
        game_scores[g] = r["score"]
        level_scores_by_game[g] = r["lvl_scores"]
        total_actions_sum += r["actions"]
        total_tiebreaker += r["tiebreaker"]
        run_dirs[g] = r["run_dir"]
        digest_mds.append(r["digest_md"])

    mean_score = sum(game_scores.values()) / max(1, len(game_scores))
    combined_digests = "\n\n---\n\n".join(digest_mds)

    return (
        game_scores,
        level_scores_by_game,
        round(mean_score, 4),
        round(total_tiebreaker, 2),
        total_actions_sum,
        run_dirs,
        combined_digests,
    )


def run_evolution(
    games: List[str],
    generations: int = 2,
    branch_factor: int = 2,
    max_steps: int = 30,
    seed_skill_path: Path = Path("skills/SKILL.md"),
    seed_policy_path: Path = Path("skills/L2_learner_policy.md"),
    inner_model: str = "gemini-2.5-flash",
    outer_model: str = "gemini-2.5-flash",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    is_mock: bool = False,
    is_dry_run: bool = False,
    workers: int = 2,
    resume_path: Optional[Path] = None,
    exp_dir: Path = Path("evolution_runs"),
) -> EvolutionTree:
    """
    Executes the multi-generation evolutionary loop with parallel evaluation
    and optional checkpoint resumption.
    """
    if resume_path:
        exp_path = Path(resume_path)
        tree = EvolutionTree.load_from_checkpoint(exp_path)
        exp_id = tree.experiment_id
        skills_dir = exp_path / "skills"
        runs_dir = exp_path / "runs"
        print(f"\n==================================================")
        print(f"[*] Resuming Evolution Experiment: {exp_id} from {exp_path}")
        print(f"[*] Found {len(tree.nodes)} existing nodes, Max Gen: {tree.get_max_generation()}")
        print(f"[*] Games: {games} | Target Gens: {generations} | Branch: {branch_factor} | Workers: {workers}")
        print(f"==================================================")
    else:
        exp_id = f"exp_{'mock' if is_mock else 'live'}_{int(time.time())}"
        exp_path = exp_dir / exp_id
        skills_dir = exp_path / "skills"
        runs_dir = exp_path / "runs"
        skills_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)
        tree = EvolutionTree(experiment_dir=exp_path, experiment_id=exp_id)
        print(f"\n==================================================")
        print(f"[*] Starting Evolution Experiment: {exp_id}")
        print(f"[*] Games: {games} | Gens: {generations} | Branch: {branch_factor} | Workers: {workers} | Mock: {is_mock} | DryRun: {is_dry_run}")
        print(f"==================================================")

    learner = LearnerAgent(
        model=outer_model,
        base_url=base_url,
        api_key=api_key,
        dry_run=is_dry_run,
    )

    # 1. Evaluate Gen 0 Root Node if not already evaluated
    if "gen0_root" not in tree.nodes:
        print(f"\n[*] Evaluating Generation 0 (Root Skill: {seed_skill_path})...")
        root_skill_dest = skills_dir / "gen0_root.md"
        shutil.copy(seed_skill_path, root_skill_dest)

        root_policy_dest = skills_dir / "gen0_policy.md"
        if seed_policy_path.exists():
            shutil.copy(seed_policy_path, root_policy_dest)
        else:
            root_policy_dest.write_text("# L2 Learner Policy\n", encoding="utf-8")

        (
            r_scores,
            r_levels,
            r_mean,
            r_tiebreaker,
            r_actions,
            r_runs,
            r_digests,
        ) = evaluate_skill_on_batch(
            skill_path=root_skill_dest,
            games=games,
            eval_id="gen0_root",
            max_steps=max_steps,
            inner_model=inner_model,
            base_url=base_url,
            api_key=api_key,
            is_mock=is_mock,
            is_dry_run=is_dry_run,
            artifact_root=runs_dir,
            workers=workers,
        )

        root_node = SkillNode(
            node_id="gen0_root",
            parent_id=None,
            generation=0,
            skill_path=str(root_skill_dest),
            learner_policy_path=str(root_policy_dest),
            mutation_rationale="Initial Gen 0 Baseline Domain Skill",
            game_scores=r_scores,
            level_scores_by_game=r_levels,
            mean_batch_score=r_mean,
            tiebreaker_score=r_tiebreaker,
            total_actions_across_batch=r_actions,
            passed_regression_gate=True,
            run_dirs=r_runs,
        )
        tree.add_node(root_node)
        print(f"[+] Gen 0 Root: Batch Score={r_mean:.4f}, Actions={r_actions}, Tiebreaker={r_tiebreaker:.2f}")
        parent_digests = r_digests
    else:
        print(f"[+] Generation 0 already evaluated in checkpoint. Skipping re-evaluation.")
        best_so_far = tree.select_best_frontier_node()
        digests = []
        for g, rdir in best_so_far.run_dirs.items():
            rd = Path(rdir)
            if rd.exists():
                d = TraceDigester.digest_run(rd)
                digests.append(TraceDigester.render_digest_markdown(d))
        parent_digests = "\n\n---\n\n".join(digests)

    # 2. Iterate Generations
    start_gen = tree.get_max_generation() + 1
    if start_gen > generations:
        print(f"[*] Checkpoint already contains {tree.get_max_generation()} generations (meets or exceeds target {generations}).")

    for gen in range(start_gen, generations + 1):
        parent_node = tree.select_best_frontier_node()
        print(f"\n[*] --- Generation {gen}/{generations} ---")
        print(f"[*] Parent Node: `{parent_node.node_id}` (Score={parent_node.mean_batch_score:.4f})")

        parent_skill_md = Path(parent_node.skill_path).read_text(encoding="utf-8")
        parent_policy_md = Path(parent_node.learner_policy_path).read_text(encoding="utf-8")

        candidates = learner.propose_offspring(
            parent_skill_markdown=parent_skill_md,
            learner_policy_markdown=parent_policy_md,
            batch_digests_markdown=parent_digests,
            num_offspring=branch_factor,
            generation=gen,
        )

        accepted_this_gen = 0
        best_child_digest = ""

        for idx, cand in enumerate(candidates):
            cid = f"gen{gen}_v{idx+1}"
            child_skill_path = skills_dir / f"{cid}.md"
            child_skill_path.write_text(cand.mutated_skill_markdown, encoding="utf-8")

            child_policy_path = skills_dir / f"{cid}_policy.md"
            if cand.mutated_learner_policy_markdown:
                child_policy_path.write_text(cand.mutated_learner_policy_markdown, encoding="utf-8")
            else:
                shutil.copy(parent_node.learner_policy_path, child_policy_path)

            print(f"\n  [>] Evaluating Offspring `{cid}`...")
            print(f"      Rationale: {cand.mutation_rationale}")

            (
                c_scores,
                c_levels,
                c_mean,
                c_tiebreaker,
                c_actions,
                c_runs,
                c_digests,
            ) = evaluate_skill_on_batch(
                skill_path=child_skill_path,
                games=games,
                eval_id=cid,
                max_steps=max_steps,
                inner_model=inner_model,
                base_url=base_url,
                api_key=api_key,
                is_mock=is_mock,
                is_dry_run=is_dry_run,
                artifact_root=runs_dir,
                workers=workers,
            )

            child_node = SkillNode(
                node_id=cid,
                parent_id=parent_node.node_id,
                generation=gen,
                skill_path=str(child_skill_path),
                learner_policy_path=str(child_policy_path),
                mutation_rationale=cand.mutation_rationale,
                game_scores=c_scores,
                level_scores_by_game=c_levels,
                mean_batch_score=c_mean,
                tiebreaker_score=c_tiebreaker,
                total_actions_across_batch=c_actions,
                passed_regression_gate=False,
                run_dirs=c_runs,
            )

            # Check regression gate against parent
            passed, reason = tree.check_regression_gate(parent_node, child_node)
            child_node.passed_regression_gate = passed
            child_node.rejection_reason = reason if not passed else None

            tree.add_node(child_node)

            if passed:
                accepted_this_gen += 1
                best_child_digest = c_digests
                print(f"      Status: [PASS] ACCEPTED ({reason}) | Score={c_mean:.4f}, Actions={c_actions}, Tiebreaker={c_tiebreaker:.2f}")
            else:
                print(f"      Status: [FAIL] REJECTED ({reason}) | Score={c_mean:.4f}, Actions={c_actions}, Tiebreaker={c_tiebreaker:.2f}")

        if best_child_digest:
            parent_digests = best_child_digest

    # 3. Finalize and Export Best Skill
    winning_node = tree.select_best_frontier_node()
    best_skill_file = exp_path / "best_SKILL.md"
    shutil.copy(winning_node.skill_path, best_skill_file)

    report = tree.render_lineage_report()
    report_file = exp_path / "lineage_report.md"
    report_file.write_text(report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    print(f"\n[+] Evolution complete! Winning skill saved to: {best_skill_file}")
    print(f"[+] Full tree persisted to: {tree.tree_file}")
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Milestone 2 & 3 Evolutionary Self-Improvement Loop")
    parser.add_argument("--games", type=str, default="ls20", help="Comma-separated game IDs (e.g. ls20 or mock_1,mock_2)")
    parser.add_argument("--generations", type=int, default=2, help="Number of evolutionary generations")
    parser.add_argument("--branch-factor", type=int, default=2, help="Offspring per generation K")
    parser.add_argument("--max-steps", type=int, default=30, help="Max env steps per game")
    parser.add_argument("--inner-model", type=str, default=os.getenv("INNER_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--outer-model", type=str, default=os.getenv("OUTER_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--mock", action="store_true", help="Run with deterministic MockArcEnvBridge")
    parser.add_argument("--dry-run", action="store_true", help="Run without external LLM calls")
    parser.add_argument("--workers", type=int, default=2, help="Parallel worker threads for batch evaluation")
    parser.add_argument("--resume", type=Path, default=None, help="Resume evolution from an existing experiment directory")
    parser.add_argument("--skill", type=Path, default=Path("skills/SKILL.md"))
    parser.add_argument("--policy", type=Path, default=Path("skills/L2_learner_policy.md"))
    parser.add_argument("--exp-dir", type=Path, default=Path("evolution_runs"))
    args = parser.parse_args()

    game_list = [g.strip() for g in args.games.split(",") if g.strip()]
    has_api_key = bool(args.api_key or os.getenv("OPENAI_API_KEY"))
    if not has_api_key and not args.base_url and not args.dry_run:
        print("[!] Note: No OPENAI_API_KEY or OPENAI_BASE_URL detected. Defaulting to zero-cost DRY-RUN mode.")
        print("[!] To execute with live LLMs: export OPENAI_API_KEY='...' (or pass --api-key).")
    is_dry_run = args.dry_run or (not has_api_key and not args.base_url)

    run_evolution(
        games=game_list,
        generations=args.generations,
        branch_factor=args.branch_factor,
        max_steps=args.max_steps,
        seed_skill_path=args.skill,
        seed_policy_path=args.policy,
        inner_model=args.inner_model,
        outer_model=args.outer_model,
        base_url=args.base_url,
        api_key=args.api_key,
        is_mock=args.mock,
        is_dry_run=is_dry_run,
        workers=args.workers,
        resume_path=args.resume,
        exp_dir=args.exp_dir,
    )


if __name__ == "__main__":
    main()
