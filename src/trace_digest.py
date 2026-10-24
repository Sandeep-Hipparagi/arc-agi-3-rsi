import json
from pathlib import Path
from typing import Dict, List, Any, Optional


class TraceDigester:
    """
    Deterministically digests raw run artifacts (`summary.json` and `trajectory.jsonl`)
    into a high-signal, compact (<3,000 tokens) Diagnostic Digest for the Learner Agent.
    """

    @staticmethod
    def digest_run(run_dir: Path) -> Dict[str, Any]:
        summary_path = run_dir / "summary.json"
        trajectory_path = run_dir / "trajectory.jsonl"

        summary: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        events: List[Dict[str, Any]] = []
        if trajectory_path.exists():
            with trajectory_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except Exception:
                            continue

        # 1. Extract Scorecard Telemetry
        official_sc = summary.get("official_scorecard", {})
        game_id = summary.get("game_id", run_dir.name)
        run_id = summary.get("run_id", run_dir.name)
        final_status = summary.get("final_status", "UNKNOWN")
        levels_completed = summary.get("levels_completed", 0)
        total_env_steps = summary.get("total_env_steps", 0)

        level_scores: List[float] = []
        level_actions: List[int] = []
        level_baselines: List[int] = []
        resets_count = 0
        overall_score = 0.0

        if official_sc and "environments" in official_sc and official_sc["environments"]:
            env_data = official_sc["environments"][0]
            runs = env_data.get("runs", [])
            overall_score = float(official_sc.get("score", 0.0))
            if runs:
                latest_run = runs[-1]
                level_scores = latest_run.get("level_scores", [])
                level_actions = latest_run.get("level_actions", [])
                level_baselines = latest_run.get("level_baseline_actions", [])
                resets_count = latest_run.get("resets", 0)

        # Fallback if scorecard is empty or synthetic
        if not level_scores and "total_env_steps" in summary:
            overall_score = 1.0 if levels_completed > 0 else 0.0

        # 2. Action Hygiene Metrics
        env_step_events = [e for e in events if e.get("event") == "env_step"]
        repl_events = [e for e in events if e.get("event") == "repl_exec"]
        wm_events = [e for e in events if e.get("event") == "update_world_model"]

        noop_steps = 0
        cyclic_revisits = 0
        repl_errors: List[str] = []

        # Track per-level seen state hashes for cycle detection
        seen_hashes_by_level: Dict[int, set] = {}

        for e in env_step_events:
            payload = e.get("payload", {})
            lvl = e.get("levels_completed", 0)
            if lvl not in seen_hashes_by_level:
                seen_hashes_by_level[lvl] = set()

            grid_changed = payload.get("grid_changed", None)
            state_hash = payload.get("state_hash", "")
            prev_state_hash = payload.get("prev_state_hash", "")

            # No-op check (wall bump / stationary)
            if grid_changed is False or (state_hash and prev_state_hash and state_hash == prev_state_hash):
                noop_steps += 1

            # Cycle check (revisiting previously seen state within same level)
            if state_hash:
                if state_hash in seen_hashes_by_level[lvl]:
                    cyclic_revisits += 1
                else:
                    seen_hashes_by_level[lvl].add(state_hash)

        # REPL errors check
        for r in repl_events:
            out = r.get("payload", {}).get("output", "")
            if "REPL Execution Error" in out or "Traceback (most recent call last)" in out:
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                err_msg = lines[-1] if lines else "Unknown REPL error"
                if err_msg not in repl_errors:
                    repl_errors.append(err_msg)

        # 3. Identify Inefficiency Clusters (>= 3 consecutive no-ops or cyclic revisits)
        waste_clusters: List[Dict[str, Any]] = []
        current_cluster: List[Dict[str, Any]] = []

        for e in env_step_events:
            p = e.get("payload", {})
            is_waste = (p.get("grid_changed") is False) or (p.get("state_hash") and p.get("state_hash") == p.get("prev_state_hash"))
            if is_waste:
                current_cluster.append(e)
            else:
                if len(current_cluster) >= 3:
                    waste_clusters.append({
                        "count": len(current_cluster),
                        "level": current_cluster[0].get("levels_completed", 0),
                        "actions": [c.get("payload", {}).get("action") for c in current_cluster],
                        "thoughts": [c.get("payload", {}).get("thought") for c in current_cluster[:3]],
                    })
                current_cluster = []

        if len(current_cluster) >= 3:
            waste_clusters.append({
                "count": len(current_cluster),
                "level": current_cluster[0].get("levels_completed", 0),
                "actions": [c.get("payload", {}).get("action") for c in current_cluster],
                "thoughts": [c.get("payload", {}).get("thought") for c in current_cluster[:3]],
            })

        # 4. Defeat / Reset Precursors
        defeat_windows: List[Dict[str, Any]] = []
        for i, e in enumerate(events):
            status = e.get("state_status", "")
            act = e.get("payload", {}).get("action", "")
            if status == "GAME_OVER" or act == "RESET":
                precursor_events = events[max(0, i - 2) : i + 1]
                defeat_windows.append({
                    "event_index": i,
                    "status": status,
                    "action": act,
                    "precursor_actions": [ev.get("payload", {}).get("action") for ev in precursor_events if ev.get("event") == "env_step"],
                    "precursor_thoughts": [ev.get("payload", {}).get("thought") for ev in precursor_events if ev.get("event") == "env_step"],
                })

        # 5. Initialization Window
        initial_turns: List[Dict[str, Any]] = []
        for e in events[:4]:
            initial_turns.append({
                "event": e.get("event"),
                "summary": e.get("payload", {}).get("summary", ""),
                "action": e.get("payload", {}).get("action", ""),
                "code": (e.get("payload", {}).get("code", "")[:120] + "...") if len(e.get("payload", {}).get("code", "")) > 120 else e.get("payload", {}).get("code", ""),
            })

        # 6. World Model Snapshot
        final_wm = summary.get("final_world_model", {})
        if not final_wm and events:
            final_wm = events[-1].get("world_model_snapshot", {})

        return {
            "run_id": run_id,
            "game_id": game_id,
            "final_status": final_status,
            "levels_completed": levels_completed,
            "total_env_steps": total_env_steps or len(env_step_events),
            "overall_score": overall_score,
            "scorecard": {
                "level_scores": level_scores,
                "level_actions": level_actions,
                "level_baselines": level_baselines,
                "resets": resets_count,
            },
            "hygiene": {
                "total_repl_calls": len(repl_events),
                "total_world_model_updates": len(wm_events),
                "noop_steps": noop_steps,
                "cyclic_revisits": cyclic_revisits,
                "repl_error_count": len(repl_errors),
                "unique_repl_errors": repl_errors,
            },
            "initial_turns": initial_turns,
            "waste_clusters": waste_clusters,
            "defeat_windows": defeat_windows,
            "final_world_model": {
                "proven_mechanics": final_wm.get("proven_mechanics", final_wm.get("proven", {})),
                "unverified_hypotheses": final_wm.get("unverified_hypotheses", final_wm.get("hypotheses", {})),
                "fatal_transitions": final_wm.get("fatal_transitions", final_wm.get("fatal", [])),
            },
        }

    @classmethod
    def render_digest_markdown(cls, digest: Dict[str, Any]) -> str:
        gid = digest.get("game_id", "Unknown")
        rid = digest.get("run_id", "Unknown")
        sc = digest.get("scorecard", {})
        hygiene = digest.get("hygiene", {})
        wm = digest.get("final_world_model", {})

        lines = [
            f"# Diagnostic Digest: `{gid}` (Run: `{rid}`)",
            "",
            "## 1. Scorecard Telemetry",
            f"- **Final Status**: `{digest.get('final_status')}`",
            f"- **Levels Completed**: `{digest.get('levels_completed')}`",
            f"- **Total Env Steps**: `{digest.get('total_env_steps')}`",
            f"- **Official Score**: `{digest.get('overall_score')}`",
            f"- **Resets**: `{sc.get('resets', 0)}`",
            "",
        ]

        # Level Breakdown Table
        level_scores = sc.get("level_scores", [])
        level_actions = sc.get("level_actions", [])
        level_baselines = sc.get("level_baselines", [])
        if level_scores or level_actions:
            lines.append("| Level | Steps Taken | Human Baseline | Ratio | Score |")
            lines.append("| :---: | :---: | :---: | :---: | :---: |")
            max_lvls = max(len(level_scores), len(level_actions), len(level_baselines))
            for i in range(max_lvls):
                act = level_actions[i] if i < len(level_actions) else "-"
                base = level_baselines[i] if i < len(level_baselines) else "-"
                score = level_scores[i] if i < len(level_scores) else 0.0
                ratio_str = f"{(act / base):.2f}x" if isinstance(act, (int, float)) and isinstance(base, (int, float)) and base > 0 else "-"
                lines.append(f"| L{i+1} | {act} | {base} | {ratio_str} | {score} |")
            lines.append("")

        # Action Hygiene Metrics
        lines.extend([
            "## 2. Action Hygiene & Inefficiency Metrics",
            f"- **Billed Env Steps**: {digest.get('total_env_steps')}",
            f"- **Free REPL Calls**: {hygiene.get('total_repl_calls', 0)}",
            f"- **Free WM Updates**: {hygiene.get('total_world_model_updates', 0)}",
            f"- **No-Op Bumps (grid_changed=False)**: {hygiene.get('noop_steps', 0)}",
            f"- **Cyclic State Revisits**: {hygiene.get('cyclic_revisits', 0)}",
            f"- **REPL Python Errors**: {hygiene.get('repl_error_count', 0)}",
        ])

        if hygiene.get("unique_repl_errors"):
            lines.append("  - Deduped Errors:")
            for err in hygiene["unique_repl_errors"][:3]:
                lines.append(f"    - `{err}`")
        lines.append("")

        # Waste Clusters
        clusters = digest.get("waste_clusters", [])
        if clusters:
            lines.append("## 3. Detected Waste Clusters (>=3 repeated/no-op moves)")
            for c in clusters[:3]:
                lines.append(f"- **Level {c.get('level')}**: {c.get('count')} consecutive wasted moves: `{c.get('actions')}`")
            lines.append("")

        # Defeats / Resets
        defeats = digest.get("defeat_windows", [])
        if defeats:
            lines.append("## 4. Defeat & Reset Transitions")
            for d in defeats[:3]:
                lines.append(f"- Status `{d.get('status')}` after action `{d.get('action')}` (Precursor moves: `{d.get('precursor_actions')}`)")
            lines.append("")

        # World Model Final State
        lines.extend([
            "## 5. Final Two-Stage World Model State",
            f"- **Proven Mechanics ({len(wm.get('proven_mechanics', {}))} rules)**:",
        ])
        if wm.get("proven_mechanics"):
            for k, v in list(wm["proven_mechanics"].items())[:5]:
                lines.append(f"  - `{k}`: {v}")
        else:
            lines.append("  - *None verified.*")

        lines.append(f"- **Unverified Hypotheses ({len(wm.get('unverified_hypotheses', {}))} items)**:")
        if wm.get("unverified_hypotheses"):
            for k, v in list(wm["unverified_hypotheses"].items())[:5]:
                lines.append(f"  - `{k}`: {v}")
        else:
            lines.append("  - *None pending.*")

        lines.append(f"- **Fatal Transitions ({len(wm.get('fatal_transitions', []))} rules)**:")
        if wm.get("fatal_transitions"):
            for f in wm["fatal_transitions"][:3]:
                lines.append(f"  - ⚠️ {f}")
        else:
            lines.append("  - *None recorded.*")
        lines.append("")

        # Primary Bottleneck Diagnosis
        lines.append("## 6. Primary Bottleneck Diagnosis")
        noops = hygiene.get("noop_steps", 0)
        cycles = hygiene.get("cyclic_revisits", 0)
        resets = sc.get("resets", 0)

        diagnoses = []
        if noops >= 3:
            diagnoses.append("⚠️ **Failure Mode 1 (Wall Bumping)**: High stationary steps detected. Recommend checking obstacle masks in `repl_exec` before stepping.")
        if resets > 0:
            diagnoses.append("⚠️ **Failure Mode 4 (Fatal State)**: Game over / reset triggered. Recommend enforcing `add_fatal` and immediate pre-condition verification.")
        if cycles >= 4:
            diagnoses.append("⚠️ **Failure Mode 3 (Slow Exploitation / Cyclic Thrashing)**: Revisiting identical states. Recommend full BFS route computation in REPL.")
        if not diagnoses:
            diagnoses.append("✅ **Clean Execution**: Low wasted steps. Target optimizing route lengths against human baseline.")

        lines.extend(diagnoses)
        return "\n".join(lines)
