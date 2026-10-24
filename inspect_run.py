import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.repl_session import ARC_PALETTE_RGB
from src.evolution_tree import EvolutionTree, SkillNode


def render_run_gif(
    run_dir: Path,
    output_gif: Optional[Path] = None,
    scale: int = 4,
    fps: int = 4,
) -> Path:
    """
    Renders an animated GIF replay of an episode with HUD telemetry.
    Loads trajectory_frames.npz, trajectory.jsonl, and summary.json.
    """
    run_dir = Path(run_dir)
    npz_path = run_dir / "trajectory_frames.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No trajectory_frames.npz found in {run_dir}")

    data = np.load(npz_path)
    settled_frames = data["settled_frames"]  # shape (N, 64, 64)
    total_frames = len(settled_frames)

    # Load summary and trajectory if available
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    game_id = summary.get("game_id", "unknown")

    step_events: List[Dict[str, Any]] = []
    trace_path = run_dir / "trajectory.jsonl"
    if trace_path.exists():
        try:
            with trace_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        evt = json.loads(line)
                        if evt.get("event") == "env_step":
                            step_events.append(evt)
        except Exception:
            pass

    # Canvas dimensions
    grid_size = 64 * scale
    hud_width = 240
    canvas_w = grid_size + hud_width
    canvas_h = max(grid_size, 256)

    gif_images: List[Image.Image] = []
    font = ImageFont.load_default()

    for idx in range(total_frames):
        frame_grid = settled_frames[idx]
        rgb_grid = ARC_PALETTE_RGB[np.clip(frame_grid, 0, 15)]  # (64, 64, 3)
        img_grid = Image.fromarray(rgb_grid.astype(np.uint8)).resize(
            (grid_size, grid_size),
            resample=Image.NEAREST,
        )

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 22, 28))
        canvas.paste(img_grid, (0, 0))

        # Divider line
        draw = ImageDraw.Draw(canvas)
        draw.line([(grid_size, 0), (grid_size, canvas_h)], fill=(50, 55, 65), width=2)

        # Retrieve step metadata
        if idx == 0:
            step_num = 0
            action_name = "INITIAL_STATE"
            status = "NOT_FINISHED"
            level = 0
            via_macro = False
            grid_changed = True
            thought = "Episode start"
            num_proven = 0
            num_hypo = 0
            num_fatal = 0
        else:
            evt_idx = idx - 1
            if evt_idx < len(step_events):
                evt = step_events[evt_idx]
                step_num = evt.get("env_steps_total", idx)
                payload = evt.get("payload", {})
                action_name = payload.get("action", "STEP")
                status = evt.get("state_status", "NOT_FINISHED")
                level = evt.get("levels_completed", 0)
                via_macro = payload.get("via_macro", False)
                grid_changed = payload.get("grid_changed", True)
                thought = payload.get("thought", "")
                wm = evt.get("world_model_snapshot", {})
                num_proven = len(wm.get("proven", {}))
                num_hypo = len(wm.get("hypotheses", {}))
                num_fatal = len(wm.get("fatal", []))
            else:
                step_num = idx
                action_name = "STEP"
                status = "NOT_FINISHED"
                level = 0
                via_macro = False
                grid_changed = True
                thought = ""
                num_proven = 0
                num_hypo = 0
                num_fatal = 0

        # Draw HUD text
        x_text = grid_size + 12
        y_text = 10
        line_height = 14

        draw.text((x_text, y_text), "ARC-AGI-3 REPLAY", fill=(80, 200, 255), font=font)
        y_text += line_height + 4

        draw.text((x_text, y_text), f"Game: {game_id}", fill=(220, 220, 220), font=font)
        y_text += line_height

        draw.text((x_text, y_text), f"Frame: {idx + 1}/{total_frames}", fill=(180, 180, 180), font=font)
        y_text += line_height

        draw.text((x_text, y_text), f"Step: {step_num}", fill=(255, 255, 255), font=font)
        y_text += line_height

        draw.text((x_text, y_text), f"Level: {level}", fill=(255, 215, 0), font=font)
        y_text += line_height + 4

        # Action tag
        act_label = f"Macro: {action_name}" if via_macro else f"Action: {action_name}"
        act_color = (130, 255, 130) if via_macro else (255, 180, 50)
        draw.text((x_text, y_text), act_label, fill=act_color, font=font)
        y_text += line_height

        # Grid changed status
        gc_text = "Grid Changed: YES" if grid_changed else "Grid Changed: NO (Wall)"
        gc_color = (100, 220, 100) if grid_changed else (255, 100, 100)
        draw.text((x_text, y_text), gc_text, fill=gc_color, font=font)
        y_text += line_height

        # Status tag
        status_color = (100, 255, 100) if status == "WIN" else ((255, 80, 80) if status == "GAME_OVER" else (160, 200, 255))
        draw.text((x_text, y_text), f"Status: {status}", fill=status_color, font=font)
        y_text += line_height + 6

        # L0 World Model telemetry
        draw.text((x_text, y_text), "World Model State:", fill=(200, 160, 255), font=font)
        y_text += line_height
        draw.text((x_text + 6, y_text), f"* Proven: {num_proven}", fill=(180, 255, 180), font=font)
        y_text += line_height
        draw.text((x_text + 6, y_text), f"* Hypotheses: {num_hypo}", fill=(255, 255, 150), font=font)
        y_text += line_height
        draw.text((x_text + 6, y_text), f"* Fatal Rules: {num_fatal}", fill=(255, 150, 150), font=font)
        y_text += line_height + 6

        # Thought snippet
        if thought:
            draw.text((x_text, y_text), "Thought:", fill=(140, 140, 140), font=font)
            y_text += line_height
            words = thought.split()
            thought_line = ""
            for w in words:
                if len(thought_line) + len(w) + 1 > 28:
                    draw.text((x_text, y_text), thought_line, fill=(180, 180, 180), font=font)
                    y_text += line_height
                    thought_line = w
                    if y_text > canvas_h - 16:
                        break
                else:
                    thought_line = f"{thought_line} {w}".strip()
            if thought_line and y_text <= canvas_h - 16:
                draw.text((x_text, y_text), thought_line, fill=(180, 180, 180), font=font)

        gif_images.append(canvas)

    if not output_gif:
        output_gif = run_dir / "replay.gif"
    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)

    if gif_images:
        gif_images[0].save(
            output_gif,
            save_all=True,
            append_images=gif_images[1:],
            duration=max(50, int(1000 / fps)),
            loop=0,
        )

    return output_gif


def render_experiment_benchmark(exp_dir: Path, output_md: Optional[Path] = None) -> str:
    """
    Renders a comprehensive GitHub-ready Markdown benchmark table comparing
    the Gen 0 baseline root skill against the winning frontier skill.
    """
    exp_dir = Path(exp_dir)
    tree_file = exp_dir / "lineage_tree.json"
    if not tree_file.exists():
        raise FileNotFoundError(f"No lineage_tree.json found in {exp_dir}")

    tree = EvolutionTree.load_from_checkpoint(exp_dir)
    root_node = tree.nodes.get(tree.root_node_id or "gen0_root")
    best_node = tree.select_best_frontier_node()

    if not root_node:
        raise ValueError("Could not find Gen 0 root node in lineage tree.")

    lines = [
        f"# ARC-AGI-3 RSI Benchmark Report",
        f"**Experiment ID**: `{tree.experiment_id}`",
        f"**Root Baseline Node**: `{root_node.node_id}`",
        f"**Winning Frontier Node**: `{best_node.node_id}`",
        "",
        "## Performance Comparison: Gen 0 vs Winning Skill",
        "",
        "| Game ID | Gen 0 Score | Frontier Score | Score Delta | Gen 0 Actions | Frontier Actions | Action Delta |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    all_games = sorted(set(list(root_node.game_scores.keys()) + list(best_node.game_scores.keys())))
    total_root_actions = root_node.total_actions_across_batch
    total_best_actions = best_node.total_actions_across_batch

    for game in all_games:
        r_score = root_node.game_scores.get(game, 0.0)
        b_score = best_node.game_scores.get(game, 0.0)
        score_diff = b_score - r_score
        score_delta_str = f"+{score_diff:.2f}" if score_diff >= 0 else f"{score_diff:.2f}"

        # Estimate actions per game from run_dirs or total
        r_run = root_node.run_dirs.get(game)
        b_run = best_node.run_dirs.get(game)

        r_acts = "-"
        if r_run and (Path(r_run) / "summary.json").exists():
            try:
                s = json.loads((Path(r_run) / "summary.json").read_text(encoding="utf-8"))
                r_acts = str(s.get("total_env_steps", "-"))
            except Exception:
                pass

        b_acts = "-"
        if b_run and (Path(b_run) / "summary.json").exists():
            try:
                s = json.loads((Path(b_run) / "summary.json").read_text(encoding="utf-8"))
                b_acts = str(s.get("total_env_steps", "-"))
            except Exception:
                pass

        act_delta_str = "-"
        if r_acts.isdigit() and b_acts.isdigit():
            diff = int(b_acts) - int(r_acts)
            act_delta_str = f"{diff:+d}" if diff != 0 else "0"

        lines.append(
            f"| `{game}` | {r_score:.2f} | {b_score:.2f} | **{score_delta_str}** | "
            f"{r_acts} | {b_acts} | {act_delta_str} |"
        )

    # Batch summary
    score_change = best_node.mean_batch_score - root_node.mean_batch_score
    score_pct = (score_change / max(0.001, root_node.mean_batch_score)) * 100.0 if root_node.mean_batch_score > 0 else 0.0
    action_change = total_best_actions - total_root_actions

    lines.extend([
        "",
        "### Aggregate Summary",
        f"- **Mean Batch Score**: `{root_node.mean_batch_score:.4f}` -> `{best_node.mean_batch_score:.4f}` ({score_pct:+.1f}%)",
        f"- **Tiebreaker Score**: `{root_node.tiebreaker_score:.2f}` -> `{best_node.tiebreaker_score:.2f}`",
        f"- **Total Environment Steps**: `{total_root_actions}` -> `{total_best_actions}` ({action_change:+d} actions)",
        f"- **Winning Skill File**: [`{best_node.skill_path}`]({best_node.skill_path})",
        "",
        "## Evolutionary Candidate Lineage",
        "",
        "| Node ID | Gen | Parent | Batch Score | Tiebreaker | Actions | Gate Status | Mutation Rationale |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
    ])

    for node in sorted(tree.nodes.values(), key=lambda n: (n.generation, n.node_id)):
        status_tag = "[PASS] ACCEPTED" if node.passed_regression_gate else f"[FAIL] REJECTED"
        if node.parent_id is None:
            status_tag = "[ROOT]"
        lines.append(
            f"| `{node.node_id}` | {node.generation} | `{node.parent_id or '-'}` | "
            f"{node.mean_batch_score:.4f} | {node.tiebreaker_score:.2f} | {node.total_actions_across_batch} | "
            f"{status_tag} | {node.mutation_rationale[:65]}... |"
        )

    markdown_content = "\n".join(lines)

    if not output_md:
        output_md = exp_dir / "BENCHMARK.md"
    Path(output_md).write_text(markdown_content, encoding="utf-8")
    return markdown_content


def main() -> None:
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Visual Trajectory & Benchmark Inspector")
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to single run directory to render replay.gif")
    parser.add_argument("--exp-dir", type=Path, default=None, help="Path to experiment directory to benchmark and render")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for 64x64 grid (default: 4 -> 256x256)")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for replay.gif (default: 4)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not (run_dir / "trajectory_frames.npz").exists():
            print(f"[-] Error: No trajectory_frames.npz found in {run_dir}. Run the episode with TaskAgent to record frames.")
        else:
            gif_path = render_run_gif(run_dir=run_dir, scale=args.scale, fps=args.fps)
            print(f"[+] Replay GIF generated: {gif_path}")

    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        bm_md = render_experiment_benchmark(exp_dir=exp_dir)
        print(f"[+] Benchmark report generated: {exp_dir / 'BENCHMARK.md'}")
        print("\n" + bm_md + "\n")

        # Also render replay.gif for each run found in exp_dir / "runs"
        runs_dir = exp_dir / "runs"
        if runs_dir.exists():
            for rdir in runs_dir.iterdir():
                if rdir.is_dir() and (rdir / "trajectory_frames.npz").exists():
                    try:
                        gpath = render_run_gif(run_dir=rdir, scale=args.scale, fps=args.fps)
                        print(f"    [+] Rendered replay GIF: {gpath}")
                    except Exception as e:
                        print(f"    [-] Could not render GIF for {rdir.name}: {e}")

    if not args.run_dir and not args.exp_dir:
        parser.print_help()


if __name__ == "__main__":
    main()
