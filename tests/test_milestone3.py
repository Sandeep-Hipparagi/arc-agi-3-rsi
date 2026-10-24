import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image
import pytest
import httpx
import openai

from src.repl_session import RLMReplSession
from src.task_memory import TwoStageWorldModel
from src.env_bridge import ArcEnvBridge, MockArcEnvBridge, StepOutcome
from src.task_agent import TaskAgent, _MockResponse, _MockChoice, _MockMessage, _MockToolCall
from src.learner_agent import LearnerAgent, SkillCandidate
from src.evolution_tree import EvolutionTree, SkillNode
from evolve import run_evolution, evaluate_skill_on_batch
from inspect_run import render_run_gif, render_experiment_benchmark


class WallBumpEnvBridge(MockArcEnvBridge):
    """
    Mock bridge that simulates a static wall bump where the grid does NOT change.
    """
    def step(self, action_name: str, x=None, y=None, thought: str = "") -> StepOutcome:
        self.env_steps_total += 1
        self.env_steps_current_level += 1
        # Player and enemy stay static; grid does not change at all
        frame = self._render_frame()
        return StepOutcome(
            frames=[frame, frame],
            state_status="NOT_FINISHED",
            levels_completed=self.levels_completed,
            available_actions=self.get_available_actions(),
            env_steps_total=self.env_steps_total,
            env_steps_current_level=self.env_steps_current_level,
        )


def test_env_step_macro_execution_and_early_stop(tmp_path: Path):
    """
    Verifies that env_step_macro executes multi-step trajectories with accurate
    per-step billing, trace tagging (via_macro=True), and early-stop guardrails
    for both level advancement and no-op wall bumps.
    """
    # 1. Test Level Advance Early Stop
    # In MockArcEnvBridge, player starts at [10, 10] and goal is at [10, 14] (4 steps right)
    env_bridge = MockArcEnvBridge(game_id="mock_macro_advance")
    run_dir = tmp_path / "macro_run_advance"
    run_dir.mkdir(parents=True, exist_ok=True)
    repl = RLMReplSession(artifact_dir=run_dir / "frames")
    world_model = TwoStageWorldModel()

    trace_events = []
    def trace_cb(evt):
        trace_events.append(evt)

    skill_md = Path("skills/SKILL.md").read_text(encoding="utf-8")
    agent = TaskAgent(
        skill_markdown=skill_md,
        repl=repl,
        world_model=world_model,
        env_bridge=env_bridge,
        trace_callback=trace_cb,
        dry_run=True,
    )
    agent.run_dir = run_dir

    # Craft an LLM completion that calls env_step_macro with 6 ACTION4 moves
    macro_call = _MockResponse(_MockChoice(_MockMessage(
        content="Executing 6-step traversal.",
        tool_calls=[
            _MockToolCall(
                "call_macro_1",
                "env_step_macro",
                {"actions": ["ACTION4"] * 6, "stop_on_noop": True, "thought": "Reach goal"}
            )
        ]
    )))

    def mock_advance(turn, outcome):
        if turn == 1:
            return macro_call
        # End episode on turn 2
        outcome.state_status = "WIN"
        return _MockResponse(_MockChoice(_MockMessage(content="Level completed, halting.", tool_calls=[])))

    with patch.object(agent, "_get_dry_run_completion", side_effect=mock_advance):
        outcome = agent.run_episode(max_env_steps=10)

    # Player reaches goal at step 4, advancing level from 0 -> 1
    # Macro must early-stop at step 4, NOT executing steps 5 and 6!
    assert outcome.levels_completed == 1
    assert outcome.env_steps_total == 4

    macro_traces = [e for e in trace_events if e.get("event") == "env_step"]
    assert len(macro_traces) == 4
    for idx, trace in enumerate(macro_traces):
        payload = trace["payload"]
        assert payload["via_macro"] is True
        assert payload["macro_index"] == idx
        assert payload["action"] == "ACTION4"

    # 2. Test No-Op Wall Bump Early Stop
    bump_bridge = WallBumpEnvBridge(game_id="mock_wall_bump")
    bump_run_dir = tmp_path / "macro_run_bump"
    bump_run_dir.mkdir(parents=True, exist_ok=True)
    bump_repl = RLMReplSession(artifact_dir=bump_run_dir / "frames")
    bump_agent = TaskAgent(
        skill_markdown=skill_md,
        repl=bump_repl,
        world_model=TwoStageWorldModel(),
        env_bridge=bump_bridge,
        trace_callback=trace_cb,
        dry_run=True,
    )
    bump_agent.run_dir = bump_run_dir

    bump_macro_call = _MockResponse(_MockChoice(_MockMessage(
        content="Testing obstacle collision.",
        tool_calls=[
            _MockToolCall(
                "call_macro_bump",
                "env_step_macro",
                {"actions": ["ACTION1", "ACTION1", "ACTION1"], "stop_on_noop": True, "thought": "Bump wall"}
            )
        ]
    )))

    def mock_bump(turn, outcome):
        if turn == 1:
            return bump_macro_call
        outcome.state_status = "WIN"
        return _MockResponse(_MockChoice(_MockMessage(content="Wall bump tested, halting.", tool_calls=[])))

    trace_events.clear()
    with patch.object(bump_agent, "_get_dry_run_completion", side_effect=mock_bump):
        bump_outcome = bump_agent.run_episode(max_env_steps=10)

    # Because grid does not change, macro halts at sub-step 1!
    assert bump_outcome.env_steps_total == 1
    bump_traces = [e for e in trace_events if e.get("event") == "env_step"]
    assert len(bump_traces) == 1
    assert bump_traces[0]["payload"]["grid_changed"] is False


def test_trajectory_frames_npz_and_gif_generation(tmp_path: Path):
    """
    Verifies that TaskAgent persists trajectory_frames.npz containing (N, 64, 64) uint8
    tensors and inspect_run.render_run_gif renders an annotated animated GIF with HUD telemetry.
    """
    run_dir = tmp_path / "test_gif_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_file = run_dir / "trajectory.jsonl"

    def trace_cb(evt):
        with trace_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt) + "\n")

    repl = RLMReplSession(artifact_dir=run_dir / "frames")
    env_bridge = MockArcEnvBridge(game_id="mock_maze")
    skill_md = Path("skills/SKILL.md").read_text(encoding="utf-8")

    agent = TaskAgent(
        skill_markdown=skill_md,
        repl=repl,
        world_model=TwoStageWorldModel(),
        env_bridge=env_bridge,
        trace_callback=trace_cb,
        dry_run=True,
    )
    agent.run_dir = run_dir

    outcome = agent.run_episode(max_env_steps=6)

    # 1. Verify trajectory_frames.npz existence and tensor shape
    npz_path = run_dir / "trajectory_frames.npz"
    assert npz_path.exists(), "trajectory_frames.npz was not saved to run_dir"

    npz_data = np.load(npz_path)
    frames_tensor = npz_data["settled_frames"]
    assert frames_tensor.dtype == np.uint8
    assert frames_tensor.ndim == 3
    assert frames_tensor.shape[1:] == (64, 64)
    # Total frames = 1 initial frame + outcome.env_steps_total
    assert frames_tensor.shape[0] == outcome.env_steps_total + 1

    # Also save summary.json for telemetry
    summary = {
        "run_id": run_dir.name,
        "game_id": "mock_maze",
        "total_env_steps": outcome.env_steps_total,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    # 2. Render animated GIF
    gif_path = render_run_gif(run_dir=run_dir, scale=2, fps=4)
    assert gif_path.exists()
    assert gif_path.name == "replay.gif"

    # 3. Verify GIF properties with Pillow
    with Image.open(gif_path) as img:
        assert getattr(img, "is_animated", False)
        assert img.n_frames == frames_tensor.shape[0]
        # Width: 64 * scale (128) + HUD width (240) = 368
        assert img.width == 128 + 240
        assert img.height >= 128


def test_parallel_evolution_and_checkpoint_resume(tmp_path: Path):
    """
    Verifies multi-thread parallel evaluation across batch games, lineage tree persistence,
    and checkpoint resumption without re-evaluating completed nodes.
    """
    exp_dir = tmp_path / "parallel_evo_exp"
    games = ["mock_1", "mock_2"]

    # 1. Run 1 generation with 2 parallel workers
    tree_gen1 = run_evolution(
        games=games,
        generations=1,
        branch_factor=2,
        max_steps=8,
        is_mock=True,
        is_dry_run=True,
        workers=2,
        exp_dir=exp_dir,
    )

    assert tree_gen1.get_max_generation() == 1
    assert "gen0_root" in tree_gen1.nodes
    assert len(tree_gen1.nodes) == 3  # Root + 2 offspring

    tree_file = tree_gen1.experiment_dir / "lineage_tree.json"
    assert tree_file.exists()

    # 2. Resume experiment targeting 2 generations
    tree_gen2 = run_evolution(
        games=games,
        generations=2,
        branch_factor=2,
        max_steps=8,
        is_mock=True,
        is_dry_run=True,
        workers=2,
        resume_path=tree_gen1.experiment_dir,
        exp_dir=exp_dir,
    )

    assert tree_gen2.get_max_generation() == 2
    # Gen 0 was preserved and not re-evaluated; Gen 2 offspring were added
    assert len(tree_gen2.nodes) == 5  # Root + 2 (Gen 1) + 2 (Gen 2)

    # 3. Verify BENCHMARK.md generation
    bm_markdown = render_experiment_benchmark(exp_dir=tree_gen2.experiment_dir)
    bm_file = tree_gen2.experiment_dir / "BENCHMARK.md"
    assert bm_file.exists()
    assert "ARC-AGI-3 RSI Benchmark Report" in bm_markdown
    assert "mock_1" in bm_markdown
    assert "mock_2" in bm_markdown


def test_llm_retry_backoff_on_transient_error():
    """
    Verifies that _call_llm_with_retry in TaskAgent and LearnerAgent cleanly catches
    transient errors (RateLimitError, APIConnectionError), performs backoff retries,
    and succeeds once the transient condition clears.
    """
    req = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
    rate_limit_resp = httpx.Response(status_code=429, request=req)
    rate_limit_err = openai.RateLimitError(message="Rate limit exceeded", response=rate_limit_resp, body=None)

    # 1. Test TaskAgent retry recovery
    agent = TaskAgent(
        skill_markdown=Path("skills/SKILL.md").read_text(encoding="utf-8"),
        repl=MagicMock(),
        world_model=MagicMock(),
        env_bridge=MagicMock(),
        dry_run=True,
    )

    mock_fn = MagicMock(side_effect=[rate_limit_err, rate_limit_err, "SUCCESS_TASK"])

    with patch("time.sleep") as mock_sleep:
        res = agent._call_llm_with_retry(mock_fn, max_retries=3)

    assert res == "SUCCESS_TASK"
    assert mock_fn.call_count == 3
    assert mock_sleep.call_count == 2

    # 2. Test LearnerAgent retry recovery
    learner = LearnerAgent(dry_run=True)
    learner_fn = MagicMock(side_effect=[
        openai.APIConnectionError(request=req),
        "SUCCESS_LEARNER"
    ])

    with patch("time.sleep") as mock_sleep:
        res_learner = learner._call_llm_with_retry(learner_fn, max_retries=2)

    assert res_learner == "SUCCESS_LEARNER"
    assert learner_fn.call_count == 2
    assert mock_sleep.call_count == 1

    # 3. Test max_retries exhaustion
    exhaust_fn = MagicMock(side_effect=rate_limit_err)
    with patch("time.sleep"):
        with pytest.raises(openai.RateLimitError):
            agent._call_llm_with_retry(exhaust_fn, max_retries=2)

    assert exhaust_fn.call_count == 3  # initial + 2 retries
