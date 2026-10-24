import numpy as np
import pytest
from pathlib import Path

from src.repl_session import RLMReplSession
from src.task_memory import TwoStageWorldModel
from src.env_bridge import ArcEnvBridge, MockArcEnvBridge
from src.task_agent import TaskAgent


def test_repl_tensor_normalization(tmp_path: Path):
    """
    Feed 1D/2D/3D frame lists of lengths 1, 5, and 20 into RLMReplSession.sync_env_state
    and assert repl.namespace["state"].shape == (16, 64, 64) and dtype == np.int32.
    """
    repl = RLMReplSession(artifact_dir=tmp_path)

    # 1. Single 2D frame (shape 64, 64)
    frame_1 = np.ones((64, 64), dtype=np.int32) * 7
    repl.sync_env_state(raw_frames=[frame_1], action_name="ACTION1", level=0, status="NOT_FINISHED")
    state = repl.namespace["state"]
    assert state.shape == (16, 64, 64), f"Expected (16, 64, 64), got {state.shape}"
    assert state.dtype == np.int32
    assert repl.namespace["num_valid_frames"] == 1
    # Check that padded leading slots match the single frame
    assert np.all(state[0] == 7)
    assert np.all(state[-1] == 7)

    # 2. List of 5 frames (shape 5, 64, 64)
    frames_5 = [np.ones((64, 64), dtype=np.int32) * i for i in range(1, 6)]
    repl.sync_env_state(raw_frames=frames_5, action_name="ACTION2", level=0, status="NOT_FINISHED")
    state = repl.namespace["state"]
    assert state.shape == (16, 64, 64)
    assert state.dtype == np.int32
    assert repl.namespace["num_valid_frames"] == 5
    # First 11 frames should be padded with frames[0] (value 1)
    assert np.all(state[0] == 1)
    assert np.all(state[10] == 1)
    # The last 5 frames should be values 1..5, with state[-1] == 5
    assert np.all(state[11] == 1)
    assert np.all(state[12] == 2)
    assert np.all(state[15] == 5)

    # 3. List of 20 frames (shape 20, 64, 64) -> slices last 16
    frames_20 = [np.ones((64, 64), dtype=np.int32) * i for i in range(20)]
    repl.sync_env_state(raw_frames=frames_20, action_name="ACTION3", level=0, status="NOT_FINISHED")
    state = repl.namespace["state"]
    assert state.shape == (16, 64, 64)
    assert state.dtype == np.int32
    assert repl.namespace["num_valid_frames"] == 20
    # Last 16 frames should be 4..19
    assert np.all(state[0] == 4)
    assert np.all(state[-1] == 19)

    # 4. Nested 3D array or (1, 64, 64) shape (typical of arc_agi observations)
    frame_nested = np.ones((1, 64, 64), dtype=np.int8) * 3
    repl.sync_env_state(raw_frames=frame_nested, action_name="ACTION4", level=0, status="NOT_FINISHED")
    state = repl.namespace["state"]
    assert state.shape == (16, 64, 64)
    assert state.dtype == np.int32
    assert np.all(state[-1] == 3)


def test_render_frame_strip(tmp_path: Path):
    """
    Execute render_frame_strip("test_strip.png") inside repl.execute(), assert the PNG
    file exists, is non-empty, and pop_rendered_strip_base64() returns valid base64 once
    and None on the second call.
    """
    repl = RLMReplSession(artifact_dir=tmp_path)
    # Put distinct test values in state
    for i in range(16):
        repl.namespace["state"][i, :, :] = i % 16

    code = "render_frame_strip('test_strip.png', scale=2)"
    result = repl.execute(code)
    assert "Saved 16-frame composite strip" in result

    png_path = tmp_path / "test_strip.png"
    assert png_path.exists(), "Rendered strip file does not exist"
    assert png_path.stat().st_size > 0, "Rendered strip file is empty"

    # pop_rendered_strip_base64 returns valid base64 once
    b64 = repl.pop_rendered_strip_base64()
    assert b64 is not None
    assert isinstance(b64, str) and len(b64) > 100

    # Second call returns None
    b64_second = repl.pop_rendered_strip_base64()
    assert b64_second is None


def test_world_model_lifecycle():
    """
    Test adding a hypothesis, promoting it to proven (asserting removal from unverified),
    refuting a hypothesis, and adding a fatal transition.
    """
    wm = TwoStageWorldModel()

    # 1. Add hypothesis
    res1 = wm.update("add_hypothesis", "red_enemy", "Red entities cause game over")
    assert "red_enemy" in wm.unverified_hypotheses
    assert "Added/updated hypothesis [red_enemy]" in res1

    # 2. Promote to proven -> automatically removed from unverified
    res2 = wm.update("promote_to_proven", "red_enemy", "Red entities confirmed lethal on contact")
    assert "red_enemy" in wm.proven_mechanics
    assert "red_enemy" not in wm.unverified_hypotheses
    assert "Promoted [red_enemy] to Proven Mechanics" in res2

    # 3. Refute hypothesis
    wm.update("add_hypothesis", "green_enemy", "Green entities are harmful")
    assert "green_enemy" in wm.unverified_hypotheses
    res3 = wm.update("refute_hypothesis", "green_enemy", "")
    assert "green_enemy" not in wm.unverified_hypotheses
    assert "Refuted and removed hypothesis [green_enemy]" in res3

    # 4. Add fatal transition ("never make the same mistake twice")
    res4 = wm.update("add_fatal", "move_into_red", "Never move right into (20, 20) when red enemy is adjacent")
    assert len(wm.fatal_transitions) == 1
    assert "move_into_red" in wm.fatal_transitions[0]
    assert "Recorded fatal transition rule" in res4

    # Duplicate fatal transition should not duplicate
    wm.update("add_fatal", "move_into_red", "Never move right into (20, 20) when red enemy is adjacent")
    assert len(wm.fatal_transitions) == 1

    # Render markdown check
    md = wm.render_markdown()
    assert "Red entities confirmed lethal on contact" in md
    assert "Never move right into (20, 20)" in md
    assert "None active" in md


def test_scratchpad_atomic_trimming():
    """
    Simulate 20 turns of assistant multi-tool calls + tool responses + vision user messages,
    run _trim_scratchpad(), and assert that every role == "tool" message is preceded by its
    matching assistant tool_calls message.
    """
    messages = [
        {"role": "user", "content": "Episode initialized. Start with OBSERVE."}
    ]

    for turn in range(20):
        call_id_1 = f"call_{turn}_repl"
        call_id_2 = f"call_{turn}_step"

        # Assistant turn with 2 tool calls
        messages.append({
            "role": "assistant",
            "content": f"Turn {turn} analysis",
            "tool_calls": [
                {"id": call_id_1, "type": "function", "function": {"name": "repl_exec", "arguments": "{}"}},
                {"id": call_id_2, "type": "function", "function": {"name": "env_step", "arguments": "{}"}},
            ]
        })

        # Matching tool responses
        messages.append({"role": "tool", "tool_call_id": call_id_1, "content": f"Output {turn} repl"})
        messages.append({"role": "tool", "tool_call_id": call_id_2, "content": f"Output {turn} step"})

        # Vision attachment every alternate turn
        if turn % 2 == 0:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Visual strip"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,mockbase64"}}
                ]
            })

    # Trim to 6 assistant turns
    trimmed = TaskAgent._trim_scratchpad(messages, max_assistant_turns=6)

    # 1. Must never start with an orphan tool message
    assert trimmed[0].get("role") != "tool", "Trimmed scratchpad starts with an orphan tool message!"

    # 2. Count assistant messages retained
    assistant_count = len([m for m in trimmed if m.get("role") == "assistant"])
    assert assistant_count == 6, f"Expected 6 assistant turns, got {assistant_count}"

    # 3. Verify every tool message is preceded by its corresponding assistant message
    active_tool_calls = set()
    for i, msg in enumerate(trimmed):
        role = msg.get("role")
        if role == "assistant":
            # Record current tool call IDs
            active_tool_calls = {tc["id"] for tc in msg.get("tool_calls", [])}
        elif role == "tool":
            tcid = msg.get("tool_call_id")
            assert tcid in active_tool_calls, (
                f"Orphan tool response at index {i} with id '{tcid}' not in active tool calls: {active_tool_calls}"
            )
        elif role == "user":
            active_tool_calls = set()


def test_live_arc_env_bridge_ls20():
    """
    Instantiate ArcEnvBridge(game_id="ls20", operation_mode="OFFLINE"), call reset(),
    take 1 step("ACTION1"), and assert state frames are (64, 64) and env_steps_total == 1.
    """
    bridge = ArcEnvBridge(game_id="ls20", operation_mode="OFFLINE")
    initial_outcome = bridge.reset()

    assert len(initial_outcome.frames) >= 1
    assert initial_outcome.frames[0].shape == (64, 64)
    assert initial_outcome.env_steps_total == 0
    assert initial_outcome.state_status in ["NOT_FINISHED", "WIN", "GAME_OVER"]
    assert "ACTION1" in initial_outcome.available_actions

    outcome = bridge.step("ACTION1")
    assert len(outcome.frames) >= 1
    assert outcome.frames[0].shape == (64, 64)
    assert outcome.env_steps_total == 1
    assert outcome.state_status in ["NOT_FINISHED", "WIN", "GAME_OVER"]

    scorecard = bridge.get_official_scorecard()
    assert isinstance(scorecard, dict)
    assert "card_id" in scorecard or "total_actions" in scorecard


def test_mock_env_bridge_and_dry_run(tmp_path: Path):
    """
    Test MockArcEnvBridge mechanics and run TaskAgent dry-run end-to-end.
    """
    mock_env = MockArcEnvBridge(game_id="mock_maze")
    obs = mock_env.reset()
    assert obs.frames[0].shape == (64, 64)
    assert obs.env_steps_total == 0

    # Step ACTION4 (right) 4 times to reach goal at (10, 14)
    for _ in range(4):
        outcome = mock_env.step("ACTION4")

    # Reaching goal should advance level
    assert outcome.levels_completed >= 1
    scorecard = mock_env.get_official_scorecard()
    assert scorecard["score"] > 0
    assert scorecard["total_levels_completed"] >= 1

    # Test TaskAgent dry-run on mock_env
    repl = RLMReplSession(artifact_dir=tmp_path)
    wm = TwoStageWorldModel()
    mock_env2 = MockArcEnvBridge(game_id="mock_maze")
    agent = TaskAgent(
        skill_markdown="# Mock Skill",
        repl=repl,
        world_model=wm,
        env_bridge=mock_env2,
        dry_run=True,
    )

    final_outcome = agent.run_episode(max_env_steps=3)
    assert final_outcome.env_steps_total >= 3
    assert len(wm.unverified_hypotheses) > 0 or len(wm.proven_mechanics) > 0
    assert (tmp_path / "strip.png").exists()
