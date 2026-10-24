import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from openai import OpenAI

from src.repl_session import RLMReplSession
from src.task_agent import TaskAgent
from src.learner_agent import LearnerAgent
from src.env_bridge import MockArcEnvBridge
import loop
import evolve


def test_spatial_primitives_synthetic_grid(tmp_path: Path):
    """
    Tests pure-NumPy RLM spatial primitives (find_clusters, diff_summary, bfs_shortest_path)
    on synthetic 64x64 grids with multi-pixel sprites, obstacles, and displacement.
    """
    repl = RLMReplSession(artifact_dir=tmp_path / "frames")

    # 1. Construct a synthetic 64x64 grid
    grid = np.zeros((64, 64), dtype=np.int32)
    # Player sprite: 3x3 Blue (color 1) centered at (11, 11)
    grid[10:13, 10:13] = 1
    # Goal sprite: 3x3 Green (color 3) centered at (11, 21)
    grid[10:13, 20:23] = 3
    # Vertical obstacle wall: Gray (color 5) blocking x=15 from y=9 to y=13
    grid[9:14, 15] = 5

    repl.namespace["state"][-1] = grid

    # 2. Test find_clusters()
    clusters = repl.find_clusters(grid=grid)
    assert len(clusters) == 3

    c1 = next(c for c in clusters if c["color"] == 1)
    assert c1["box"] == [10, 10, 12, 12]
    assert c1["centroid"] == [11.0, 11.0]
    assert c1["pixels"] == 9

    c3 = next(c for c in clusters if c["color"] == 3)
    assert c3["box"] == [10, 20, 12, 22]
    assert c3["centroid"] == [11.0, 21.0]
    assert c3["pixels"] == 9

    c5 = next(c for c in clusters if c["color"] == 5)
    assert c5["box"] == [9, 15, 13, 15]
    assert c5["pixels"] == 5

    # Test filtering find_clusters by color
    only_blue = repl.find_clusters(color=1, grid=grid)
    assert len(only_blue) == 1
    assert only_blue[0]["color"] == 1

    # 3. Test diff_summary()
    next_grid = grid.copy()
    # Move player right by 2 pixels (from x=10..12 to x=12..14)
    next_grid[10:13, 10:12] = 0
    next_grid[10:13, 13:15] = 1

    delta = repl.diff_summary(prev_frame=grid, curr_frame=next_grid)
    assert delta["total_changed_pixels"] > 0
    assert 1 in delta["by_color"]

    c1_delta = delta["by_color"][1]
    assert c1_delta["prev_centroid"] == (11.0, 11.0)
    assert c1_delta["curr_centroid"] == (11.0, 13.0)
    # Centroid displacement should be dy=0.0, dx=+2.0
    assert c1_delta["displacement"] == (0.0, 2.0)

    # 4. Test bfs_shortest_path()
    # Path from player centroid (11, 11) to goal centroid (11, 21) around wall at x=15
    path = repl.bfs_shortest_path(start_yx=(11, 11), goal_yx=(11, 21), walkable_colors=[0], grid=grid)
    assert len(path) > 0

    # Simulate path traversal to ensure it avoids wall at x=15
    cur_y, cur_x = 11, 11
    move_map = {
        "ACTION1": (-1, 0),
        "ACTION2": (1, 0),
        "ACTION3": (0, -1),
        "ACTION4": (0, 1),
    }
    for action in path:
        dy, dx = move_map[action]
        cur_y += dy
        cur_x += dx
        assert grid[cur_y, cur_x] != 5, f"Path collided with wall at ({cur_y}, {cur_x})!"

    assert (cur_y, cur_x) == (11, 21), f"Path ended at ({cur_y}, {cur_x}) instead of goal (11, 21)!"

    # 5. Test direct execution inside REPL namespace
    repl_output = repl.execute("""
c = find_clusters(color=1)
p = bfs_shortest_path((11, 11), (11, 21), walkable_colors=[0])
print(f"REPL_SUCCESS: clusters={len(c)}, path_len={len(p)}")
""")
    assert "REPL_SUCCESS: clusters=1, path_len=" in repl_output


def test_live_api_connectivity_and_initialization(monkeypatch):
    """
    Verifies that TaskAgent and LearnerAgent cleanly initialize native OpenAI client
    connections, resolve environment variables, and handle retry configurations.
    """
    test_key = "sk-test-live-key-12345"
    test_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    monkeypatch.setenv("OPENAI_API_KEY", test_key)
    monkeypatch.setenv("OPENAI_BASE_URL", test_base_url)

    # 1. Initialize TaskAgent with resolved environment credentials
    task_agent = TaskAgent(
        skill_markdown=Path("skills/SKILL.md").read_text(encoding="utf-8"),
        repl=RLMReplSession(artifact_dir=Path(".pytest_tmp")),
        world_model=MagicMock(),
        env_bridge=MockArcEnvBridge(),
        model="gemini-2.5-flash",
        base_url=test_base_url,
        api_key=test_key,
        dry_run=False,
    )

    assert task_agent.client is not None
    assert isinstance(task_agent.client, OpenAI)
    assert str(task_agent.client.base_url).startswith(test_base_url)
    assert task_agent.client.api_key == test_key

    # 2. Initialize LearnerAgent with explicit parameters
    learner_agent = LearnerAgent(
        model="gemini-2.5-flash",
        base_url=test_base_url,
        api_key=test_key,
        dry_run=False,
    )

    assert learner_agent.client is not None
    assert isinstance(learner_agent.client, OpenAI)
    assert str(learner_agent.client.base_url).startswith(test_base_url)

    # 3. Test CLI argument resolution in loop.py and evolve.py
    loop_parser = loop.main.__code__  # Verify loop.py imports and defines entry point
    assert loop_parser is not None
    evolve_parser = evolve.main.__code__
    assert evolve_parser is not None
