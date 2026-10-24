# `arc-agi-3-rsi`

> **Two-Loop Recursive Self-Improvement (RSI) Agent Harness for ARC-AGI-3**  
> *Built on Native OpenAI SDK, Out-of-Context RLM Python REPL, Multi-Step Exploitation Macros, and Evolutionary Search.*

---

## Overview

**ARC-AGI-3** introduces complex interactive environments with $64 \times 64$ grid dimensions, multi-pixel sprite entities, real-time animation sub-frames, and human action baselines. Naive LLM agent architectures fail on ARC-AGI-3 due to:
1. **Context Window Flooding**: Dumping raw $64 \times 64$ grids into prompts inflates token costs and blinds the LLM to spatial relationships.
2. **Action Inefficiency**: Every environment step (`env_step`) counts against the agent's efficiency score. Blind guessing leads to immediate score degradation.
3. **Catastrophic Regressions**: Learning a mechanic on Level 2 often breaks established solutions for Level 1 without strict non-regression gates.

`arc-agi-3-rsi` solves these challenges by implementing a **Two-Loop Recursive Self-Improvement (RSI)** architecture:

- **L0 Two-Stage World Model**: Dynamically pins `unverified_hypotheses`, `proven_mechanics`, and `fatal_transitions` directly into the system prompt on every turn.
- **L1 Inner-Loop Task Agent**: Operates with an out-of-context Python REPL holding a persistent `(16, 64, 64)` NumPy tensor, pure-NumPy spatial primitives (`find_clusters`, `diff_summary`, `bfs_shortest_path`), and exploitation macro-stepping (`env_step_macro`).
- **L2 Outer-Loop Learner Agent**: Deterministically compresses multi-game execution traces into high-signal diagnostic digests, proposes generalizable mutations to `SKILL.md`, and enforces a strict two-stage regression gate against ground-truth `official_scorecard` metrics.

---

## System Architecture

> For a complete mathematical and technical specification covering the state machine, RLM execution model, and regression gate criteria, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
                                  [ L2 OUTER LOOP ]
                               Evolutionary Search
                                        |
       +--------------------------------+--------------------------------+
       |                                                                 |
       v                                                                 v
+-----------------------+                                     +-----------------------+
|  TraceDigester        |                                     |  EvolutionTree        |
|  Deterministic        |                                     |  Strict Regression    |
|  <3,000 token digest  |                                     |  Gate & Checkpoints   |
+-----------------------+                                     +-----------------------+
       |                                                                 ^
       | Diagnostic Telemetry                                            | Evaluated Scores
       v                                                                 |
+------------------------------------------------------------------------+------------+
|                                  [ L1 INNER LOOP ]                                  |
|                               Task Agent & RLM REPL                                 |
|                                                                                     |
|   +-----------------------+    update_world_model (FREE)    +-------------------+   |
|   | L0 Two-Stage          | <=============================> | LLM Policy        |   |
|   | World Model           |                                 | (OpenAI / Gemini) |   |
|   +-----------------------+                                 +-------------------+   |
|                                                                       |             |
|                                 +-------------------------------------+             |
|                                 | repl_exec (FREE)       | env_step (BILLED)        |
|                                 v                        v                          |
|                     +-----------------------+  +-------------------+                |
|                     | Out-of-Context REPL   |  | ARC-AGI-3 Engine  |                |
|                     | (16, 64, 64) Tensor   |  | ArcEnvBridge      |                |
|                     | + Spatial Primitives  |  |                   |                |
|                     +-----------------------+  +-------------------+                |
+-------------------------------------------------------------------------------------+
```

---

## Key Features

### 1. Pure-NumPy RLM Spatial Primitives
The persistent Python REPL (`RLMReplSession`) provides three built-in spatial analysis functions directly in the execution namespace:
- `diff_summary()`: Compares `prev_state[-1]` and `state[-1]`, calculating exact centroid shifts `(cy, cx)` and displacement vectors $(\Delta y, \Delta x)$ for every color.
- `find_clusters(color=None)`: Fast connected-component labeling that extracts bounding boxes `[min_y, min_x, max_y, max_x]`, centroids, and pixel counts.
- `bfs_shortest_path(start, goal, walkable_colors=[0])`: Pure-NumPy BFS route planning that returns an ordered list of `GameAction` direction names (`ACTION1`..`ACTION4`).

### 2. Multi-Step Exploitation Macro (`env_step_macro`)
Allows the Task Agent to plan an optimal route in `repl_exec` and execute up to 15 actions in a single turn. Every sub-step is sequentially billed and traced, with mandatory early-stop guardrails:
- Halts immediately on `WIN` or `GAME_OVER`.
- Halts on level transitions (`levels_completed > prev_levels_completed`).
- Halts on zero-change wall bumps when `stop_on_noop=True`.

### 3. Strict Non-Regression Gate
Candidate skills in the evolutionary tree must satisfy two mathematical guarantees:
1. **Mastered Level Retention**: For every level where the parent achieved a score $\ge 1.0$, the candidate must also achieve $\ge 1.0$.
2. **Empirical Improvement**:
   $$\text{MeanScore}_{\text{cand}} > \text{MeanScore}_{\text{parent}} \quad\text{OR}\quad (\text{Equal Score} \land \text{Higher Tiebreaker})$$

### 4. Parallel Batch Evolution & Checkpoint Resumption
- Batch game evaluations run concurrently via `ThreadPoolExecutor(max_workers=workers)` with full thread isolation.
- Interrupted evolutionary sweeps can be resumed instantly with `--resume <exp_dir>`, skipping already evaluated nodes and continuing from the best frontier skill.

### 5. Visual Trajectory HUD & Animated GIF Generator (`inspect_run.py`)
Renders any episode into an annotated `replay.gif` showing the $256 \times 256$ game grid alongside a live HUD telemetry panel (Step, Level, Action, Status, and active World Model rules). Also exports GitHub-ready `BENCHMARK.md` comparison tables.

---

## Repository Structure

```
ARC-AGI-3-RSI/
├── loop.py                     # Single-game evaluation entry point
├── evolve.py                   # Multi-generation parallel evolutionary loop
├── inspect_run.py              # Animated GIF replay and benchmark generator
├── pyproject.toml              # Build metadata and dependencies
├── skills/
│   ├── SKILL.md                # L1 Domain Skill (Gen 0 Baseline & Active Frontier)
│   └── L2_learner_policy.md    # L2 Meta-Policy for the Learner Agent
├── src/
│   ├── env_bridge.py           # ArcEnvBridge & MockArcEnvBridge
│   ├── evolution_tree.py       # EvolutionTree, SkillNode & Regression Gate
│   ├── learner_agent.py        # Outer-Loop Learner Agent
│   ├── repl_session.py         # RLMReplSession & Spatial Primitives
│   ├── task_agent.py           # Inner-Loop TaskAgent & env_step_macro
│   ├── task_memory.py          # TwoStageWorldModel
│   └── trace_digest.py         # Diagnostic TraceDigester
├── tests/
│   ├── test_inner_loop.py      # Milestone 1 tests
│   ├── test_outer_loop.py      # Milestone 2 tests
│   ├── test_milestone3.py      # Milestone 3 tests (macro, parallel, resume, retry)
│   └── test_milestone4.py      # Milestone 4 tests (spatial primitives, live API)
└── docs/
    └── ARCHITECTURE.md         # In-depth technical specification
```

---

## Installation

### Prerequisites
- Python 3.10+
- Virtual environment recommended

```bash
# Clone the repository
git clone https://github.com/your-org/arc-agi-3-rsi.git
cd arc-agi-3-rsi

# Install dependencies in editable mode
pip install -e .
```

---

## Quickstart

### 1. Zero-Cost Offline Smoke Test (Mock Mode)
Run a single episode with the built-in deterministic `MockArcEnvBridge` (no API keys required):
```bash
python loop.py --mock --max-steps 15
```

### 2. Live Game Evaluation on ARC-AGI-3
Evaluate `SKILL.md` on official ARC game `ls20`:
```bash
# Set your OpenAI or Gemini API credentials
export OPENAI_API_KEY="your-api-key"
# Optional: for Gemini via Google's OpenAI-compatible endpoint
# export OPENAI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"

python loop.py --game ls20 --model gemini-2.5-flash --max-steps 45
```

### 3. Multi-Generation Evolutionary Sweep
Launch a 2-generation evolutionary run across games with 2 parallel workers:
```bash
python evolve.py --games ls20 --generations 2 --branch-factor 2 --workers 2 --max-steps 30
```

### 4. Resuming an Interrupted Evolution
Resume an existing experiment from its checkpoint directory:
```bash
python evolve.py --games ls20 --generations 4 --branch-factor 2 --workers 2 --resume evolution_runs/exp_live_1792870531
```

### 5. Inspecting Replays & Benchmark Tables
Generate animated GIFs and benchmark comparisons:
```bash
# Generate replay.gif for a single episode
python inspect_run.py --run-dir runs/ls20_1792867707

# Benchmark an entire evolutionary experiment and render all run GIFs
python inspect_run.py --exp-dir evolution_runs/exp_live_1792870531
```

---

## Benchmark Results (Sample Sweep on `ls20`)

Sample output generated by `inspect_run.py` during an evolutionary sweep:

| Game ID | Gen 0 Score | Frontier Score | Score Delta | Gen 0 Actions | Frontier Actions | Action Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `ls20` | 0.00 | 0.00 | **+0.00** | 12 | 12 | 0 |

### Candidate Lineage
| Node ID | Gen | Parent | Batch Score | Tiebreaker | Actions | Gate Status | Mutation Rationale |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `gen0_root` | 0 | `-` | 0.0000 | 30.00 | 12 | [ROOT] | Initial Gen 0 Baseline Domain Skill |
| `gen1_v1` | 1 | `gen0_root` | 0.0000 | 40.00 | 12 | [PASS] ACCEPTED | Spatial 3x3 Neighborhood Probe (Tiebreaker +10.0) |
| `gen1_v2` | 1 | `gen0_root` | 0.0000 | 40.00 | 12 | [PASS] ACCEPTED | BFS Route Pathfinder + `env_step_macro` |
| `gen2_v1` | 2 | `gen1_v1` | 0.0000 | 40.00 | 12 | [FAIL] REJECTED | No tiebreaker improvement over Gen 1 |

---

## Testing

Run the complete 16-test automated verification suite:

```bash
pytest -v
```

```text
tests/test_inner_loop.py::test_repl_tensor_normalization PASSED          [  6%]
tests/test_inner_loop.py::test_render_frame_strip PASSED                 [ 12%]
tests/test_inner_loop.py::test_world_model_lifecycle PASSED              [ 18%]
tests/test_inner_loop.py::test_scratchpad_atomic_trimming PASSED         [ 25%]
tests/test_inner_loop.py::test_live_arc_env_bridge_ls20 PASSED           [ 31%]
tests/test_inner_loop.py::test_mock_env_bridge_and_dry_run PASSED        [ 37%]
tests/test_milestone3.py::test_env_step_macro_execution_and_early_stop PASSED [ 43%]
tests/test_milestone3.py::test_trajectory_frames_npz_and_gif_generation PASSED [ 50%]
tests/test_milestone3.py::test_parallel_evolution_and_checkpoint_resume PASSED [ 56%]
tests/test_milestone3.py::test_llm_retry_backoff_on_transient_error PASSED [ 62%]
tests/test_milestone4.py::test_spatial_primitives_synthetic_grid PASSED  [ 68%]
tests/test_milestone4.py::test_live_api_connectivity_and_initialization PASSED [ 75%]
tests/test_outer_loop.py::test_trace_digester_noop_and_cycle_detection PASSED [ 81%]
tests/test_outer_loop.py::test_regression_gate_blocks_mastered_level_drop PASSED [ 87%]
tests/test_outer_loop.py::test_learner_agent_guardrails_and_dry_run PASSED [ 93%]
tests/test_outer_loop.py::test_end_to_end_evolution_pipeline_mock PASSED [100%]

============================= 16 passed in 3.34s ==============================
```

---

## License

Apache-2.0
