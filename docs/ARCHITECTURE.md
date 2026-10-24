# ARC-AGI-3 Recursive Self-Improvement (RSI) Architecture Specification

## 1. Executive Summary & Design Invariants

The `arc-agi-3-rsi` framework implements a **Two-Loop Recursive Self-Improvement (RSI)** architecture specifically tailored for the interactive reasoning demands of **ARC-AGI-3**. 

ARC-AGI-3 differs fundamentally from prior grid-world benchmarks:
- **Spatial Resolution & Dynamics**: Games are represented on a $64 \times 64$ grid with multi-pixel entity sprites that frequently move at non-unit velocities across up to 16 sub-frames of animation.
- **Strict Action Billing**: Every environment transition (`env_step`) counts against the agent's efficiency score relative to human baselines. Blind exploration leads to catastrophic action penalties.
- **Multi-Level Mastery**: Games contain progressive levels where mechanics, obstacles, and enemy behaviors evolve.

```
+-----------------------------------------------------------------------------------+
|                           L2 OUTER-LOOP LEARNER AGENT                             |
|                                                                                   |
|   +-----------------------+     +-------------------+     +-------------------+   |
|   | Deterministic Trace   | --> | Mutation Engine   | --> | Strict Regression |   |
|   | Digester (<3k tokens) |     | (SKILL.md / Meta) |     | Gate & Lineage    |   |
|   +-----------------------+     +-------------------+     +-------------------+   |
+------------------------------------------|----------------------------------------+
                                           | Mutated SKILL.md
                                           v
+-----------------------------------------------------------------------------------+
|                            L1 INNER-LOOP TASK AGENT                               |
|                                                                                   |
|  [Pinned System Prompt]                                                           |
|  +-----------------------+       +-------------------+                            |
|  | Mutated SKILL.md      |       | L0 Two-Stage      |                            |
|  | Operational Heuristics| <---> | World Model       |                            |
|  +-----------------------+       +-------------------+                            |
|             ^                              ^                                      |
|             |                              | update_world_model (FREE)            |
|             v                              v                                      |
|  +---------------------------------------------------+                            |
|  |          LLM Policy (gemini-2.5-flash / OpenAI)   |                            |
|  +---------------------------------------------------+                            |
|             | repl_exec (FREE)             | env_step / env_step_macro (BILLED)   |
|             v                              v                                      |
|  +-----------------------+       +-------------------+                            |
|  | Out-of-Context REPL   |       | ARC-AGI-3 Engine  |                            |
|  | (16, 64, 64) Tensor   | <===> | ArcEnvBridge      |                            |
|  | + Spatial Primitives  |       |                   |                            |
|  +-----------------------+       +-------------------+                            |
+-----------------------------------------------------------------------------------+
```

---

## 2. In-Depth Component Specifications

### 2.1 L0: Two-Stage World Model (`TwoStageWorldModel`)
Located in [`src/task_memory.py`](../src/task_memory.py), this component prevents catastrophic forgetting and enforces epistemic humility across turns.
- **`unverified_hypotheses`**: Key-value hypotheses formed during preliminary observations.
- **`proven_mechanics`**: Verified empirical rules promoted only when transitions reproduce deterministically.
- **`fatal_transitions`**: Hard spatial constraints ("never make the same mistake twice"). If an action triggers a reset or `GAME_OVER`, the exact spatial precondition is recorded here.
- **Dynamic System Injection**: On every LLM turn, the World Model renders markdown directly into the pinned system prompt:
  ```markdown
  ### Proven Mechanics (Empirically Verified)
  - `player_avatar`: Color 1 sprite moves via ACTION1..ACTION4 with step_size=1
  ### Banned / Fatal Transitions (Do Not Repeat)
  - `hazard_collision`: Color 2 red entity moving at (20, 20) triggers instant GAME_OVER
  ```

---

### 2.2 L1: Out-of-Context RLM REPL & Spatial Primitives
Located in [`src/repl_session.py`](../src/repl_session.py), `RLMReplSession` maintains the environment tensor entirely outside the LLM context window.

#### Tensor Normalization
- Variable frame sequences from the ARC engine are normalized into fixed `(16, 64, 64)` `int32` tensors.
- Leading slots are padded with the initial frame; `state[-1]` is **always** the latest settled frame, while `state[0..15]` captures sub-frame motion.

#### Zero-Dependency Pure-NumPy Spatial Primitives
The REPL injects three high-speed spatial primitives into the execution namespace:

1. **`diff_summary(prev_frame=None, curr_frame=None) -> Dict[str, Any]`**:
   - Detects all changed coordinates between settled steps.
   - Computes whole-sprite centroid positions `(cy, cx)` and displacement vectors $(\Delta y, \Delta x)$ per color.
   - Outputs structured telemetry and formatted ASCII summaries.

2. **`find_clusters(color=None, grid=None) -> List[Dict[str, Any]]`**:
   - Pure-NumPy connected-component flood fill (4-connectivity).
   - Extracts bounding boxes `[min_y, min_x, max_y, max_x]`, centroids `[cy, cx]`, and exact pixel counts for all non-background sprites.

3. **`bfs_shortest_path(start_yx, goal_yx, walkable_colors=[0], step_size=1, grid=None) -> List[str]`**:
   - Pure-NumPy BFS shortest-path search.
   - Automatically treats `grid[start]` and `grid[goal]` as traversable so sprites can navigate out of their initial footprint and reach target tiles.
   - Translates coordinates into directional `GameAction` strings (`ACTION1`=UP, `ACTION2`=DOWN, `ACTION3`=LEFT, `ACTION4`=RIGHT).

---

### 2.3 Action Efficiency & Multi-Step Exploitation Macro (`env_step_macro`)
Located in [`src/task_agent.py`](../src/task_agent.py), the Task Agent observes strict economic discipline:
- `repl_exec` and `update_world_model` are **FREE** (0 environment steps).
- `env_step` and `env_step_macro` call `env_bridge.step()` and increment billed action counters.

#### Macro Early-Stop Guardrails
When executing multi-step trajectories during the exploitation phase:
```python
env_step_macro(actions=["ACTION4", "ACTION4", "ACTION4"], stop_on_noop=True, thought="Reach goal")
```
The macro sequentially steps the environment, bills each action, emits per-step traces with `"via_macro": True`, and **aborts immediately** if:
1. `state_status in ("WIN", "GAME_OVER")`
2. `levels_completed > prev_levels_completed` (level transition)
3. `stop_on_noop and not grid_changed` (unexpected wall bump)

---

### 2.4 L2: Diagnostic Trace Digestion & Evolutionary Search

#### Deterministic Trace Digester (`TraceDigester`)
Raw multi-game JSONL execution traces can exceed 100k+ tokens. `TraceDigester` ([`src/trace_digest.py`](../src/trace_digest.py)) compresses runs into high-signal `<3,000`-token Markdown digests containing:
- Official scorecard telemetry (`level_scores`, `level_actions`, `level_baseline_actions`).
- Action efficiency hygiene: No-op wall bump ratio and cyclic state revisit detection via MD5 state hashing.
- World Model growth metrics: Number of proven mechanics vs unverified hypotheses.

#### Two-Stage Regression Gate (`EvolutionTree`)
Located in [`src/evolution_tree.py`](../src/evolution_tree.py), candidate skills must satisfy two strict conditions to be accepted:
1. **Non-Regression on Mastered Levels**:
   $$\forall (g, l) \text{ s.t. } Score_{parent}(g, l) \ge 1.0 \implies Score_{candidate}(g, l) \ge 1.0$$
2. **Strict Improvement**:
   $$MeanScore_{cand} > MeanScore_{parent}$$
   $$\text{OR } (MeanScore_{cand} == MeanScore_{parent} \land Tiebreaker_{cand} > Tiebreaker_{parent})$$
   $$\text{OR } (MeanScore_{cand} == MeanScore_{parent} \land Tiebreaker_{cand} == Tiebreaker_{parent} \land Actions_{cand} < Actions_{parent})$$

#### Tiebreaker Formulation
$$Tiebreaker = 10 \cdot |\text{proven\_mechanics}| - 5 \cdot \text{resets} - 1 \cdot \text{noop\_steps}$$

---

## 3. Parallel Batch Execution & Checkpoint Resumption

In [`evolve.py`](../evolve.py), batch game evaluations run concurrently via `ThreadPoolExecutor(max_workers=workers)`.
- **Thread Safety**: Every worker thread instantiates an independent `ArcEnvBridge`, `RLMReplSession`, `TwoStageWorldModel`, and `TaskAgent`. Mutable state is never shared across threads.
- **Checkpoint Resumption (`--resume <exp_dir>`)**:
  - Automatically loads existing search nodes from `<exp_dir>/lineage_tree.json`.
  - Reconstructs frontier parent digests directly from stored run directories.
  - Skips already completed generations and seamlessly resumes evolution from the highest evaluated generation.

---

## 4. Visual Replay & HUD Telemetry Generation

In [`inspect_run.py`](../inspect_run.py), episodes are rendered into animated GIFs with synchronous HUD telemetry sidebars:
- Grid scaled with nearest-neighbor interpolation to $256 \times 256$ using the official 16-color palette.
- Sidebar HUD displays Game ID, Frame counter, Step number, Level, Action/Macro index, Grid Changed status, and active L0 World Model counts.
- Outputs `replay.gif` and generates GitHub-ready comparative `BENCHMARK.md` reports.
