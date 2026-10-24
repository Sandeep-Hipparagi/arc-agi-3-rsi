# ARC-AGI-3 Domain Skill (Gen 0 Baseline)

## 1. Core Operating Cycle
You are solving an unknown interactive grid game where every `env_step` counts against your efficiency score against a human baseline. Internal `repl_exec` and `update_world_model` calls are FREE (0 env steps). Never guess blindly. Follow this strict 5-stage cycle:

1. **OBSERVE (Multi-Frame RLM Inspection)**:
   - Never dump full 64x64 arrays into prompt context.
   - Use `repl_exec` to inspect `state` (a fixed `(16, 64, 64)` `int32` NumPy array where `state[-1]` is the latest settled frame and `state[0..15]` holds the animation sequence).
   - Call `find_clusters()` to automatically extract bounding boxes, centroids, and pixel counts for all sprites on `state[-1]`.
   - Call `diff_summary()` after any action to compute exact pixel deltas, centroid movements, and displacement vectors $(\Delta y, \Delta x)$ by color.
   - Compute motion vectors across sub-frames (`np.argwhere(state[i] != state[i-1])`) to detect automated environment dynamics (e.g., patrolling enemies, gravity, waterfalls, timers).
   - If visual inspection is needed, call `render_frame_strip('strip.png')` inside `repl_exec` to view the 16-frame composite strip.

2. **GROW (Two-Stage World Model Hygiene)**:
   - **Unverified Hypotheses**: Whenever you see a new color, object, or state transition, log a falsifiable hypothesis via `update_world_model(op="add_hypothesis", ...)`.
   - **Proven Mechanics**: Only promote a hypothesis to Proven Mechanics via `update_world_model(op="promote_to_proven", ...)` after observing deterministic confirmation across transitions (e.g., confirming controllable avatar motion vectors from `diff_summary()`).
   - **Fatal States ("Never Make the Same Mistake Twice")**: If an action causes a level reset or `GAME_OVER`, immediately record the exact spatial precondition under Banned / Fatal Transitions via `update_world_model(op="add_fatal", ...)`, then call `RESET`.

3. **PLAN (Minimal-Action Probing vs. Exploitation)**:
   - *Exploration Phase*: When mechanics are unknown, design a 1-step probe that tests the highest-uncertainty hypothesis with zero risk of hitting known fatal states.
   - *Exploitation Phase*: Once player movement, obstacles, enemies, and goal conditions are proven, call `bfs_shortest_path(start, goal, walkable_colors=[0])` in `repl_exec` to compute the shortest collision-free sequence of moves (`ACTION1`..`ACTION4`).

4. **EXECUTE**:
   - *Exploration Phase*: Execute a single `env_step` probe while mechanics, obstacles, or enemy hazards are uncertain.
   - *Exploitation Phase*: Once player movement, collision rules, and pathways are verified in `repl_exec` (via `bfs_shortest_path` or collision probing), execute multi-step paths in a single turn using `env_step_macro(actions=[...], stop_on_noop=True)`. The macro automatically halts on level advance, game over, or unexpected obstacles. Each executed step is billed.

5. **EFFICIENCY CHECK**:
   - After every 3 `env_step` calls, check if net progress toward the goal or hypothesis resolution was made. If repeating cyclic states, halt movement and re-analyze `history` in the REPL.

## 2. Reusable REPL Inspection Snippets
Use or adapt these snippets inside `repl_exec`:
```python
# 1. Built-in connected component sprite extraction
clusters = find_clusters()  # Returns list of dicts: [{'color': c, 'box': [min_y, min_x, max_y, max_x], 'centroid': [cy, cx], 'pixels': N}]

# 2. Built-in transition motion & displacement analysis
delta = diff_summary()  # Compares prev_state[-1] and state[-1], prints displacement vectors (dy, dx) by color

# 3. Built-in BFS pathfinding for exploitation macros
# Example: player at (10, 10), goal at (10, 18), walkable background is color 0
route = bfs_shortest_path(start_yx=(10, 10), goal_yx=(10, 18), walkable_colors=[0])
# Once path is verified clear, call: env_step_macro(actions=route, stop_on_noop=True, thought="Follow BFS trajectory")

# 4. Palette & entity pixel counts in latest settled frame
vals, counts = np.unique(state[-1], return_counts=True)
print(dict(zip(vals.tolist(), counts.tolist())))

# 5. Sub-frame motion analysis across 16-frame animation tensor
for idx in range(1, 16):
    step_diff = np.argwhere(state[idx] != state[idx-1])
    if len(step_diff) > 0:
        print(f"Sub-frame {idx} motion: {len(step_diff)} pixels changed")

# 6. On-demand visual inspection
render_frame_strip("strip.png", scale=2)
```
