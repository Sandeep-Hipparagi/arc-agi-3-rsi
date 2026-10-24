import json
import os
import types
import hashlib
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import numpy as np
import openai
from openai import OpenAI

from src.repl_session import RLMReplSession
from src.task_memory import TwoStageWorldModel
from src.env_bridge import ArcEnvBridge, StepOutcome


TASK_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "repl_exec",
            "description": (
                "Execute Python code in the persistent RLM REPL. "
                "`state` is a (16, 64, 64) numpy int32 array where `state[-1]` is the latest frame "
                "and `state[0..15]` holds the 16-frame sequence. `prev_state` holds the previous "
                "(16, 64, 64) tensor. Call `render_frame_strip('strip.png')` inside code to view "
                "the 16-frame visual strip. FREE operation (0 env steps)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code using numpy (`np`), `state`, `prev_state`, `history`, `render_frame_strip`."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_world_model",
            "description": (
                "Update the L0 Two-Stage World Model. Separate unverified hypotheses from "
                "proven mechanics and fatal state transitions. FREE operation (0 env steps)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "add_hypothesis",
                            "promote_to_proven",
                            "refute_hypothesis",
                            "add_fatal"
                        ]
                    },
                    "key": {
                        "type": "string",
                        "description": "Short identifier (e.g., 'color_2_red_entity', 'action_1_up')."
                    },
                    "description": {
                        "type": "string",
                        "description": "Precise mechanic rule, spatial coordinate condition, or hypothesis."
                    }
                },
                "required": ["op", "key", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "env_step",
            "description": (
                "Execute ONE action in the ARC-AGI-3 environment. BILLED operation "
                "(counts toward human baseline efficiency score). Only call after verifying "
                "state in `repl_exec` and recording hypotheses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "One of the available GameAction names (e.g., ACTION1..ACTION7, RESET)."
                    },
                    "x": {
                        "type": "integer",
                        "description": "Optional X coordinate (0..63) if action is ACTION6."
                    },
                    "y": {
                        "type": "integer",
                        "description": "Optional Y coordinate (0..63) if action is ACTION6."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Concise hypothesis being tested or goal step being executed."
                    }
                },
                "required": ["action", "thought"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "env_step_macro",
            "description": (
                "Execute a verified sequence of up to 15 actions in the ARC-AGI-3 environment "
                "during the EXPLOITATION phase (after verifying path/mechanics in repl_exec). "
                "Automatically halts early on level completion, GAME_OVER, WIN, or unexpected "
                "wall-bump (no-op). Each executed step is billed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of GameAction names (e.g. ['ACTION1', 'ACTION1', 'ACTION3']), max length 15."
                    },
                    "stop_on_noop": {
                        "type": "boolean",
                        "description": "Halt macro immediately if any step fails to change the 64x64 grid (default True)."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Summary of the REPL-verified trajectory being executed."
                    }
                },
                "required": ["actions", "thought"]
            }
        }
    }
]


class _MockFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _MockToolCall:
    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.type = "function"
        self.function = _MockFunction(name, json.dumps(arguments))


class _MockMessage:
    def __init__(self, content: str, tool_calls: List[_MockToolCall]):
        self.content = content
        self.tool_calls = tool_calls


class _MockChoice:
    def __init__(self, message: _MockMessage):
        self.message = message


class _MockResponse:
    def __init__(self, choice: _MockChoice):
        self.choices = [choice]


class TaskAgent:
    """
    Inner-Loop Task Agent powered by native `openai.OpenAI` SDK.
    Compatible with Gemini 2.5 Flash, GPT-4o-mini, Qwen, vLLM, and Ollama endpoints.
    """
    def __init__(
        self,
        skill_markdown: str,
        repl: RLMReplSession,
        world_model: TwoStageWorldModel,
        env_bridge: Any,
        model: str = "gemini-2.5-flash",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_vision: bool = True,
        rolling_window_turns: int = 12,
        trace_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        dry_run: bool = False,
        run_dir: Optional[Path] = None,
    ):
        self.dry_run = dry_run
        self.model = model
        self.skill_markdown = skill_markdown
        self.repl = repl
        self.world_model = world_model
        self.env_bridge = env_bridge
        self.enable_vision = enable_vision
        self.rolling_window_turns = rolling_window_turns
        self.trace_callback = trace_callback
        self.run_dir = run_dir
        self.scratchpad_messages: List[Dict[str, Any]] = []
        self._settled_frames_history: List[np.ndarray] = []

        if not self.dry_run:
            resolved_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
            self.client: Optional[OpenAI] = OpenAI(base_url=base_url, api_key=resolved_key)
        else:
            self.client = None

    def _call_llm_with_retry(self, call_fn, max_retries: int = 4, **kwargs) -> Any:
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                return call_fn(**kwargs)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ) as e:
                if attempt == max_retries:
                    raise
                jitter = random.uniform(0.05, 0.2)
                time.sleep(backoff + jitter)
                backoff *= 2.0
            except openai.APIError as e:
                status = getattr(e, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    jitter = random.uniform(0.05, 0.2)
                    time.sleep(backoff + jitter)
                    backoff *= 2.0
                else:
                    raise

    def _build_messages(self, last_outcome: StepOutcome) -> List[Dict[str, Any]]:
        system_content = (
            f"{self.skill_markdown}\n\n"
            f"---\n"
            f"{self.world_model.render_markdown()}\n\n"
            f"---\n"
            f"## Live Environment Telemetry\n"
            f"- Game ID: `{self.env_bridge.game_id}`\n"
            f"- Current Status: `{last_outcome.state_status}`\n"
            f"- Levels Completed: `{last_outcome.levels_completed}`\n"
            f"- Env Steps (Current Level): `{last_outcome.env_steps_current_level}`\n"
            f"- Env Steps (Total): `{last_outcome.env_steps_total}`\n"
            f"- Available Actions: `{last_outcome.available_actions}`\n"
            f"- REPL `state` shape: `(16, 64, 64)` (`num_valid_frames={len(last_outcome.frames)}`)"
        )

        trimmed = self._trim_scratchpad(self.scratchpad_messages, self.rolling_window_turns)
        return [{"role": "system", "content": system_content}] + trimmed

    @staticmethod
    def _trim_scratchpad(messages: List[Dict[str, Any]], max_assistant_turns: int) -> List[Dict[str, Any]]:
        """
        Trims scratchpad while guaranteeing atomic turn integrity:
        - NEVER leaves an orphan `role == 'tool'` message at the start.
        - NEVER separates an `assistant` `tool_calls` message from its `tool` responses.
        - Preserves leading user context if needed.
        """
        if not messages:
            return []

        blocks: List[List[Dict[str, Any]]] = []
        current_block: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                current_block.append(msg)
            elif role == "tool":
                current_block.append(msg)
            elif role == "user":
                # Check if this user message is a vision attachment immediately following tool responses
                is_vision = isinstance(msg.get("content"), list) and any(
                    isinstance(c, dict) and c.get("type") == "image_url" for c in msg.get("content")
                )
                if is_vision and current_block and any(m.get("role") == "tool" for m in current_block):
                    current_block.append(msg)
                else:
                    if current_block:
                        blocks.append(current_block)
                        current_block = []
                    current_block.append(msg)
            else:
                current_block.append(msg)

        if current_block:
            blocks.append(current_block)

        # Count blocks that contain assistant messages
        assistant_blocks = [b for b in blocks if any(m.get("role") == "assistant" for m in b)]
        if len(assistant_blocks) <= max_assistant_turns:
            return messages

        retained_assistant_blocks = assistant_blocks[-max_assistant_turns:]
        first_retained_block = retained_assistant_blocks[0]
        first_idx = blocks.index(first_retained_block)

        # Include an immediately preceding user block if present
        if first_idx > 0 and all(m.get("role") == "user" for m in blocks[first_idx - 1]):
            retained_blocks = blocks[first_idx - 1 :]
        else:
            retained_blocks = blocks[first_idx:]

        flat_result: List[Dict[str, Any]] = []
        for b in retained_blocks:
            flat_result.extend(b)

        # Final sanity check: ensure no orphan tool messages at start
        while flat_result and flat_result[0].get("role") == "tool":
            flat_result.pop(0)

        return flat_result

    def _get_dry_run_completion(self, internal_turn: int, outcome: StepOutcome) -> _MockResponse:
        """
        Deterministic mock model completion following the 5-stage inner-loop cycle.
        Used for fast, zero-cost smoke testing and verification.
        """
        step = outcome.env_steps_total
        cycle_idx = step % 4

        if cycle_idx == 0:
            tool_calls = [
                _MockToolCall(
                    f"call_repl_{internal_turn}",
                    "repl_exec",
                    {"code": "render_frame_strip('strip.png'); vals, counts = np.unique(state[-1], return_counts=True); print('Grid palette counts:', dict(zip(vals.tolist()[:5], counts.tolist()[:5])))"}
                ),
                _MockToolCall(
                    f"call_wm_{internal_turn}",
                    "update_world_model",
                    {"op": "add_hypothesis", "key": f"nav_entity_step_{step}", "description": f"Controllable entity detected at step {step}"}
                ),
                _MockToolCall(
                    f"call_env_{internal_turn}",
                    "env_step",
                    {"action": "ACTION1", "thought": "Observe player reaction to ACTION1"}
                )
            ]
        elif cycle_idx == 1:
            tool_calls = [
                _MockToolCall(
                    f"call_repl_{internal_turn}",
                    "repl_exec",
                    {"code": "diff = np.argwhere(prev_state[-1] != state[-1]); print(f'Pixels changed after ACTION1: {len(diff)}')"}
                ),
                _MockToolCall(
                    f"call_wm_{internal_turn}",
                    "update_world_model",
                    {"op": "promote_to_proven", "key": f"nav_entity_step_{step-1}", "description": "Entity motion confirmed across transitions"}
                ),
                _MockToolCall(
                    f"call_env_{internal_turn}",
                    "env_step",
                    {"action": "ACTION2", "thought": "Probe opposite movement direction (ACTION2)"}
                )
            ]
        elif cycle_idx == 2:
            has_spatial_probe = "Local 3x3 Obstacle" in self.skill_markdown
            has_bfs_planner = "Deterministic BFS" in self.skill_markdown
            if has_bfs_planner:
                tool_calls = [
                    _MockToolCall(
                        f"call_wm_adv_{internal_turn}",
                        "update_world_model",
                        {"op": "promote_to_proven", "key": "collision_safety", "description": "BFS path verified clear toward target"}
                    ),
                    _MockToolCall(
                        f"call_macro_{internal_turn}",
                        "env_step_macro",
                        {"actions": ["ACTION4", "ACTION4"], "stop_on_noop": True, "thought": "Execute 2-step BFS macro along cleared path"}
                    )
                ]
            elif has_spatial_probe:
                tool_calls = [
                    _MockToolCall(
                        f"call_wm_adv_{internal_turn}",
                        "update_world_model",
                        {"op": "promote_to_proven", "key": "collision_safety", "description": "3x3 neighborhood verified clear before advance"}
                    ),
                    _MockToolCall(
                        f"call_env_{internal_turn}",
                        "env_step",
                        {"action": "ACTION4", "thought": "Advance exploration with verified collision safety"}
                    )
                ]
            else:
                tool_calls = [
                    _MockToolCall(
                        f"call_env_{internal_turn}",
                        "env_step",
                        {"action": "ACTION4", "thought": "Advance exploration toward right side (ACTION4)"}
                    )
                ]
        else:
            tool_calls = [
                _MockToolCall(
                    f"call_env_{internal_turn}",
                    "env_step",
                    {"action": "ACTION3", "thought": "Advance exploration toward left side (ACTION3)"}
                )
            ]

        msg = _MockMessage(content="Analyzing grid state and formulating next step.", tool_calls=tool_calls)
        return _MockResponse(_MockChoice(msg))

    def run_episode(self, max_env_steps: int = 300, max_internal_turns_per_step: int = 8) -> StepOutcome:
        outcome = self.env_bridge.reset()
        self.repl.sync_env_state(
            raw_frames=outcome.frames,
            action_name=None,
            level=outcome.levels_completed,
            status=outcome.state_status,
        )
        self._settled_frames_history = [self.repl.namespace["state"][-1].copy().astype(np.uint8)]

        self.scratchpad_messages.append({
            "role": "user",
            "content": (
                "Episode initialized. `state` (16, 64, 64) is loaded in the REPL. "
                "Start with OBSERVE using `repl_exec` before taking your first `env_step`."
            )
        })

        internal_streak = 0
        empty_call_streak = 0
        internal_turn_counter = 0

        while outcome.env_steps_total < max_env_steps and outcome.state_status != "WIN":
            internal_turn_counter += 1
            messages = self._build_messages(outcome)

            if self.dry_run or self.client is None:
                response = self._get_dry_run_completion(internal_turn_counter, outcome)
            else:
                response = self._call_llm_with_retry(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    tools=TASK_AGENT_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                )

            msg = response.choices[0].message
            assistant_dict: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            self.scratchpad_messages.append(assistant_dict)

            if not msg.tool_calls:
                empty_call_streak += 1
                if empty_call_streak >= 5:
                    break
                elif empty_call_streak >= 2:
                    self.scratchpad_messages.append({
                        "role": "user",
                        "content": (
                            "No tool call detected. You must invoke a tool: `repl_exec` to inspect the tensor, "
                            "`update_world_model` to record hypotheses, or `env_step` to act."
                        )
                    })
                else:
                    self.scratchpad_messages.append({
                        "role": "user",
                        "content": "Please call one of the available tools: `repl_exec`, `update_world_model`, or `env_step`."
                    })
                continue

            empty_call_streak = 0
            pending_vision_strips: List[str] = []

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                if fn_name == "repl_exec":
                    internal_streak += 1
                    code = args.get("code", "")
                    output = self.repl.execute(code)
                    self._emit_trace("repl_exec", {"code": code, "output": output}, outcome)

                    self.scratchpad_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    })

                    b64_img = self.repl.pop_rendered_strip_base64()
                    if b64_img and self.enable_vision:
                        pending_vision_strips.append(b64_img)

                elif fn_name == "update_world_model":
                    internal_streak += 1
                    res = self.world_model.update(
                        op=args.get("op", ""),
                        key=args.get("key", ""),
                        description=args.get("description", ""),
                    )
                    self._emit_trace("update_world_model", args, outcome)
                    self.scratchpad_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": res,
                    })

                elif fn_name == "env_step":
                    internal_streak = 0
                    action_name = args.get("action", "ACTION1")
                    prev_levels = outcome.levels_completed

                    outcome = self.env_bridge.step(
                        action_name=action_name,
                        x=args.get("x"),
                        y=args.get("y"),
                        thought=args.get("thought", ""),
                    )
                    self.repl.sync_env_state(
                        raw_frames=outcome.frames,
                        action_name=action_name,
                        level=outcome.levels_completed,
                        status=outcome.state_status,
                    )
                    self._settled_frames_history.append(self.repl.namespace["state"][-1].copy().astype(np.uint8))

                    frame_hash = hashlib.md5(self.repl.namespace["state"][-1].tobytes()).hexdigest()[:12]
                    prev_hash = hashlib.md5(self.repl.namespace["prev_state"][-1].tobytes()).hexdigest()[:12]
                    grid_changed = bool(frame_hash != prev_hash)

                    if outcome.levels_completed > prev_levels:
                        step_summary = (
                            f"🎉 LEVEL COMPLETED! Advanced from Level {prev_levels} to Level {outcome.levels_completed}! "
                            f"Status={outcome.state_status}, SubFramesCaptured={len(outcome.frames)}. "
                            f"Environment layout and mechanics may have changed. "
                            f"CRITICAL: Re-run OBSERVE via `repl_exec` on the new level layout before moving!"
                        )
                    else:
                        step_summary = (
                            f"Executed {action_name}. Status={outcome.state_status}, "
                            f"LevelsCompleted={outcome.levels_completed}, "
                            f"SubFramesCaptured={len(outcome.frames)}, "
                            f"LevelSteps={outcome.env_steps_current_level}. "
                            f"REPL `state` (16, 64, 64) updated."
                        )

                    if outcome.state_status == "GAME_OVER":
                        step_summary += (
                            "\n⚠️ ALERT: GAME OVER! The last action resulted in defeat. "
                            "1) Log a fatal transition rule with `update_world_model(op='add_fatal', ...)` so you never repeat this. "
                            "2) Call `env_step(action='RESET', thought='Resetting level after game over')` to resume play."
                        )

                    self._emit_trace(
                        "env_step",
                        {
                            **args,
                            "summary": step_summary,
                            "state_hash": frame_hash,
                            "prev_state_hash": prev_hash,
                            "grid_changed": grid_changed,
                        },
                        outcome,
                    )
                    self.scratchpad_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": step_summary,
                    })

                    if outcome.env_steps_total >= max_env_steps:
                        break

                elif fn_name == "env_step_macro":
                    internal_streak = 0
                    macro_actions = args.get("actions", [])
                    if not isinstance(macro_actions, list):
                        macro_actions = [str(macro_actions)]
                    macro_actions = macro_actions[:15]
                    stop_on_noop = args.get("stop_on_noop", True)
                    macro_thought = args.get("thought", "")

                    executed_count = 0
                    early_stop_reason: Optional[str] = None

                    for idx, act in enumerate(macro_actions):
                        if outcome.env_steps_total >= max_env_steps:
                            early_stop_reason = f"Max step budget ({max_env_steps}) reached."
                            break

                        action_name = str(act).upper()
                        prev_levels = outcome.levels_completed

                        outcome = self.env_bridge.step(
                            action_name=action_name,
                            thought=f"[Macro {idx+1}/{len(macro_actions)}] {macro_thought}",
                        )
                        self.repl.sync_env_state(
                            raw_frames=outcome.frames,
                            action_name=action_name,
                            level=outcome.levels_completed,
                            status=outcome.state_status,
                        )
                        executed_count += 1
                        self._settled_frames_history.append(self.repl.namespace["state"][-1].copy().astype(np.uint8))

                        frame_hash = hashlib.md5(self.repl.namespace["state"][-1].tobytes()).hexdigest()[:12]
                        prev_hash = hashlib.md5(self.repl.namespace["prev_state"][-1].tobytes()).hexdigest()[:12]
                        grid_changed = bool(frame_hash != prev_hash)

                        sub_summary = (
                            f"Macro sub-step {idx+1}/{len(macro_actions)} ({action_name}): Status={outcome.state_status}, "
                            f"LevelsCompleted={outcome.levels_completed}, GridChanged={grid_changed}."
                        )
                        self._emit_trace(
                            "env_step",
                            {
                                "action": action_name,
                                "thought": macro_thought,
                                "summary": sub_summary,
                                "state_hash": frame_hash,
                                "prev_state_hash": prev_hash,
                                "grid_changed": grid_changed,
                                "via_macro": True,
                                "macro_index": idx,
                                "macro_total": len(macro_actions),
                            },
                            outcome,
                        )

                        if outcome.state_status == "WIN":
                            early_stop_reason = f"🎉 WIN achieved at sub-step {idx+1}!"
                            break
                        if outcome.state_status == "GAME_OVER":
                            early_stop_reason = f"⚠️ GAME_OVER encountered at sub-step {idx+1} ({action_name})!"
                            break
                        if outcome.levels_completed > prev_levels:
                            early_stop_reason = f"🎉 Level advance! Progressed to Level {outcome.levels_completed} at sub-step {idx+1}."
                            break
                        if stop_on_noop and not grid_changed:
                            early_stop_reason = f"No-op wall bump at sub-step {idx+1} ({action_name}) with zero grid change."
                            break

                    macro_summary = (
                        f"Executed {executed_count}/{len(macro_actions)} macro actions. Status={outcome.state_status}, "
                        f"LevelsCompleted={outcome.levels_completed}, LevelSteps={outcome.env_steps_current_level}. "
                        f"Details: {early_stop_reason or 'All planned macro actions completed successfully.'}"
                    )
                    self.scratchpad_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": macro_summary,
                    })

                    if outcome.env_steps_total >= max_env_steps:
                        break

            # Attach visual strips after all tool responses to keep OpenAI tool-response ordering intact
            for b64_img in pending_vision_strips:
                self.scratchpad_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Rendered 16-frame visual strip (`F0`..`F15`):"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
                        },
                    ],
                })

            if internal_streak >= max_internal_turns_per_step:
                self.scratchpad_messages.append({
                    "role": "user",
                    "content": (
                        f"You have made {internal_streak} consecutive internal calls without stepping. "
                        "Synthesize your current hypothesis and execute an `env_step` now."
                    )
                })
                internal_streak = 0

        # Save trajectory frames archive for visual replay inspection
        save_dir = self.run_dir or self.repl.artifact_dir.parent
        if save_dir and self._settled_frames_history:
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
                tensor = np.stack(self._settled_frames_history, axis=0)
                np.savez_compressed(save_dir / "trajectory_frames.npz", settled_frames=tensor)
            except Exception:
                pass

        return outcome

    def _emit_trace(self, event_type: str, payload: Dict[str, Any], outcome: StepOutcome) -> None:
        if self.trace_callback:
            self.trace_callback({
                "event": event_type,
                "env_steps_total": outcome.env_steps_total,
                "levels_completed": outcome.levels_completed,
                "state_status": outcome.state_status,
                "payload": payload,
                "world_model_snapshot": {
                    "hypotheses": dict(self.world_model.unverified_hypotheses),
                    "proven": dict(self.world_model.proven_mechanics),
                    "fatal": list(self.world_model.fatal_transitions),
                },
            })
