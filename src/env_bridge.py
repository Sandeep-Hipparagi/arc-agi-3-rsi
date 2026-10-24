from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import numpy as np

try:
    import arc_agi
    from arcengine import GameAction, GameState
    HAS_ARC_AGI = True
except ImportError:
    HAS_ARC_AGI = False


@dataclass
class StepOutcome:
    frames: List[np.ndarray]      # Raw sub-frames returned by this transition (1..16)
    state_status: str             # 'NOT_FINISHED', 'WIN', 'GAME_OVER', etc.
    levels_completed: int
    available_actions: List[str]
    env_steps_total: int
    env_steps_current_level: int


class ArcEnvBridge:
    """
    Wraps `arc_agi.Arcade` and extracts multi-frame sequences per action.
    Only calls to `step()` increment the environment action counter.
    """
    def __init__(
        self,
        game_id: str = "ls20",
        operation_mode: str = "OFFLINE",
        render_mode: Optional[str] = None
    ):
        self.game_id = game_id
        self.env_steps_total = 0
        self.env_steps_current_level = 0
        self.current_level = 0
        self._last_levels_completed = 0

        if not HAS_ARC_AGI:
            raise RuntimeError(
                "Package `arc-agi` is not installed. Run `pip install arc-agi` to connect to ARC-AGI-3."
            )

        mode_upper = operation_mode.upper()
        if mode_upper == "OFFLINE":
            mode = arc_agi.OperationMode.OFFLINE
        elif mode_upper == "ONLINE":
            mode = arc_agi.OperationMode.ONLINE
        else:
            mode = arc_agi.OperationMode.NORMAL

        self.arcade = arc_agi.Arcade(operation_mode=mode)
        self.env = self.arcade.make(game_id, render_mode=render_mode)
        if self.env is None and mode == arc_agi.OperationMode.OFFLINE:
            # Fall back to NORMAL mode to attempt downloading if missing locally
            temp_arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
            self.env = temp_arcade.make(game_id, render_mode=render_mode)
            if self.env is not None:
                self.arcade = temp_arcade

        if self.env is None:
            raise ValueError(f"Failed to initialize ARC-AGI-3 environment for game_id='{game_id}'")

    def get_available_actions(self) -> List[str]:
        if self.env and hasattr(self.env, "action_space"):
            actions = [a.name for a in self.env.action_space]
            if "RESET" not in actions:
                actions.append("RESET")
            return actions
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"]

    @staticmethod
    def _extract_frames(obs: Any) -> List[np.ndarray]:
        """
        Extracts 1..16 (64, 64) frames from `arc_agi` FrameData observation.
        """
        raw = getattr(obs, "frame", None)
        if raw is None:
            return [np.zeros((64, 64), dtype=np.int32)]

        arr = np.asarray(raw, dtype=np.int32)
        if arr.ndim == 2:
            return [arr]
        elif arr.ndim == 3:
            return [arr[i] for i in range(arr.shape[0])]
        elif arr.ndim == 4:
            return [np.squeeze(arr[i]) for i in range(arr.shape[0])]
        return [np.zeros((64, 64), dtype=np.int32)]

    def reset(self) -> StepOutcome:
        obs = self.env.reset()
        self.env_steps_current_level = 0
        frames = self._extract_frames(obs)
        raw_state = getattr(obs, "state", None)
        status = getattr(raw_state, "name", str(raw_state)) if raw_state is not None else "UNKNOWN"
        levels_done = getattr(obs, "levels_completed", 0) if obs else 0
        self._last_levels_completed = levels_done

        return StepOutcome(
            frames=frames,
            state_status=status,
            levels_completed=levels_done,
            available_actions=self.get_available_actions(),
            env_steps_total=self.env_steps_total,
            env_steps_current_level=self.env_steps_current_level,
        )

    def step(
        self,
        action_name: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        thought: str = ""
    ) -> StepOutcome:
        """
        Executes a single billed environment action (`ACTION1`..`ACTION6` or `RESET`).
        """
        action_upper = action_name.upper()
        if action_upper not in GameAction.__members__:
            raise ValueError(f"Invalid GameAction: '{action_name}'. Valid actions: {list(GameAction.__members__.keys())}")
        action_enum = GameAction[action_upper]

        data: Optional[Dict[str, int]] = None
        if action_upper == "ACTION6" and x is not None and y is not None:
            data = {"x": int(x), "y": int(y)}

        obs = self.env.step(
            action_enum,
            data=data,
            reasoning={"thought": thought} if thought else None
        )

        self.env_steps_total += 1
        self.env_steps_current_level += 1

        frames = self._extract_frames(obs)
        raw_state = getattr(obs, "state", None)
        status = getattr(raw_state, "name", str(raw_state)) if raw_state is not None else "UNKNOWN"
        levels_done = getattr(obs, "levels_completed", 0) if obs else 0

        if levels_done > self._last_levels_completed:
            # Advanced to next level; reset per-level step counter
            self.env_steps_current_level = 0
            self._last_levels_completed = levels_done

        return StepOutcome(
            frames=frames,
            state_status=status,
            levels_completed=levels_done,
            available_actions=self.get_available_actions(),
            env_steps_total=self.env_steps_total,
            env_steps_current_level=self.env_steps_current_level,
        )

    def get_official_scorecard(self) -> Dict[str, Any]:
        """
        Wraps `self.arcade.get_scorecard()` so official arc_agi scoring metrics
        are captured alongside custom telemetry.
        """
        if hasattr(self, "arcade") and self.arcade is not None:
            try:
                sc = self.arcade.get_scorecard()
                if hasattr(sc, "model_dump"):
                    return sc.model_dump()
                elif hasattr(sc, "dict"):
                    return sc.dict()
                elif isinstance(sc, dict):
                    return sc
                return {"raw_scorecard": str(sc)}
            except Exception as e:
                return {"error": str(e)}
        return {}


class MockArcEnvBridge:
    """
    Deterministic 64x64 mock environment with moving red enemy and goal tile.
    Allows zero-cost offline testing and verification of the inner-loop agent harness.
    """
    def __init__(self, game_id: str = "mock_maze", win_at_step: Optional[int] = None):
        self.game_id = game_id
        self.win_at_step = win_at_step
        self.env_steps_total = 0
        self.env_steps_current_level = 0
        self.levels_completed = 0
        self.state_status = "NOT_FINISHED"
        self._last_levels_completed = 0
        self.resets = 0
        self.level_baseline_actions = [10, 20]
        self.level_actions = [0, 0]

        # Starting state coordinates
        self.player_pos = [10, 10]  # [y, x]
        self.goal_pos = [10, 14]    # [y, x] - 4 steps right from start
        self.enemy_pos = [20, 20]   # [y, x]
        self.enemy_dir = 1

    def get_available_actions(self) -> List[str]:
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"]

    def _render_frame(self) -> np.ndarray:
        grid = np.zeros((64, 64), dtype=np.int32)
        # Goal: Green (3)
        gy, gx = self.goal_pos
        grid[max(0, gy-1):min(64, gy+2), max(0, gx-1):min(64, gx+2)] = 3
        # Enemy: Red (2)
        ey, ex = self.enemy_pos
        grid[max(0, ey-1):min(64, ey+2), max(0, ex-1):min(64, ex+2)] = 2
        # Player: Blue (1)
        py, px = self.player_pos
        grid[max(0, py-1):min(64, py+2), max(0, px-1):min(64, px+2)] = 1
        return grid

    def reset(self) -> StepOutcome:
        self.env_steps_current_level = 0
        self.player_pos = [10, 10]
        self.goal_pos = [10, 14]
        self.enemy_pos = [20, 20]
        self.enemy_dir = 1
        self.state_status = "NOT_FINISHED"
        frames = [self._render_frame()]
        return StepOutcome(
            frames=frames,
            state_status=self.state_status,
            levels_completed=self.levels_completed,
            available_actions=self.get_available_actions(),
            env_steps_total=self.env_steps_total,
            env_steps_current_level=self.env_steps_current_level,
        )

    def step(
        self,
        action_name: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        thought: str = ""
    ) -> StepOutcome:
        action_upper = action_name.upper()
        self.env_steps_total += 1
        self.env_steps_current_level += 1
        current_lvl_idx = min(self.levels_completed, len(self.level_actions) - 1)
        self.level_actions[current_lvl_idx] += 1

        sub_frames: List[np.ndarray] = [self._render_frame()]

        if action_upper == "RESET":
            self.resets += 1
            self.player_pos = [10, 10]
            self.enemy_pos = [20, 20]
            self.state_status = "NOT_FINISHED"
        elif self.state_status == "GAME_OVER":
            # In GAME_OVER, only RESET restores NOT_FINISHED
            pass
        else:
            # Movement: ACTION1=UP, ACTION2=DOWN, ACTION3=LEFT, ACTION4=RIGHT
            if action_upper == "ACTION1":
                self.player_pos[0] = max(2, self.player_pos[0] - 1)
            elif action_upper == "ACTION2":
                self.player_pos[0] = min(61, self.player_pos[0] + 1)
            elif action_upper == "ACTION3":
                self.player_pos[1] = max(2, self.player_pos[1] - 1)
            elif action_upper == "ACTION4":
                self.player_pos[1] = min(61, self.player_pos[1] + 1)

            # Enemy moves
            self.enemy_pos[1] += self.enemy_dir
            if self.enemy_pos[1] >= 25:
                self.enemy_dir = -1
            elif self.enemy_pos[1] <= 15:
                self.enemy_dir = 1

            sub_frames.append(self._render_frame())

            # Check collision with enemy -> GAME_OVER
            if abs(self.player_pos[0] - self.enemy_pos[0]) <= 1 and abs(self.player_pos[1] - self.enemy_pos[1]) <= 1:
                self.state_status = "GAME_OVER"
            # Check reaching goal -> advance level or WIN
            elif self.player_pos == self.goal_pos:
                self.levels_completed += 1
                if self.levels_completed >= 2:
                    self.state_status = "WIN"
                else:
                    self.player_pos = [10, 10]
                    self.goal_pos = [10, 18]
                    self.env_steps_current_level = 0
                    self.state_status = "NOT_FINISHED"

        if self.win_at_step and self.env_steps_total >= self.win_at_step:
            self.state_status = "WIN"

        sub_frames.append(self._render_frame())

        return StepOutcome(
            frames=sub_frames,
            state_status=self.state_status,
            levels_completed=self.levels_completed,
            available_actions=self.get_available_actions(),
            env_steps_total=self.env_steps_total,
            env_steps_current_level=self.env_steps_current_level,
        )

    def get_official_scorecard(self) -> Dict[str, Any]:
        level_scores = [0.0, 0.0]
        for i in range(self.levels_completed):
            actions = max(1, self.level_actions[i])
            baseline = self.level_baseline_actions[i]
            level_scores[i] = min(115.0, round(((baseline / actions) ** 2) * 100, 2))

        completed_all = (self.state_status == "WIN" or self.levels_completed >= 2)
        score_val = sum(level_scores) / len(level_scores) if level_scores else 0.0

        run_data = {
            "id": None,
            "guid": f"mock-guid-{self.game_id}",
            "score": score_val,
            "levels_completed": self.levels_completed,
            "actions": self.env_steps_total,
            "resets": self.resets,
            "state": self.state_status,
            "completed": completed_all,
            "level_scores": level_scores,
            "level_actions": list(self.level_actions),
            "level_baseline_actions": list(self.level_baseline_actions),
            "number_of_levels": 2,
            "number_of_environments": 1,
            "message": None,
        }

        return {
            "source_url": None,
            "tags": None,
            "opaque": None,
            "card_id": f"mock-{self.game_id}",
            "api_key": None,
            "score": score_val,
            "environments": [{
                "id": self.game_id,
                "runs": [run_data],
                "score": score_val,
                "actions": self.env_steps_total,
                "levels_completed": self.levels_completed,
                "completed": completed_all,
                "level_count": 2,
                "resets": self.resets,
            }],
            "tags_scores": [],
            "competition_mode": None,
            "total_environments_completed": 1 if completed_all else 0,
            "total_environments": 1,
            "total_levels_completed": self.levels_completed,
            "total_levels": 2,
            "total_actions": self.env_steps_total,
        }
