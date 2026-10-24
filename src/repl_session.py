import io
import base64
import contextlib
import traceback
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from PIL import Image, ImageDraw

# Standard 16-color ARC-AGI-3 RGB Palette (0..15)
ARC_PALETTE_RGB = np.array([
    [0, 0, 0],         # 0: Black
    [0, 116, 217],     # 1: Blue
    [255, 65, 54],     # 2: Red
    [46, 204, 64],     # 3: Green
    [255, 220, 0],     # 4: Yellow
    [170, 170, 170],   # 5: Gray
    [240, 18, 190],    # 6: Magenta
    [255, 133, 27],    # 7: Orange
    [127, 219, 255],   # 8: Light Blue / Cyan
    [135, 12, 37],     # 9: Maroon
    [57, 204, 204],    # 10: Teal
    [177, 13, 201],    # 11: Purple
    [255, 255, 255],   # 12: White
    [1, 255, 112],     # 13: Lime
    [221, 221, 221],   # 14: Light Gray
    [102, 51, 0],      # 15: Brown
], dtype=np.uint8)


class RLMReplSession:
    """
    Persistent Python execution environment holding the (16, 64, 64) `state`
    tensor out-of-context, plus `render_frame_strip()` for on-demand vision inspection.
    """
    def __init__(self, artifact_dir: Path, max_output_chars: int = 3500):
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.max_output_chars = max_output_chars
        self.last_rendered_strip: Optional[Path] = None

        self.namespace: Dict[str, Any] = {
            "np": np,
            "state": np.zeros((16, 64, 64), dtype=np.int32),      # Always (16, 64, 64)
            "prev_state": np.zeros((16, 64, 64), dtype=np.int32), # Previous step's (16, 64, 64)
            "num_valid_frames": 0,                                # Actual sub-frames returned in last step
            "history": [],                                        # List of transition metadata
            "step_count": 0,
            "current_level": 0,
            "render_frame_strip": self.render_frame_strip,
            "diff_summary": self.diff_summary,
            "find_clusters": self.find_clusters,
            "bfs_shortest_path": self.bfs_shortest_path,
        }

    @staticmethod
    def _normalize_to_16_frames(raw_frames: Any) -> np.ndarray:
        """
        Normalizes a variable-length frame structure (single 2D grid, 3D array,
        list of 2D/3D frames, or FrameData raw nested lists) into a fixed (16, 64, 64) tensor.
        Pads leading slots with frames[0] so `state[-1]` is always the latest settled frame.
        """
        if raw_frames is None:
            return np.zeros((16, 64, 64), dtype=np.int32)

        frame_list: List[np.ndarray] = []

        if isinstance(raw_frames, np.ndarray):
            if raw_frames.size == 0:
                return np.zeros((16, 64, 64), dtype=np.int32)
            if raw_frames.ndim == 2:
                frame_list = [raw_frames]
            elif raw_frames.ndim == 3:
                frame_list = [raw_frames[i] for i in range(raw_frames.shape[0])]
            elif raw_frames.ndim == 4:
                frame_list = [np.squeeze(raw_frames[i]) for i in range(raw_frames.shape[0])]
            else:
                sq = np.squeeze(raw_frames)
                if sq.ndim == 2:
                    frame_list = [sq]
                elif sq.ndim == 3:
                    frame_list = [sq[i] for i in range(sq.shape[0])]
                else:
                    return np.zeros((16, 64, 64), dtype=np.int32)
        elif isinstance(raw_frames, list):
            if not raw_frames:
                return np.zeros((16, 64, 64), dtype=np.int32)

            # Check if list directly forms a 2D/3D array
            try:
                direct_arr = np.asarray(raw_frames, dtype=np.int32)
                if direct_arr.ndim == 2 and direct_arr.shape == (64, 64):
                    frame_list = [direct_arr]
                elif direct_arr.ndim == 3 and direct_arr.shape[1:] == (64, 64):
                    frame_list = [direct_arr[i] for i in range(direct_arr.shape[0])]
                elif direct_arr.ndim == 4 and direct_arr.shape[2:] == (64, 64):
                    frame_list = [direct_arr[i].reshape(64, 64) for i in range(direct_arr.shape[0])]
            except Exception:
                pass

            if not frame_list:
                for item in raw_frames:
                    item_arr = np.asarray(item, dtype=np.int32)
                    item_arr = np.squeeze(item_arr)
                    if item_arr.ndim == 2:
                        frame_list.append(item_arr)
                    elif item_arr.ndim == 3:
                        for i in range(item_arr.shape[0]):
                            frame_list.append(item_arr[i])

        if not frame_list:
            return np.zeros((16, 64, 64), dtype=np.int32)

        clean_frames: List[np.ndarray] = []
        for f in frame_list:
            fa = np.asarray(f, dtype=np.int32)
            if fa.shape != (64, 64):
                fa = np.squeeze(fa)
                if fa.shape != (64, 64):
                    canvas = np.zeros((64, 64), dtype=np.int32)
                    h = min(64, fa.shape[0] if fa.ndim >= 1 else 0)
                    w = min(64, fa.shape[1] if fa.ndim >= 2 else 0)
                    if h > 0 and w > 0:
                        canvas[:h, :w] = fa[:h, :w]
                    fa = canvas
            clean_frames.append(fa)

        # Slice to last 16 frames max
        clean_frames = clean_frames[-16:]

        # Pad leading slots if fewer than 16 frames
        if len(clean_frames) < 16:
            pad_count = 16 - len(clean_frames)
            first_frame = clean_frames[0].copy()
            clean_frames = [first_frame.copy() for _ in range(pad_count)] + clean_frames

        tensor = np.stack(clean_frames, axis=0)
        return tensor.astype(np.int32)

    def sync_env_state(
        self,
        raw_frames: Any,
        action_name: Optional[str],
        level: int,
        status: str
    ) -> None:
        self.namespace["prev_state"] = self.namespace["state"].copy()
        self.namespace["state"] = self._normalize_to_16_frames(raw_frames)

        if raw_frames is None:
            n_frames = 0
        elif isinstance(raw_frames, np.ndarray):
            n_frames = 1 if raw_frames.ndim == 2 else raw_frames.shape[0]
        elif isinstance(raw_frames, list):
            n_frames = len(raw_frames)
        else:
            n_frames = 1

        self.namespace["num_valid_frames"] = n_frames
        self.namespace["current_level"] = level

        if action_name is not None:
            self.namespace["step_count"] += 1
            self.namespace["history"].append({
                "step": self.namespace["step_count"],
                "action": action_name,
                "num_subframes": n_frames,
                "final_frame": self.namespace["state"][-1].copy(),
                "level": level,
                "status": status,
            })

    def render_frame_strip(self, filename: str = "strip.png", scale: int = 2) -> str:
        """
        Renders the 16 frames in `state` (16, 64, 64) horizontally into a composite PNG
        so vision-capable models can inspect motion trajectories directly.
        """
        out_path = self.artifact_dir / Path(filename).name
        state_tensor: np.ndarray = self.namespace["state"]  # (16, 64, 64)
        cell_w = 64 * scale
        cell_h = 64 * scale
        pad = 4
        header_h = 16

        total_w = 16 * (cell_w + pad) + pad
        total_h = cell_h + header_h + 2 * pad

        canvas = Image.new("RGB", (total_w, total_h), (24, 24, 24))
        draw = ImageDraw.Draw(canvas)

        for idx in range(16):
            frame_idx = np.clip(state_tensor[idx], 0, 15)
            rgb_grid = ARC_PALETTE_RGB[frame_idx]  # (64, 64, 3)
            img = Image.fromarray(rgb_grid, mode="RGB").resize(
                (cell_w, cell_h), resample=Image.Resampling.NEAREST
            )
            x0 = pad + idx * (cell_w + pad)
            y0 = pad + header_h
            canvas.paste(img, (x0, y0))
            draw.text((x0 + 2, 2), f"F{idx}", fill=(220, 220, 220))

        canvas.save(out_path)
        self.last_rendered_strip = out_path
        msg = f"Saved 16-frame composite strip to {out_path} ({total_w}x{total_h} px)."
        print(msg)
        return msg

    def pop_rendered_strip_base64(self) -> Optional[str]:
        """Returns base64 PNG data if `render_frame_strip` was called during the last REPL turn."""
        if self.last_rendered_strip and self.last_rendered_strip.exists():
            data = base64.b64encode(self.last_rendered_strip.read_bytes()).decode("ascii")
            self.last_rendered_strip = None
            return data
        return None

    def diff_summary(
        self,
        prev_frame: Optional[np.ndarray] = None,
        curr_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Computes pixel-level and entity-level deltas between prev_frame and curr_frame
        (defaults to prev_state[-1] and state[-1]). Groups changes by color, calculates
        centroid coordinates, and returns displacement vectors (dy, dx).
        """
        if prev_frame is None:
            prev_frame = self.namespace["prev_state"][-1]
        if curr_frame is None:
            curr_frame = self.namespace["state"][-1]

        diff_mask = (prev_frame != curr_frame)
        changed_count = int(np.sum(diff_mask))
        if changed_count == 0:
            msg = "[diff_summary] No pixel changes detected between frames."
            print(msg)
            return {"total_changed_pixels": 0, "by_color": {}}

        unique_colors = set(np.unique(prev_frame[diff_mask])).union(set(np.unique(curr_frame[diff_mask])))
        unique_colors.discard(0)
        if not unique_colors:
            unique_colors = {0}

        by_color: Dict[int, Any] = {}
        print(f"--- Frame Delta Summary (Total Changed Pixels: {changed_count}) ---")
        for c in sorted(unique_colors):
            prev_pts = np.argwhere(prev_frame == c)
            curr_pts = np.argwhere(curr_frame == c)

            prev_centroid = (
                round(float(np.mean(prev_pts[:, 0])), 2),
                round(float(np.mean(prev_pts[:, 1])), 2)
            ) if len(prev_pts) > 0 else None

            curr_centroid = (
                round(float(np.mean(curr_pts[:, 0])), 2),
                round(float(np.mean(curr_pts[:, 1])), 2)
            ) if len(curr_pts) > 0 else None

            disp = None
            if prev_centroid is not None and curr_centroid is not None:
                dy = round(curr_centroid[0] - prev_centroid[0], 2)
                dx = round(curr_centroid[1] - prev_centroid[1], 2)
                disp = (dy, dx)
                disp_str = f"dy={dy:+.1f}, dx={dx:+.1f}"
            else:
                disp_str = "Appeared/Disappeared"

            by_color[int(c)] = {
                "prev_count": len(prev_pts),
                "curr_count": len(curr_pts),
                "prev_centroid": prev_centroid,
                "curr_centroid": curr_centroid,
                "displacement": disp,
            }
            print(f"Color {c}: Prev Centroid={prev_centroid} -> Curr Centroid={curr_centroid} | Disp={disp_str} | Pixels: {len(prev_pts)} -> {len(curr_pts)}")

        return {"total_changed_pixels": changed_count, "by_color": by_color}

    def find_clusters(
        self,
        color: Optional[int] = None,
        grid: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Pure NumPy connected-component labeling detecting contiguous pixel blobs for
        each color (defaults to state[-1]). Returns bounding boxes and centroids.
        """
        if grid is None:
            grid = self.namespace["state"][-1]

        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        clusters: List[Dict[str, Any]] = []

        colors_to_check = [color] if color is not None else [int(c) for c in np.unique(grid) if c != 0]

        for c in colors_to_check:
            c_mask = (grid == c)
            for y in range(H):
                for x in range(W):
                    if c_mask[y, x] and not visited[y, x]:
                        q = deque([(y, x)])
                        visited[y, x] = True
                        comp_ys = [y]
                        comp_xs = [x]
                        while q:
                            cy, cx = q.popleft()
                            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and c_mask[ny, nx]:
                                    visited[ny, nx] = True
                                    comp_ys.append(ny)
                                    comp_xs.append(nx)
                                    q.append((ny, nx))

                        clusters.append({
                            "color": int(c),
                            "box": [int(min(comp_ys)), int(min(comp_xs)), int(max(comp_ys)), int(max(comp_xs))],
                            "centroid": [round(float(np.mean(comp_ys)), 2), round(float(np.mean(comp_xs)), 2)],
                            "pixels": len(comp_ys),
                        })

        clusters.sort(key=lambda item: (item["color"], -item["pixels"]))
        print(f"[find_clusters] Detected {len(clusters)} distinct entity clusters.")
        for cl in clusters[:10]:
            print(f"  * Color {cl['color']}: Box={cl['box']}, Centroid={cl['centroid']}, Pixels={cl['pixels']}")
        if len(clusters) > 10:
            print(f"  ... ({len(clusters) - 10} additional clusters omitted)")
        return clusters

    def bfs_shortest_path(
        self,
        start_yx: Any,
        goal_yx: Any,
        walkable_colors: Any = (0,),
        step_size: int = 1,
        grid: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Pure NumPy/Python BFS pathfinding from start_yx to goal_yx avoiding non-walkable colors.
        Returns ordered sequence of GameAction direction names (ACTION1..ACTION4).
        """
        if grid is None:
            grid = self.namespace["state"][-1]

        sy, sx = int(round(start_yx[0])), int(round(start_yx[1]))
        gy, gx = int(round(goal_yx[0])), int(round(goal_yx[1]))

        if (sy, sx) == (gy, gx):
            print(f"[bfs_shortest_path] Start is already at Goal ({sy},{sx}).")
            return []

        H, W = grid.shape
        visited = set([(sy, sx)])
        q = deque([(sy, sx, [])])

        moves = [
            ("ACTION1", -step_size, 0),  # UP
            ("ACTION2", step_size, 0),   # DOWN
            ("ACTION3", 0, -step_size),  # LEFT
            ("ACTION4", 0, step_size),   # RIGHT
        ]

        walkable_set = set(walkable_colors if isinstance(walkable_colors, (list, tuple, set)) else [walkable_colors])
        if 0 <= sy < H and 0 <= sx < W:
            walkable_set.add(int(grid[sy, sx]))  # Start position color is walkable to exit avatar sprite
        if 0 <= gy < H and 0 <= gx < W:
            walkable_set.add(int(grid[gy, gx]))  # Goal tile is reachable target

        while q:
            cy, cx, path = q.popleft()
            for act, dy, dx in moves:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                    if grid[ny, nx] in walkable_set or (ny, nx) == (gy, gx):
                        visited.add((ny, nx))
                        new_path = path + [act]
                        if (ny, nx) == (gy, gx):
                            print(f"[bfs_shortest_path] Route found: {len(new_path)} steps from ({sy},{sx}) to ({gy},{gx}).")
                            return new_path
                        q.append((ny, nx, new_path))

        print(f"[bfs_shortest_path] No path found from ({sy},{sx}) to ({gy},{gx}).")
        return []

    def execute(self, code: str) -> str:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(code, self.namespace)
            stdout_str = stdout_buf.getvalue()
            stderr_str = stderr_buf.getvalue()
            parts = []
            if stdout_str:
                parts.append(stdout_str)
            if stderr_str:
                parts.append(f"[stderr]:\n{stderr_str}")
            output = "\n".join(parts) if parts else ""
        except Exception:
            output = f"REPL Execution Error:\n{traceback.format_exc()}"

        if not output.strip():
            output = "[Executed successfully with no stdout. Use print() to inspect values.]"

        if len(output) > self.max_output_chars:
            half = self.max_output_chars // 2
            output = (
                output[:half]
                + f"\n... [TRUNCATED {len(output) - self.max_output_chars} CHARS] ...\n"
                + output[-half:]
            )
        return output
