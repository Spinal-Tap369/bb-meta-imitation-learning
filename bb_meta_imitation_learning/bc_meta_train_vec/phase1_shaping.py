# bc_meta_train_vec/phase1_shaping.py

import numpy as np
from collections import deque
import gymnasium as gym

class Phase1ShapingWrapper(gym.Wrapper):
    """
    Phase-1 reward shaping without touching the env package.
    Adds: coverage, geodesic progress, spin penalty, stuck penalty, transition bonus.
    All signals are small vs your step cost (-0.01), and computed in O(1) per step.

    Action mapping assumed: 0=Left, 1=Right, 2=Forward (for orientation deltas we use env's agent_ori).
    """

    def __init__(
        self,
        env,
        *,
        # coverage
        visit_bonus: float = 0.002,
        # progress (geodesic)
        progress_coef: float = 0.006,
        progress_clip: float = 0.02,
        use_geodesic: bool = True,
        # spin detection (windowed)
        spin_window: int = 16,
        spin_rot_thresh: float = 2.0 * np.pi,  # ~one full revolution in window
        spin_disp_thresh: float = 0.4,         # <= this many cells net displacement
        spin_penalty: float = -0.004,
        # stuck-in-cell (windowed)
        stuck_window: int = 8,
        stuck_rot_thresh: float = 0.75 * np.pi,  # must be turning while stuck
        stuck_penalty: float = -0.003,
        # phase transition
        transition_bonus: float = 0.2,
    ):
        super().__init__(env)
        self.enabled = True

        # knobs
        self.visit_bonus = float(visit_bonus)
        self.progress_coef = float(progress_coef)
        self.progress_clip = float(progress_clip)
        self.use_geodesic = bool(use_geodesic)

        self.spin_window = int(spin_window)
        self.spin_rot_thresh = float(spin_rot_thresh)
        self.spin_disp_thresh = float(spin_disp_thresh)
        self.spin_penalty = float(spin_penalty)

        self.stuck_window = int(stuck_window)
        self.stuck_rot_thresh = float(stuck_rot_thresh)
        self.stuck_penalty = float(stuck_penalty)

        self.transition_bonus = float(transition_bonus)

        # state (filled on reset)
        self._visited_mask = None          # bool (n,n)
        self._dist_map = None              # float32 (n,n)
        self._prev_d = None                # previous geodesic distance
        self._last_ori = None              # previous orientation (radians)

        # O(1) spin window stats
        self._spin_abs = deque(maxlen=self.spin_window)  # abs(dtheta) history
        self._spin_abs_sum = 0.0
        self._spin_cells = deque(maxlen=self.spin_window)  # (x,y) history

        # O(1) stuck window stats
        self._stuck_abs = deque(maxlen=self.stuck_window)  # abs(dtheta) history
        self._stuck_abs_sum = 0.0
        self._same_cell_run = 0
        self._last_cell = None

    # ————— lifecycle —————

    def set_enabled(self, flag: bool):
        self.enabled = bool(flag)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        mc = self.env.unwrapped.maze_core

        n = int(mc._cell_walls.shape[0])
        self._visited_mask = np.zeros((n, n), dtype=bool)
        sx, sy = int(mc._agent_grid[0]), int(mc._agent_grid[1])
        self._visited_mask[sx, sy] = True

        self._dist_map = self._compute_geodesic_map(mc._cell_walls, mc._goal) if self.use_geodesic else None
        self._prev_d = None

        self._last_ori = float(mc._agent_ori)

        self._spin_abs.clear();  self._spin_abs_sum = 0.0
        self._spin_cells.clear(); self._spin_cells.append((sx, sy))

        self._stuck_abs.clear(); self._stuck_abs_sum = 0.0
        self._same_cell_run = 1
        self._last_cell = (sx, sy)

        return obs, info

    # ————— shaping —————

    def step(self, action):
        mc = self.env.unwrapped.maze_core
        before_phase = mc.phase

        obs, reward, terminated, truncated, info = self.env.step(action)

        # always maintain orientation/cell history (cheap, C-backed deque ops)
        dtheta = self._angle_delta(self._last_ori, float(mc._agent_ori))
        self._last_ori = float(mc._agent_ori)
        cx, cy = int(mc._agent_grid[0]), int(mc._agent_grid[1])

        # update spin window O(1)
        if len(self._spin_abs) == self.spin_window:
            self._spin_abs_sum -= self._spin_abs[0]
        self._spin_abs.append(abs(dtheta))
        self._spin_abs_sum += self._spin_abs[-1]
        self._spin_cells.append((cx, cy))

        # update stuck window O(1)
        if len(self._stuck_abs) == self.stuck_window:
            self._stuck_abs_sum -= self._stuck_abs[0]
        self._stuck_abs.append(abs(dtheta))
        self._stuck_abs_sum += self._stuck_abs[-1]

        if not self.enabled or mc.phase != 1:
            # still keep state current; no shaping added
            return obs, reward, terminated, truncated, info

        added = 0.0

        # (1) coverage / first-visit
        if not self._visited_mask[cx, cy]:
            self._visited_mask[cx, cy] = True
            added += self.visit_bonus

        # (2) progress bonus (geodesic or skip if unreachable)
        if self._dist_map is not None:
            d_now = float(self._dist_map[cx, cy])
            if np.isfinite(d_now):
                if self._prev_d is None:
                    self._prev_d = d_now
                else:
                    delta = self._prev_d - d_now
                    added += float(np.clip(self.progress_coef * delta, -self.progress_clip, self.progress_clip))
                    self._prev_d = d_now

        # (3) spin penalty (large rotation, tiny displacement, within window)
        if len(self._spin_cells) == self.spin_window and self._spin_abs_sum >= self.spin_rot_thresh:
            x0, y0 = self._spin_cells[0]
            disp = np.hypot(cx - x0, cy - y0)
            if disp <= self.spin_disp_thresh:
                added += self.spin_penalty

        # (4) stuck-in-cell penalty (same cell many steps, while turning)
        if self._last_cell == (cx, cy):
            self._same_cell_run += 1
        else:
            self._same_cell_run = 1
            self._last_cell = (cx, cy)

        if self._same_cell_run >= self.stuck_window and self._stuck_abs_sum >= self.stuck_rot_thresh:
            added += self.stuck_penalty

        # (5) phase-1 → phase-2 transition bonus
        if before_phase == 1 and mc.phase == 2 and self.transition_bonus > 0.0:
            added += self.transition_bonus

        if added != 0.0:
            info = dict(info)
            info["shaping"] = info.get("shaping", 0.0) + added

        return obs, reward + added, terminated, truncated, info

    # ————— helpers —————

    @staticmethod
    def _angle_delta(prev: float, cur: float) -> float:
        d = cur - prev
        # wrap to [-pi, pi]
        if d > np.pi:
            d -= 2.0 * np.pi
        elif d < -np.pi:
            d += 2.0 * np.pi
        return d

    @staticmethod
    def _compute_geodesic_map(walls: np.ndarray, goal) -> np.ndarray:
        """BFS from goal on the free grid. O(n^2) per reset; n is maze width."""
        n = int(walls.shape[0])
        dist = np.full((n, n), np.inf, dtype=np.float32)
        gx, gy = int(goal[0]), int(goal[1])
        if not (0 <= gx < n and 0 <= gy < n) or walls[gx, gy] != 0:
            return dist
        from collections import deque
        q = deque()
        q.append((gx, gy))
        dist[gx, gy] = 0.0
        # 4-neighbors
        while q:
            x, y = q.popleft()
            nd = dist[x, y] + 1.0
            # unrolled neighbors (branchless checks are overkill here; grid is tiny)
            if x+1 < n and walls[x+1, y] == 0 and nd < dist[x+1, y]:
                dist[x+1, y] = nd; q.append((x+1, y))
            if x-1 >= 0 and walls[x-1, y] == 0 and nd < dist[x-1, y]:
                dist[x-1, y] = nd; q.append((x-1, y))
            if y+1 < n and walls[x, y+1] == 0 and nd < dist[x, y+1]:
                dist[x, y+1] = nd; q.append((x, y+1))
            if y-1 >= 0 and walls[x, y-1] == 0 and nd < dist[x, y-1]:
                dist[x, y-1] = nd; q.append((x, y-1))
        return dist
