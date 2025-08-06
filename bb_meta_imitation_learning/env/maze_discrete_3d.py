# env/maze_discrete_3d.py

import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding
import copy
from copy import deepcopy
import random

from env.maze_base import MazeBase
from env.ray_caster_utils import maze_view
from env.maze_task import MAZE_TASK_MANAGER
from env.dynamics import vector_move_with_collision, PI  # Import collision functions and PI

class MazeCoreDiscrete3D(MazeBase):
    """
    Core logic for the discrete 3D maze environment.
    Handles agent movement, phase transitions, and reward calculations.

    This version uses a step penalty, a collision penalty, and a goal reward.
    It now uses collision detection from dynamics.py (via vector_move_with_collision).
    
    Expected action: a tuple (turn, step) where:
      - turn: -1 for turning left 15°, 0 for no turn, +1 for turning right 15°.
      - step: typically +1 to move forward (or -1 to move backward if allowed, but here we disable backward motion).
    
    A no-op action (0, 0) will not update the agent’s state or register a timestep.
    """
    def __init__(
            self,
            collision_dist=0.20,
            max_vision_range=12.0,
            fol_angle=0.6 * np.pi,
            resolution_horizon=320,
            resolution_vertical=320,
            max_steps=5000,
            task_type="ESCAPE",
            phase_step_limit=250,
            collision_penalty=-0.005  # adjusted collision penalty
        ):
        super(MazeCoreDiscrete3D, self).__init__(
            collision_dist=collision_dist,
            max_vision_range=max_vision_range,
            fol_angle=fol_angle,
            resolution_horizon=resolution_horizon,
            resolution_vertical=resolution_vertical,
            task_type=task_type,
            max_steps=max_steps,
            phase_step_limit=phase_step_limit
        )
        # Use a continuous angle (in radians) to represent orientation.
        self._agent_ori = 0.0  # starting orientation in radians
        self.collision_penalty = collision_penalty

        # Phase metrics for logging and reward computation.
        self.phase_metrics = {
            1: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0},
            2: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0}
        }

        # Base rewards.
        self._step_reward = -0.01         # per timestep
        self._goal_reward = 1.0           # reward for reaching the goal

    def reset(self):
        observation = super(MazeCoreDiscrete3D, self).reset()
        self._starting_position = deepcopy(self._agent_grid)
        # Reset phase metrics.
        self.phase_metrics = {
            1: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0},
            2: {"goal_rewards": 0.0, "steps": 0, "step_rewards": 0.0,
                "collisions": 0, "collision_rewards": 0.0}
        }
        return observation

    def do_action(self, action):
        """
        Executes the given (turn, step) action.
        Converts the discrete command into continuous control signals and applies
        collision-aware movement using vector_move_with_collision from dynamics.py.

        If the action is (0, 0) (i.e. no rotation and no movement), no update occurs
        and no timestep penalty is incurred.
        """
        # Check for no-op.
        if action == (0, 0):
            return 0.0, False, False

        assert isinstance(action, tuple) and len(action) == 2, "Action must be a tuple (turn, step)"
        turn_command, step_command = action

        # Use a smaller time step to reduce step size.
        dt = 0.1
        # For a 15° rotation per unit turn with dt=0.1 and turn_factor=5.0,
        # we need: turn_rate * 0.1 * 5 = 0.2618, so turn_rate ≈ 0.5236.
        turn_rate = 0.5236 * turn_command

        # Use the step command directly; with dt=0.1 and move_factor=6.0,
        # a step of 1 moves the agent approximately 0.6 cells.
        # Prevent backward motion:
        walk_speed = step_command if step_command > 0 else 0.0

        # Apply collision-aware movement.
        new_ori, new_loc, collision = vector_move_with_collision(
            self._agent_ori, self._agent_loc, turn_rate, walk_speed, dt,
            self._cell_walls, self._cell_size, self.collision_dist
        )
        self._agent_ori = new_ori
        self._agent_loc = new_loc
        self._agent_grid = self.get_loc_grid(new_loc)

        self.phase_metrics[self.phase]["steps"] += 1
        self.current_phase_steps += 1

        # Compute the reward and check termination.
        reward, done = self.evaluation_rule()

        if collision:
            reward += self.collision_penalty
            self.phase_metrics[self.phase]["collisions"] += 1
            self.phase_metrics[self.phase]["collision_rewards"] += self.collision_penalty

        # Check if the agent reached the goal.
        agent_at_goal = (tuple(self._agent_grid) == tuple(self._goal))
        if self.task_type == "ESCAPE":
            if agent_at_goal:
                if self.phase == 1:
                    self.phase_metrics[1]["goal_rewards"] += self._goal_reward
                    self.phase = 2
                    self.current_phase_steps = 0
                    self._agent_grid = deepcopy(self._starting_position)
                    self._agent_loc = self.get_cell_center(self._agent_grid)
                    done = False
                    self.update_observation()
                elif self.phase == 2:
                    done = True

        if self.phase == 1 and self.current_phase_steps >= self.phase_step_limit:
            self.phase = 2
            self.current_phase_steps = 0
            self._agent_grid = deepcopy(self._starting_position)
            self._agent_loc = self.get_cell_center(self._agent_grid)
            done = False
            self.update_observation()
        elif self.phase == 2 and self.current_phase_steps >= self.phase_step_limit:
            done = True

        self.update_observation()
        return reward, done, collision

    def evaluation_rule(self):
        self.steps += 1
        self._agent_trajectory.append(np.copy(self._agent_grid))
        agent_at_goal = (tuple(self._agent_grid) == tuple(self._goal))
        if self.task_type == "ESCAPE":
            if self.phase == 1:
                self.phase_metrics[1]["step_rewards"] += self._step_reward
                reward = self._step_reward
                done = False
            elif self.phase == 2:
                step_r = self._step_reward
                self.phase_metrics[2]["step_rewards"] += step_r
                if agent_at_goal:
                    self.phase_metrics[2]["goal_rewards"] += self._goal_reward
                    reward = step_r + self._goal_reward
                else:
                    reward = step_r
                done = agent_at_goal or self.episode_is_over()
        else:
            reward = 0.0
            done = False
        return reward, done

    # Old turn() and move() methods for reference.
    def turn(self, direction):
        delta = np.deg2rad(15)
        self._agent_ori = (self._agent_ori + direction * delta) % (2 * np.pi)

    def move(self, step):
        tmp_grid = deepcopy(self._agent_grid)
        dx = int(round(np.cos(self._agent_ori) * step))
        dy = int(round(np.sin(self._agent_ori) * step))
        tmp_grid[0] += dx
        tmp_grid[1] += dy
        if (0 <= tmp_grid[0] < self._n and 0 <= tmp_grid[1] < self._n and
                self._cell_walls[tmp_grid[0], tmp_grid[1]] == 0):
            self._agent_grid = tmp_grid
            self._agent_loc = self.get_cell_center(self._agent_grid)

    def render_init(self, view_size):
        super(MazeCoreDiscrete3D, self).render_init(view_size)
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion

    def render_observation(self):
        import pygame
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))
        agent_pos = np.array(self._agent_loc) * self._pos_conversion
        dx = self._ori_size * np.cos(self._agent_ori)
        dy = self._ori_size * np.sin(self._agent_ori)
        center_pos = (int(agent_pos[0] + self._view_size), int(self._view_size - agent_pos[1]))
        end_pos = (int(agent_pos[0] + self._view_size + dx), int(self._view_size - agent_pos[1] - dy))
        radius = int(0.15 * self._pos_conversion)
        pygame.draw.circle(self._screen, pygame.Color("green"), center_pos, radius)
        pygame.draw.line(self._screen, pygame.Color("green"), center_pos, end_pos, width=1)

    def movement_control(self, keys):
        """
        For keyboard control:
          - Left arrow: (-1, 0) to turn left 15°.
          - Right arrow: (1, 0) to turn right 15°.
          - Up arrow: (0, 1) to step forward.
          - Down arrow: returns (0, 0) so that no timestep is registered.
        """
        if keys[pygame.K_LEFT]:
            return (-1, 0)
        if keys[pygame.K_RIGHT]:
            return (1, 0)
        if keys[pygame.K_UP]:
            return (0, 1)
        # if keys[pygame.K_DOWN]:
        #     return (0, 0)
        return (0, 0)

    def update_observation(self):
        self._observation = maze_view(
            self._agent_loc,
            self._agent_ori,
            self._agent_height,
            self._cell_walls,
            self._cell_transparents,
            self._cell_texts,
            self._cell_size,
            MAZE_TASK_MANAGER.grounds,
            MAZE_TASK_MANAGER.ceil,
            self._wall_height,
            1.0,
            self.max_vision_range,
            0.20,
            self.fol_angle,
            self.resolution_horizon,
            self.resolution_vertical
        )
        self._obs_surf = pygame.surfarray.make_surface(self._observation)
        self._observation = np.clip(self._observation, 0.0, 255.0).astype("float32")

    def get_observation(self):
        return np.copy(self._observation)
    
    def randomize_start(self):
        """
        Randomizes the agent’s starting cell from among the passable cells.
        Also updates self._start so that subsequent goal sampling uses this new value.
        """
        valid_cells = [(i, j) for i in range(self._n) for j in range(self._n)
                       if self._cell_walls[i, j] == 0]
        # Randomly choose a new start.
        new_start = random.choice(valid_cells)
        self._agent_grid = np.array(new_start)
        self._agent_loc = self.get_cell_center(self._agent_grid)
        self._starting_position = copy.deepcopy(self._agent_grid)
        # Update the official start used for goal sampling.
        self._start = tuple(new_start)
        
    def randomize_goal(self, min_distance=3.0):
        """
        Randomizes the goal cell from among the passable cells such that the new goal is at
        least min_distance (in cell units) away from the current self._start.
        If no cell meets the criterion, the min_distance is relaxed gradually.
        In ESCAPE mode, update the goal indicator in _cell_transparents.
        """
        valid_cells = [(i, j) for i in range(self._n) for j in range(self._n)
                       if self._cell_walls[i, j] == 0]
        current_min_distance = min_distance
        filtered = [cell for cell in valid_cells
                    if np.linalg.norm(np.array(cell) - np.array(self._start)) >= current_min_distance]
        while not filtered:
            current_min_distance *= 0.9  # reduce by 10%
            if current_min_distance < 0.5:
                filtered = valid_cells
                break
            filtered = [cell for cell in valid_cells
                        if np.linalg.norm(np.array(cell) - np.array(self._start)) >= current_min_distance]
        new_goal = random.choice(filtered)
        self._goal = tuple(new_goal)
        # In ESCAPE mode, update the goal indicator.
        if self.task_type == "ESCAPE":
            self._cell_transparents = np.zeros_like(self._cell_walls, dtype="int32")
            self._cell_transparents[self._goal] = 1

