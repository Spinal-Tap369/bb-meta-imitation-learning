# env/maze_env.py

import numpy
import gymnasium as gym
import pygame
from gymnasium.utils import seeding
from gymnasium import spaces 
from env.maze_discrete_3d import MazeCoreDiscrete3D  

# Define discrete actions: mapping from an action index to a tuple (turn, step)
# Here, we interpret:
#   - 0: turn left (i.e. (-1, 0))
#   - 1: turn right (i.e. (1, 0))
#   - 3: step forward (i.e. (0, 1))
# DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] # only need 3 actions
DISCRETE_ACTIONS = [(-1, 0), (1, 0), (0, 1)]

class MetaMazeDiscrete3D(gym.Env):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            resolution=(320, 320),
            max_steps=5000,
            task_type="SURVIVAL",
            collision_penalty=-0.001,
            phase_step_limit=400 
            ):
        super(MetaMazeDiscrete3D, self).__init__()
        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.collision_penalty = collision_penalty
        self.maze_core = MazeCoreDiscrete3D(
            resolution_horizon=resolution[0],
            resolution_vertical=resolution[1],
            max_steps=max_steps,
            task_type=task_type,
            phase_step_limit=phase_step_limit,
            collision_penalty=collision_penalty
        )
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=255.0, 
            shape=(resolution[0], resolution[1], 3), 
            dtype=numpy.float32
        )
        self.need_reset = True
        self.need_set_task = True
        self.seed()

    def seed(self, seed=None):
        from gymnasium.utils import seeding
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False

    def reset(self, seed=None, options=None):
        if self.need_set_task:
            raise Exception('Must call "set_task" before reset')
        observation = self.maze_core.reset()
        info = {}
        if self.enable_render:
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
            print("Environment has been reset and rendering initialized.")
        self.need_reset = False
        self.key_done = False
        return observation, info

    def step(self, action=None):
        if self.need_reset:
            raise Exception('Must "reset" before doing any actions')
        if action is None:
            pygame.time.delay(100)
            action = self.maze_core.movement_control(pygame.key.get_pressed())
            print(f"Keyboard Action: {action}")
        else:
            action = DISCRETE_ACTIONS[action]
        try:
            reward, done, collision = self.maze_core.do_action(action)
            if done:
                self.need_reset = True
                terminated = True
                truncated = False
                phase1 = self.maze_core.phase_metrics[1]
                phase2 = self.maze_core.phase_metrics[2]
                phase1_total = phase1["step_rewards"] + phase1["goal_rewards"] + phase1["collision_rewards"]
                phase2_total = phase2["step_rewards"] + phase2["goal_rewards"] + phase2["collision_rewards"]
                reports = {
                    "Phase1": {
                        "Goal Rewards": phase1["goal_rewards"],
                        "Total Steps": phase1["steps"],
                        "Total Step Rewards": phase1["step_rewards"],
                        "Total Collisions": phase1["collisions"],
                        "Total Collision Rewards": phase1["collision_rewards"],
                        "Total Reward": phase1_total
                    },
                    "Phase2": {
                        "Goal Rewards": phase2["goal_rewards"],
                        "Total Steps": phase2["steps"],
                        "Total Step Rewards": phase2["step_rewards"],
                        "Total Collisions": phase2["collisions"],
                        "Total Collision Rewards": phase2["collision_rewards"],
                        "Total Reward": phase2_total
                    }
                }
            else:
                terminated = False
                truncated = False
                reports = {}
            info = {
                "steps": self.maze_core.steps, 
                "collision": collision, 
                "phase": self.maze_core.phase,
                "agent_grid": self.maze_core._agent_grid 
            }
            if done:
                info["phase_reports"] = reports
            return self.maze_core.get_observation(), reward, terminated, truncated, info
        except Exception as e:
            print(f"Error during environment step: {e}")
            raise

    def render(self, mode="human"):
        if mode != "human":
            raise NotImplementedError("Only 'human' mode is supported")
        done, keys = self.maze_core.render_update()
        if done:
            print("Render Window Closed by User.")

    def save_trajectory(self, file_name):
        self.maze_core.render_trajectory(file_name)
        print(f"Trajectory saved to {file_name}.")