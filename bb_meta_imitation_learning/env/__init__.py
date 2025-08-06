# env\__init__.py

from gymnasium.envs.registration import register

register(
    id="MetaMazeDiscrete3D-v0",
    entry_point="env.maze_env:MetaMazeDiscrete3D",
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (40, 60),
        "max_steps": 500,
        "task_type": "ESCAPE",
        "phase_step_limit": 250,  # Define steps per phase
        "collision_penalty": -0.005  # Define collision penalty
    }
)

