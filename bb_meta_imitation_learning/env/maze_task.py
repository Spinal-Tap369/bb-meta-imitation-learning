# env\maze_task.py

"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
from collections import namedtuple
from numpy import random as npyrnd
from numpy.linalg import norm

class MazeTaskManager(object):

    # Configurations that decides a specific task
    TaskConfig = namedtuple("TaskConfig", ["start", "goal", "cell_walls", "cell_texts",
        "cell_size", "wall_height", "agent_height", "initial_life", "max_life",
        "step_reward", "goal_reward", "food_rewards", "food_interval"])

    def __init__(self, texture_dir, verbose=False):
        pathes = os.path.split(os.path.abspath(__file__))
        texture_dir = os.sep.join([pathes[0], texture_dir])
        texture_files = os.listdir(texture_dir)
        texture_files.sort()
        grounds = [None]
        for file_name in texture_files:
            if(file_name.find("wall") >= 0):
                grounds.append(pygame.surfarray.array3d(
                    pygame.image.load(os.sep.join([texture_dir, file_name]))))
            if(file_name.find("ground") >= 0):
                grounds[0] = pygame.surfarray.array3d(
                    pygame.image.load(os.sep.join([texture_dir, file_name])))
            if(file_name.find("ceil") >= 0):
                self.ceil = pygame.surfarray.array3d(
                    pygame.image.load(os.sep.join([texture_dir, file_name])))
            if(file_name.find("arrow") >= 0):
                self.arrow = pygame.surfarray.array3d(
                    pygame.image.load(os.sep.join([texture_dir, file_name])))
        self.grounds = numpy.asarray(grounds, dtype="float32")
        self.verbose = verbose

    @property
    def n_texts(self):
        return self.grounds.shape[0]

    def sample_task(self,
            n=15,
            allow_loops=True,
            cell_size=2.0,
            wall_height=3.2,
            agent_height=1.6,
            step_reward=-0.01,
            goal_reward=None,
            food_reward=0.50,
            initial_life=1.0,
            max_life=2.0,
            food_density=0.010,
            food_interval=100,
            crowd_ratio=0.0):

        assert n > 6, "Minimum required cells are 7"
        assert n % 2 != 0, "Cell Numbers can only be odd"
        if(self.verbose):
            print("Generating a random maze of size %dx%d, with allow_loops=%s, crowd_ratio=%f"
                  % (n, n, allow_loops, crowd_ratio))

        # Initialize a walls grid
        cell_walls = numpy.ones(shape=(n, n), dtype="int32")
        # Initialize a textures grid (same shape)
        cell_texts = numpy.random.randint(1, self.n_texts, size=(n, n))

        # Dig out passable cells in a checkerboard pattern
        for i in range(1, n, 2):
            for j in range(1, n, 2):
                cell_walls[i, j] = 0

        # Randomize a start
        s_x = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
        s_y = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
        start = (s_x, s_y)

        # Randomize a goal. Ensure start != goal
        # Also ensure it’s at least 2 steps away from start so we don’t get immediate overlap
        goal = (n - 2, n - 2)
        for _attempt in range(1000):
            e_x = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
            e_y = random.randint(0, (n - 1) // 2 - 1) * 2 + 1
            dist = numpy.sqrt((e_x - s_x)**2 + (e_y - s_y)**2)
            if (e_x, e_y) != start and dist > 2:
                goal = (e_x, e_y)
                break

        # Maze generation with prim-like approach (skip details for brevity)
        wall_dict = dict()
        path_dict = dict()
        rev_path_dict = dict()
        path_idx = 0
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if(cell_walls[i, j] > 0):
                    wall_dict[(i, j)] = 0
                else:
                    path_dict[(i, j)] = path_idx
                    rev_path_dict[path_idx] = [(i, j)]
                    path_idx += 1

        max_cell_walls = numpy.prod(cell_walls[1:-1, 1:-1].shape)
        while (len(rev_path_dict) > 1 or
               (allow_loops and numpy.sum(cell_walls[1:-1, 1:-1]) > max_cell_walls * crowd_ratio)):
            wall_list = list(wall_dict.keys())
            random.shuffle(wall_list)
            for i, j in wall_list:
                new_path_id = -1
                connected_path_id = dict()
                abandon_path_id = dict()
                max_duplicate = 1
                for d_i, d_j in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if (d_i > 0 and d_i < n and d_j > 0 and d_j < n and cell_walls[d_i, d_j] < 1):
                        pid = path_dict[(d_i, d_j)]
                        connected_path_id[pid] = connected_path_id.get(pid, 0) + 1
                        if connected_path_id[pid] > max_duplicate:
                            max_duplicate = connected_path_id[pid]
                        if new_path_id < 0:
                            new_path_id = pid
                            new_i, new_j = d_i, d_j
                        elif pid != new_path_id:
                            abandon_path_id[pid] = (d_i, d_j)
                if len(abandon_path_id) >= 1 and max_duplicate < 2:
                    break
                if len(abandon_path_id) >= 1 and max_duplicate > 1 and allow_loops:
                    break

            if new_path_id < 0:
                continue

            rev_path_dict[new_path_id].append((i, j))
            path_dict[(i, j)] = new_path_id
            cell_walls[i, j] = 0
            del wall_dict[(i, j)]

            for pid in abandon_path_id:
                rev_path_dict[new_path_id].extend(rev_path_dict[pid])
                for (xx, yy) in rev_path_dict[pid]:
                    path_dict[(xx, yy)] = new_path_id
                del rev_path_dict[pid]

        # Paint passable cells as texture=0
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if(cell_walls[i, j] < 1):
                    cell_texts[i, j] = 0

        # Default goal reward if not specified
        assert step_reward < 0, "step_reward must be < 0"
        if(goal_reward is None):
            def_goal_reward = - numpy.sqrt(n) * n * step_reward
        else:
            def_goal_reward = goal_reward
        assert def_goal_reward > 0, "goal_reward must be > 0"

        # Food points generation
        food_rewards = numpy.clip(numpy.random.rand(n, n)*food_reward, 0.10, food_reward)
        food_rewards *= 1.0 - cell_walls
        exp_food = (n - 1)*(n - 1)*food_density
        while(numpy.sum(food_rewards) > exp_food):
            mask = (numpy.random.rand(n, n) < 0.90).astype("float32")
            food_rewards *= mask

        food_interval = food_interval * (food_rewards > 1.0e-3).astype("int32")

        return MazeTaskManager.TaskConfig(
            start=start,
            goal=goal,
            cell_walls=cell_walls,
            cell_texts=cell_texts,
            cell_size=cell_size,
            step_reward=step_reward,
            goal_reward=def_goal_reward,
            wall_height=wall_height,
            agent_height=agent_height,
            initial_life=initial_life,
            max_life=max_life,
            food_rewards=food_rewards,
            food_interval=food_interval
        )

MAZE_TASK_MANAGER = MazeTaskManager("img")
MazeTaskSampler = MAZE_TASK_MANAGER.sample_task