# mri_train/explore.py

from dataclasses import dataclass
from typing import List, Optional, Dict
import os, sys, logging, time as _time
import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from bb_meta_imitation_learning.env.maze_task import MazeTaskManager
from .phase1_shaping import Phase1ShapingWrapper

logger = logging.getLogger(__name__)

@dataclass
class ExploreRollout:
    obs6: torch.Tensor           # (T,6,H,W)  (CPU)
    actions: torch.Tensor        # (T,)       (CPU)
    rewards: torch.Tensor        # (T,)       (CPU)
    beh_logits: torch.Tensor     # (T,A)      (CPU)
    reuse_count: int

def make_base_env():
    try:
        env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
    except ModuleNotFoundError:
        import importlib
        try:
            pkg = importlib.import_module("bb_meta_imitation_learning.env")
            sys.modules.setdefault("env", pkg)
            sys.modules.setdefault("env.maze_env", importlib.import_module("bb_meta_imitation_learning.env.maze_env"))
            env = gym.make("MetaMazeDiscrete3D-v0", enable_render=False)
        except Exception:
            from bb_meta_imitation_learning.env.maze_env import MetaMazeDiscrete3D
            env = MetaMazeDiscrete3D(enable_render=False)
    return env

def _make_env_fn_with_task(task_cfg: MazeTaskManager.TaskConfig):
    def _thunk():
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
        env = make_base_env()
        env.unwrapped.set_task(task_cfg)
        env = Phase1ShapingWrapper(env)  # shaping only affects phase-1
        return env
    return _thunk

def collect_explore_vec(
    policy_net,
    task_cfgs: List[MazeTaskManager.TaskConfig],
    device,
    max_steps: int = 250,
    seed_base: Optional[int] = None,
    dbg=False, dbg_timing=True, dbg_level="INFO"
) -> List[ExploreRollout]:
    n = len(task_cfgs)
    if n == 0:
        return []
    vec_env = gym.vector.AsyncVectorEnv([_make_env_fn_with_task(cfg) for cfg in task_cfgs], shared_memory=False)

    t0_total = _time.time()
    t_env = 0.0
    t_net = 0.0
    fwd_calls_total = 0
    fwd_calls_batched = 0
    fwd_batch_size_accum = 0

    try:
        try:
            obs_batch, _ = vec_env.reset(seed=([int(seed_base + i) for i in range(n)] if seed_base is not None else None))
        except TypeError:
            obs_batch = vec_env.reset(seed=([int(seed_base + i) for i in range(n)] if seed_base is not None else None))

        obs0 = obs_batch[0]
        if obs0.ndim == 3 and obs0.shape[-1] == 3:
            H, W = int(obs0.shape[0]), int(obs0.shape[1])
        elif obs0.ndim == 3 and obs0.shape[0] == 3:
            H, W = int(obs0.shape[1]), int(obs0.shape[2])
        else:
            H, W = int(obs0.shape[0]), int(obs0.shape[1])

        states_buf = [torch.empty((max_steps, 6, H, W), device=device, dtype=torch.float32) for _ in range(n)]
        beh_logits_buf: List[List[torch.Tensor]] = [[] for _ in range(n)]
        actions_list: List[List[int]] = [[] for _ in range(n)]
        rewards_list: List[List[float]] = [[] for _ in range(n)]
        steps_i = [0 for _ in range(n)]
        alive = [True for _ in range(n)]
        last_a = [0.0 for _ in range(n)]
        last_r = [0.0 for _ in range(n)]
        last_obs = list(obs_batch)
        n_alive = n

        while n_alive > 0:
            alive_indices = []
            for i in range(n):
                if not alive[i]:
                    continue
                obsi = last_obs[i]
                if obsi.ndim == 3 and obsi.shape[-1] == 3:
                    img_np = obsi.transpose(2, 0, 1)
                elif obsi.ndim == 3 and obsi.shape[0] == 3:
                    img_np = obsi
                else:
                    img_np = obsi.transpose(2, 0, 1)
                img = torch.from_numpy(img_np).to(device=device, dtype=torch.float32) / 255.0
                pa = torch.full((1, H, W), last_a[i], device=device, dtype=torch.float32)
                pr = torch.full((1, H, W), last_r[i], device=device, dtype=torch.float32)
                bb = torch.zeros((1, H, W), device=device, dtype=torch.float32)
                obs6_t = torch.cat([img, pa, pr, bb], dim=0)
                states_buf[i][steps_i[i]] = obs6_t
                alive_indices.append(i)

            t_net_start = _time.time()
            B_alive = len(alive_indices)
            T_lens = [steps_i[i] + 1 for i in alive_indices]
            T_max = max(T_lens)
            batch_seq = torch.zeros((B_alive, T_max, 6, H, W), device=device, dtype=torch.float32)
            for bi, i in enumerate(alive_indices):
                t = T_lens[bi]
                batch_seq[bi, -t:, :, :, :] = states_buf[i][:t]

            with torch.inference_mode():
                logits_batch, _ = policy_net(batch_seq)
                logits_last = logits_batch[:, -1, :]

            dist = torch.distributions.Categorical(logits=logits_last)
            actions_alive = dist.sample().detach().cpu().numpy().astype(np.int64)

            for bi, i in enumerate(alive_indices):
                beh_logits_buf[i].append(logits_last[bi].detach().cpu())

            t_net += (_time.time() - t_net_start)
            fwd_calls_total += 1
            fwd_batch_size_accum += B_alive
            if B_alive >= 2:
                fwd_calls_batched += 1

            t_env_start = _time.time()
            actions = np.zeros((n,), dtype=np.int64)
            for bi, i in enumerate(alive_indices):
                actions[i] = actions_alive[bi]
            obs_batch, rew_batch, term_batch, trunc_batch, info_batch = vec_env.step(actions)
            t_env += (_time.time() - t_env_start)

            for bi, i in enumerate(alive_indices):
                rewards_list[i].append(float(rew_batch[i]))
                actions_list[i].append(int(actions[i]))
                last_a[i] = float(actions[i])
                last_r[i] = float(rew_batch[i])
                last_obs[i] = obs_batch[i]
                steps_i[i] += 1
                if steps_i[i] >= max_steps or bool(term_batch[i]) or bool(trunc_batch[i]):
                    alive[i] = False
                    n_alive -= 1

            if all(si >= max_steps for si in steps_i):
                break

        rollouts: List[ExploreRollout] = []
        for i in range(n):
            T_i = steps_i[i]
            obs6 = states_buf[i][:T_i].contiguous().detach().cpu()
            actions = torch.tensor(actions_list[i], dtype=torch.long)
            rewards = torch.tensor(rewards_list[i], dtype=torch.float32)
            beh_logits = torch.stack(beh_logits_buf[i], dim=0).detach().cpu()
            rollouts.append(ExploreRollout(obs6=obs6, actions=actions, rewards=rewards, beh_logits=beh_logits, reuse_count=0))

        if dbg:
            elapsed = _time.time() - t0_total
            total_steps = sum(steps_i)
            avg_fwd_batch = (fwd_batch_size_accum / max(fwd_calls_total, 1))
            vec_ratio = 100.0 * (fwd_calls_batched / max(fwd_calls_total, 1))
            logger.info(
                "[VEC][SUMMARY] envs=%d total_steps=%d elapsed=%.2fs env=%.2fs net=%.2fs "
                "throughput=%.1f steps/s fwd_calls=%d batched_calls=%d (%.1f%%) avg_fwd_batch=%.2f",
                n, total_steps, elapsed, t_env, t_net,
                (total_steps / max(elapsed, 1e-6)), fwd_calls_total, fwd_calls_batched,
                vec_ratio, avg_fwd_batch
            )

        del states_buf, beh_logits_buf
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rollouts

    finally:
        try:
            vec_env.close()
        except Exception:
            pass
