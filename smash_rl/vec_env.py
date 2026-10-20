"""Many SmashEnvs stepped in lockstep: one worker process and one Dolphin each.

Workers run smash_rl.env.env_worker, which imports neither PyTorch nor SB3, so each
costs ~50 MB; the main process holds the model and acts for all games at once.
"""

import contextlib
import multiprocessing as mp
import os
import sys
import types

import numpy as np

from smash_rl.dolphin import kill_children_on_exit
from smash_rl.env import CloudpickledFn, env_worker, make_env, step_and_autoreset


@contextlib.contextmanager
def _lean_spawn():
    """Start spawned processes without the main script and with single-threaded BLAS.

    A spawned process first re-imports the main script (as __mp_main__), so a training
    script's `import torch` would land in every worker; hiding __main__ while the workers
    start skips that (they only need smash_rl.env). numpy's OpenBLAS also commits memory
    for one buffer per CPU thread in every process, which the workers don't need.
    """
    main = sys.modules["__main__"]
    blas = os.environ.get("OPENBLAS_NUM_THREADS")
    sys.modules["__main__"] = types.ModuleType("__main__")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        sys.modules["__main__"] = main
        if blas is None:
            del os.environ["OPENBLAS_NUM_THREADS"]
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = blas


def stack_obs(observations) -> dict:
    return {
        "players": np.stack([o["players"] for o in observations]),
        "stage": np.array([o["stage"] for o in observations], dtype=np.int64),
        "frame": np.array([o["frame"] for o in observations], dtype=np.int64),
    }


def _stack_steps(results):
    obs, rewards, dones, infos = zip(*results)
    return stack_obs(obs), np.array(rewards, dtype=np.float32), np.array(dones, dtype=bool), list(infos)


class SubprocVecEnv:
    """Envs in non-daemon worker processes (libmelee starts a helper process per env,
    and daemon processes may not have children)."""

    def __init__(self, env_fns, start_method: str = "spawn"):
        self.n = len(env_fns)
        kill_children_on_exit()  # Windows: workers (and their Dolphins) die with the main process
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.processes = []
        with _lean_spawn():
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
                process = ctx.Process(target=env_worker, args=(work_remote, remote, CloudpickledFn(env_fn)),
                                      daemon=False)
                process.start()
                self.processes.append(process)
                work_remote.close()
        self.closed = False

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return stack_obs([obs for obs, _ in results]), [info for _, info in results]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", np.asarray(action)))
        return _stack_steps([remote.recv() for remote in self.remotes])

    def env_method(self, name, *args, **kwargs) -> list:
        for remote in self.remotes:
            remote.send(("env_method", (name, args, kwargs)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join()
        self.closed = True


class LocalVecEnv:
    """The same interface with the envs in this process (one game, or debugging)."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n = len(self.envs)

    def reset(self):
        results = [env.reset() for env in self.envs]
        return stack_obs([obs for obs, _ in results]), [info for _, info in results]

    def step(self, actions):
        return _stack_steps([step_and_autoreset(env, a) for env, a in zip(self.envs, actions)])

    def env_method(self, name, *args, **kwargs) -> list:
        return [getattr(env, name)(*args, **kwargs) for env in self.envs]

    def close(self):
        for env in self.envs:
            env.close()


def make_vec_env(n_envs: int = 4, base_port: int = 51441, subprocess: bool = True, **env_kwargs):
    """n games (one Dolphin each) on Slippi ports base_port .. base_port + n - 1."""
    thunks = [make_env(i, base_port, **env_kwargs) for i in range(n_envs)]
    if subprocess and n_envs > 1:
        return SubprocVecEnv(thunks)
    return LocalVecEnv(thunks)
