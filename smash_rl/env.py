"""Super Smash Bros. Melee environment: one Dolphin, one decision per frame.

The agent controls port 1. Each step takes one controller as class indices
(smash_rl.codec) and advances one frame; a decision takes effect `delay` frames
after the frame it was made on, like a human's reaction time. Observations are
raw frames (smash_rl.state) of [agent, opponent] plus the stage index and frame
number; smash_rl.codec turns them into model inputs.

One episode is one full game. Between games the env walks the menus itself; with
one stage and a fixed opponent (windowed) a finished game restarts instantly instead.
The reward is smash_rl.reward.Reward when `reward` gives its weights, else 0.
"""

import collections
import copy
import itertools
import signal
import weakref

import cloudpickle
import gymnasium as gym
import melee
import numpy as np

from smash_rl import codec
from smash_rl.dolphin import DEFAULT_ISO, Dolphin
from smash_rl.reward import Reward
from smash_rl.state import from_libmelee, stage_index

IN_GAME = (melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH)
# Opponents for opponent="RANDOM": everyone on the character select screen. The CPU
# can't be made to pick Sheik, but a CPU Zelda may transform into her mid-game.
RANDOM_OPPONENTS = tuple(c for c in melee.Character
                         if c.value < 27 and c not in (melee.Character.SHEIK, melee.Character.NANA))
_LIVE_ENVS = weakref.WeakSet()  # envs in this process, closed when a worker exits


class DelayQueue:
    """Applies each decision `delay` frames after the frame it was made on (delay >= 1)."""

    def __init__(self, delay: int):
        if delay < 1:
            raise ValueError(f"delay must be at least 1 frame, got {delay}")
        self.delay = delay
        self.reset()

    def reset(self):
        self._queue = collections.deque(codec.NEUTRAL.copy() for _ in range(self.delay - 1))

    def push(self, action) -> np.ndarray:
        self._queue.append(np.asarray(action, dtype=np.int64))
        return self._queue.popleft()


class SmashEnv(gym.Env):
    """Agent (port 1) vs a CPU, a person, a bot or a second policy (port 2), one frame per step.

    opponent: a character name for a CPU at cpu_level, "RANDOM" for a new CPU character
    every game, "HUMAN" for a person with a Dolphin-mapped controller on port 2, or
    "POLICY" for self-play: port 2 plays `character` too, and each step takes two
    controllers (2, 4) [port 1, port 2], each applied `delay` frames later.
    opponent_bot: a bot that plays port 2 instead (e.g. smash_rl.ganondorf.GanondorfBot,
    with .character, .reset() and .act(gamestate, controller)); `opponent` is then ignored.
    stages: one or more stage names; with several, each game picks one at random.
    reward: keyword arguments of smash_rl.reward.Reward (None: reward 0).
    save_replays: a directory where Dolphin saves a Slippi replay of every game.
    """

    metadata = {"render_modes": []}

    def __init__(self, character: str = "FOX", opponent: str = "FOX", stages=("FINAL_DESTINATION",),
                 cpu_level: int = 9, delay: int = 18, port: int = 51441, headless: bool = False,
                 dolphin_path: str = None, dolphin_home: str = None, iso_path: str = DEFAULT_ISO,
                 gfx_backend: str = "OGL", gfx_settings: dict = None, emulation_speed: float = 0.0,
                 max_menu_frames: int = 6000, connect_timeout: float = 120.0, save_replays: str = None,
                 opponent_bot=None, reward: dict = None):
        super().__init__()
        self.character = melee.Character[character.upper()]
        # Each env gets its own copy: in-process envs share their constructor arguments.
        self.opponent_bot = copy.deepcopy(opponent_bot)
        if self.opponent_bot is not None:
            opponent = self.opponent_bot.character
        self.policy_opponent = opponent.upper() == "POLICY"
        if self.policy_opponent:
            opponent = character
        self.random_opponent = opponent.upper() == "RANDOM"
        self.human_opponent = opponent.upper() == "HUMAN"
        self.opponent = None if self.random_opponent or self.human_opponent else melee.Character[opponent.upper()]
        self.stages = tuple(melee.Stage[s.upper()] for s in ([stages] if isinstance(stages, str) else stages))
        self.stage = self.stages[0]
        self.cpu_level = int(cpu_level)
        self.port = port
        self.max_menu_frames = max_menu_frames
        # Instant Match restarts a finished game without the menus and keeps the stage, so it
        # needs a fixed CPU and one stage; headless never uses it (it crashes the ExiAI build),
        # nor does saving replays (Slippi only finishes a replay when the game ends normally).
        instant_restart = (not (self.random_opponent or self.human_opponent or headless or save_replays)
                           and len(self.stages) == 1)
        self.dolphin = Dolphin(
            port, headless=headless, dolphin_path=dolphin_path, dolphin_home=dolphin_home, iso_path=iso_path,
            gfx_backend=gfx_backend, gfx_settings=gfx_settings, emulation_speed=emulation_speed,
            connect_timeout=connect_timeout, human_opponent=self.human_opponent,
            instant_restart=instant_restart, save_replays=save_replays)
        self.queue = DelayQueue(delay)
        self.opponent_queue = DelayQueue(delay)
        self.reward_fn = Reward(**reward) if reward is not None else None
        self.action_space = gym.spaces.MultiDiscrete(codec.ACTION_SIZES)
        _LIVE_ENVS.add(self)
        self._gs = None
        self._players = None
        self._game_over = False
        self._running_level = None
        self._stats = {}
        self._frames = 0

    def set_cpu_level(self, level: int):
        """Takes effect from the next game (restarting Dolphin when games restart instantly)."""
        self.cpu_level = int(np.clip(level, 1, 9))

    def _go_to_game(self):
        """Navigate menus until a fresh game with the right setup has started."""
        pad, cpu_pad = self.dolphin.pad, self.dolphin.cpu_pad
        # A bot or policy plays port 2 as a human player (level 0), set once its coin is down.
        level = 0 if (self.opponent_bot is not None or self.policy_opponent) else self.cpu_level
        need_new_game = self._game_over
        last_frame = self._gs.frame if (need_new_game and self._gs is not None) else None
        # MenuHelper remembers that it already picked a stage, so use fresh ones for every visit.
        menu, cpu_menu = melee.MenuHelper(), melee.MenuHelper()
        # A person can take as long as they like on the character select screen.
        for _ in itertools.count() if self.human_opponent else range(self.max_menu_frames):
            gs = self.dolphin.next_state()
            if gs.menu_state in IN_GAME:
                pad.release_all()
                if cpu_pad is not None:
                    cpu_pad.release_all()
                if need_new_game:
                    # Wait for the previous game to end and the next one to begin.
                    if gs.frame < 0 or (last_frame is not None and gs.frame < last_frame):
                        need_new_game = False
                    last_frame = gs.frame
                    continue
                p1, p2 = gs.players.get(1), gs.players.get(2)
                if p1 is not None and p2 is not None and p1.stock > 0 and p2.stock > 0:
                    return gs
                continue
            need_new_game = False
            cpu_ready = False
            if cpu_pad is not None and gs.menu_state == melee.Menu.CHARACTER_SELECT:
                p2 = gs.players.get(2)
                cpu_ready = (p2 is not None and p2.cpu_level == level and p2.character == self.opponent
                             and (level > 0 or p2.coin_down))
                cpu_menu.menu_helper_simple(gs, cpu_pad, self.opponent, self.stage,
                                            cpu_level=level, autostart=False)
            elif cpu_pad is not None:
                cpu_pad.release_all()
            # Only start once the CPU slot is set up (a person on port 2 presses Start
            # themselves); autostart also makes port 1 pick the stage.
            menu.menu_helper_simple(gs, pad, self.character, self.stage, cpu_level=0,
                                    autostart=gs.menu_state != melee.Menu.CHARACTER_SELECT or cpu_ready)
        raise RuntimeError("stuck in menus")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_opponent:
            self.opponent = RANDOM_OPPONENTS[self.np_random.integers(len(RANDOM_OPPONENTS))]
        if len(self.stages) > 1:
            self.stage = self.stages[self.np_random.integers(len(self.stages))]
        for attempt in range(3):
            try:
                if self.dolphin.alive() and self._running_level != self.cpu_level and self.dolphin.instant_restart:
                    self.dolphin.stop()
                if not self.dolphin.alive():
                    self.dolphin.start()
                    self._running_level = self.cpu_level
                    self._game_over = False
                gs = self._go_to_game()
                break
            except Exception as e:  # dolphin crashed or hung in menus: restart it
                if self.human_opponent:
                    raise  # most likely the person closed the Dolphin window
                print(f"[port {self.port}] restarting dolphin after: {e}")
                self.dolphin.stop()
        else:
            raise RuntimeError("could not start a game after 3 attempts")
        if self.human_opponent:
            self.opponent = gs.players[2].character
        self._gs = gs
        self._players = from_libmelee(gs)
        self._game_over = False
        self._frames = 0
        self.queue.reset()
        self.opponent_queue.reset()
        if self.opponent_bot is not None:
            self.opponent_bot.reset()
        if self.reward_fn is not None:
            self.reward_fn.reset(gs.stage)
        self._prev = self._snapshot(gs)
        self._stats = {"damage_dealt": 0.0, "damage_taken": 0.0, "kills": 0, "deaths": 0}
        return self._obs(), self._info(gs, done=False)

    def step(self, action):
        action = np.asarray(action)
        if self.policy_opponent:
            codec.apply_controller(self.dolphin.cpu_pad, self.opponent_queue.push(action[1]))
            action = action[0]
        codec.apply_controller(self.dolphin.pad, self.queue.push(action))
        if self.opponent_bot is not None:  # reacts to the latest frame, without delay
            self.opponent_bot.act(self._gs, self.dolphin.cpu_pad)
        try:
            gs = self.dolphin.next_state()
        except Exception as e:
            if self.human_opponent:
                raise  # most likely the person closed the Dolphin window
            # A crashed or frozen Dolphin only ends this game: reset() starts a new one,
            # so one bad emulator doesn't stop a night of training.
            print(f"[port {self.port}] ending the game after: {e}", flush=True)
            self.dolphin.stop()
            self._game_over = True
            return self._obs(), 0.0, False, True, dict(self._info(self._gs, done=True), crashed=True)
        self._frames += 1
        # Leaving the game, a player disappearing or the frame counter going back (an
        # instantly restarted game) all end the episode; the last in-game state is kept.
        in_game = (gs.menu_state in IN_GAME and gs.frame >= self._gs.frame
                   and 1 in gs.players and 2 in gs.players)
        reward = 0.0
        if in_game:
            self._track(gs)
            players = from_libmelee(gs)
            if self.reward_fn is not None:
                reward = self.reward_fn(self._players, players)
            self._gs, self._players = gs, players
        players = self._gs.players
        terminated = not in_game or players[1].stock == 0 or players[2].stock == 0
        self._game_over = terminated
        return self._obs(), reward, terminated, False, self._info(self._gs, done=terminated)

    @staticmethod
    def _snapshot(gs):
        p1, p2 = gs.players[1], gs.players[2]
        return float(p1.percent), int(p1.stock), float(p2.percent), int(p2.stock)

    def _track(self, gs):
        my_pct, my_stock, op_pct, op_stock = self._snapshot(gs)
        prev_my_pct, prev_my_stock, prev_op_pct, prev_op_stock = self._prev
        self._prev = (my_pct, my_stock, op_pct, op_stock)
        deaths = max(0, prev_my_stock - my_stock)
        kills = max(0, prev_op_stock - op_stock)
        # Percent resets to 0 on death, so only count increases within a stock.
        self._stats["damage_taken"] += max(0.0, my_pct - prev_my_pct) if not deaths else 0.0
        self._stats["damage_dealt"] += max(0.0, op_pct - prev_op_pct) if not kills else 0.0
        self._stats["kills"] += kills
        self._stats["deaths"] += deaths

    def _obs(self) -> dict:
        return {"players": self._players, "stage": stage_index(self._gs.stage), "frame": int(self._gs.frame)}

    def _info(self, gs, done: bool) -> dict:
        info = dict(self._stats, frames=self._frames, cpu_level=self.cpu_level,
                    opponent=getattr(self.opponent, "name", str(self.opponent)))
        if done:
            me, op = gs.players[1], gs.players[2]
            info["won"] = float(op.stock < me.stock or (op.stock == me.stock and op.percent > me.percent))
            info["stock_diff"] = int(me.stock) - int(op.stock)
        return info

    def close(self):
        self.dolphin.stop()


def step_and_autoreset(env, action):
    """One step; a finished game continues with the next game's first observation
    (info["final_observation"] keeps the finished game's last one)."""
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        info["final_observation"] = obs
        obs, _ = env.reset()
    return obs, reward, done, info


def make_env(rank: int = 0, base_port: int = 51441, **env_kwargs):
    """Return a thunk that builds one Smash env with its own Dolphin port."""

    def _init():
        return SmashEnv(port=base_port + rank, **env_kwargs)

    return _init


class CloudpickledFn:
    """Sends a function (lambdas and closures too) to a worker with cloudpickle."""

    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, data):
        self.fn = cloudpickle.loads(data)


def _exit_on_signal(signum, frame):
    raise SystemExit(1)


def env_worker(remote, parent_remote, env_fn: CloudpickledFn):
    """Worker process of smash_rl.vec_env: runs one env, closes its Dolphin on exit.

    Leaves Ctrl+C / hangups to the main process. Imports neither PyTorch nor SB3, so
    each worker stays small (~50 MB).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if hasattr(signal, "SIGHUP"):  # not on Windows
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit_on_signal)
    parent_remote.close()
    try:
        env = env_fn.fn()
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break
            if cmd == "step":
                remote.send(step_and_autoreset(env, data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "env_method":
                remote.send(getattr(env, data[0])(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    finally:
        for live_env in list(_LIVE_ENVS):
            live_env.close()
