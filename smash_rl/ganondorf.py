"""A friend's Ganondorf bot (ganondorf_agent, Stable-Baselines3 PPO) as the port 2 opponent.

Its policy is a small MLP (847 -> 256 -> 256 -> 31 actions, ReLU) that picks a controller
every 3 frames with no reaction delay. It runs here with numpy so the env workers stay
PyTorch-free; the observation, left/right mirroring, action table and button timing copy
ganondorf_agent.py, and it plays like GanondorfAgent(deterministic=True).
"""

import io
import json
import os
import zipfile

import melee
import numpy as np

B = melee.Button
# Stick positions (x, y) in libmelee's 0..1 range, 0.5 is neutral.
N, R, L, U, D = (0.5, 0.5), (1.0, 0.5), (0.0, 0.5), (0.5, 1.0), (0.5, 0.0)
UR, UL, DR, DL = (0.85, 0.85), (0.15, 0.85), (0.85, 0.15), (0.15, 0.15)
DIRS9 = [N, R, L, U, D, UR, UL, DR, DL]
# The 31 actions (main stick, button or None); the order is part of the trained network.
ACTIONS = (
    [(d, None) for d in DIRS9]
    + [(d, B.BUTTON_A) for d in DIRS9]
    + [(d, B.BUTTON_B) for d in (N, R, L, U, D)]
    + [(d, B.BUTTON_X) for d in (N, R, L)]
    + [(N, B.BUTTON_Z)]
    + [(d, B.BUTTON_L) for d in (N, R, L, D)]
)
# Action i with the stick flipped left <-> right.
MIRRORED_ACTION = [
    next(j for j, (st, bt) in enumerate(ACTIONS) if bt == button and round(st[0], 6) == round(1.0 - stick[0], 6)
         and round(st[1], 6) == round(stick[1], 6))
    for stick, button in ACTIONS
]
# Released on the last frame of each action so they can be pressed again; shield stays held.
TAP_BUTTONS = (B.BUTTON_A, B.BUTTON_B, B.BUTTON_X, B.BUTTON_Z)
NUM_ANIMATIONS = 400
PLAYER_FEATURES = 20
# Our names for the actor's layers in the SB3 policy (the value network isn't needed).
LAYERS = {"w0": "mlp_extractor.policy_net.0.weight", "b0": "mlp_extractor.policy_net.0.bias",
          "w1": "mlp_extractor.policy_net.2.weight", "b1": "mlp_extractor.policy_net.2.bias",
          "w2": "action_net.weight", "b2": "action_net.bias"}


def _action_id(player) -> int:
    value = getattr(player.action, "value", player.action)
    return int(value) if 0 <= int(value) < NUM_ANIMATIONS else NUM_ANIMATIONS - 1


class GanondorfBot:
    """Plays `port` against `opponent_port`; call act() every in-game frame, reset() per game."""

    def __init__(self, weights: dict, meta: dict, port: int = 2, opponent_port: int = 1):
        self.weights = {name: np.asarray(weights[name], dtype=np.float32) for name in LAYERS}
        self.character = meta.get("character", "GANONDORF")
        self.port, self.opponent_port = port, opponent_port
        self.frame_skip = int(meta.get("frame_skip", 3))
        self.canonical_side = meta.get("canonical_side", "right")
        self.opponent_pool = [melee.Character[c] for c in meta.get("opponent", "FOX,MARTH,CPTFALCON").split(",")]
        self.stage = melee.Stage[meta.get("stage", "FINAL_DESTINATION")]
        self.edge = melee.stages.EDGE_GROUND_POSITION[self.stage]
        expected = 2 * (PLAYER_FEATURES + NUM_ANIMATIONS) + 4 + len(self.opponent_pool)
        if self.weights["w0"].shape[1] != expected or self.weights["w2"].shape[0] != len(ACTIONS):
            raise ValueError(f"policy has {self.weights['w0'].shape[1]} inputs and {self.weights['w2'].shape[0]} "
                             f"actions, the settings give {expected} and {len(ACTIONS)}")
        self.reset()

    @classmethod
    def load(cls, model_zip: str, port: int = 2, opponent_port: int = 1) -> "GanondorfBot":
        """Read the policy from the SB3 model .zip (needs PyTorch) and meta.json next to it."""
        import torch

        with zipfile.ZipFile(model_zip) as z:
            settings = json.loads(z.read("data"))
            # weights_only: plain tensors, nothing in the file gets executed.
            state = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
        activation = settings.get("policy_kwargs", {}).get("activation_fn", "")
        if "ReLU" not in activation:
            raise ValueError(f"expected a ReLU policy, {model_zip} has {activation or 'the default (Tanh)'}")
        meta_path = os.path.join(os.path.dirname(os.path.abspath(model_zip)), "meta.json")
        meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
        return cls({name: state[key].numpy() for name, key in LAYERS.items()}, meta, port, opponent_port)

    def reset(self):
        self._frame = 0
        self._button = None

    def logits(self, obs: np.ndarray) -> np.ndarray:
        w = self.weights
        h = np.maximum(w["w0"] @ obs + w["b0"], 0.0)
        h = np.maximum(w["w1"] @ h + w["b1"], 0.0)
        return w["w2"] @ h + w["b2"]

    def _player_features(self, p, mirror: bool) -> np.ndarray:
        f = np.zeros(PLAYER_FEATURES + NUM_ANIMATIONS, dtype=np.float32)
        sx = -1.0 if mirror else 1.0  # flips every left/right quantity
        f[:PLAYER_FEATURES] = (
            sx * p.position.x / 100.0,
            p.position.y / 100.0,
            p.percent / 100.0,
            p.stock / 4.0,
            sx * (1.0 if p.facing else -1.0),
            float(p.on_ground),
            float(p.off_stage),
            float(p.invulnerable),
            min(p.hitlag_left, 30) / 10.0,
            min(p.hitstun_frames_left, 120) / 30.0,
            p.jumps_left / 2.0,
            p.shield_strength / 60.0,
            sx * p.speed_air_x_self / 3.0,
            sx * p.speed_ground_x_self / 3.0,
            p.speed_y_self / 3.0,
            np.clip(sx * p.speed_x_attack / 5.0, -5, 5),
            np.clip(p.speed_y_attack / 5.0, -5, 5),
            min(p.action_frame, 120) / 60.0,
            (self.edge - abs(p.position.x)) / 100.0,
            float(p.iasa) if isinstance(p.iasa, (bool, int, float)) else 0.0,
        )
        f[PLAYER_FEATURES + _action_id(p)] = 1.0
        return f

    def observation(self, gs):
        """(847-float observation, mirrored?) for the current game state."""
        me, op = gs.players[self.port], gs.players[self.opponent_port]
        dx = (op.position.x - me.position.x) / 100.0
        dy = (op.position.y - me.position.y) / 100.0
        # The game is mirrored so the opponent always appears on the canonical side.
        mirror = (self.canonical_side == "left" and dx > 0) or (self.canonical_side == "right" and dx < 0)
        if mirror:
            dx = -dx
        glob = np.array([dx, dy, np.hypot(dx, dy), min(gs.frame, 28800) / 28800.0], dtype=np.float32)
        opp = np.zeros(len(self.opponent_pool), dtype=np.float32)
        if op.character in self.opponent_pool:  # anyone else: all zeros, as in its training
            opp[self.opponent_pool.index(op.character)] = 1.0
        obs = np.concatenate([self._player_features(me, mirror), self._player_features(op, mirror), glob, opp])
        return np.clip(obs, -10.0, 10.0).astype(np.float32), mirror

    def act(self, gs, controller):
        """Set `controller` for the next frame from the current game state."""
        if self.port not in gs.players or self.opponent_port not in gs.players:
            return
        step = self._frame % self.frame_skip
        self._frame += 1
        if step == 0:
            obs, mirror = self.observation(gs)
            action = int(np.argmax(self.logits(obs)))
            if mirror:  # it saw a mirrored game, so flip its stick back
                action = MIRRORED_ACTION[action]
            stick, self._button = ACTIONS[action]
            controller.release_all()
            controller.tilt_analog(B.BUTTON_MAIN, *stick)
            if self._button is not None:
                controller.press_button(self._button)
        if step == self.frame_skip - 1 and self._button in TAP_BUTTONS and self.frame_skip > 1:
            controller.release_button(self._button)
