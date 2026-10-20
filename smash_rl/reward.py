"""Phase 2 reward for port 1, from consecutive raw frames [agent, opponent].

Mostly zero-sum, as in slippi-ai: +kill per stock taken and -kill per stock lost, and
damage x (percent dealt - percent taken). Small shaping terms stop known self-play
exploits without changing what wins: a penalty per ledge grab (planking), a penalty per
frame offstage beyond a few seconds (stalling), and a bonus for moving towards the
opponent (running away to win on time).
"""

import melee
import numpy as np

EDGE_CATCHING = melee.Action.EDGE_CATCHING.value
MAX_STEP = 20.0  # longer moves in one frame are respawns, not movement


class Reward:
    def __init__(self, kill: float = 1.0, damage: float = 0.01, ledge_grab: float = 0.02,
                 approach: float = 0.0002, stall: float = 0.001, stall_frames: int = 180):
        self.kill, self.damage, self.ledge_grab = kill, damage, ledge_grab
        self.approach, self.stall, self.stall_frames = approach, stall, stall_frames
        self.reset(melee.Stage.FINAL_DESTINATION)

    def reset(self, stage: melee.Stage):
        """A new game on `stage`."""
        self.edge = melee.stages.EDGE_GROUND_POSITION.get(stage, 85.0)
        self.offstage_frames = 0

    def __call__(self, prev: np.ndarray, cur: np.ndarray) -> float:
        lost = max(0, int(prev[0]["stocks"]) - int(cur[0]["stocks"]))
        taken = max(0, int(prev[1]["stocks"]) - int(cur[1]["stocks"]))
        # Percent drops to 0 on a death, so damage only counts within a stock.
        damage_taken = 0.0 if lost else max(0.0, float(cur[0]["percent"] - prev[0]["percent"]))
        damage_dealt = 0.0 if taken else max(0.0, float(cur[1]["percent"] - prev[1]["percent"]))
        r = self.kill * (taken - lost) + self.damage * (damage_dealt - damage_taken)
        if cur[0]["action"] == EDGE_CATCHING and prev[0]["action"] != EDGE_CATCHING:
            r -= self.ledge_grab
        offstage = abs(float(cur[0]["x"])) > self.edge or float(cur[0]["y"]) < -5.0
        self.offstage_frames = self.offstage_frames + 1 if offstage and not lost else 0
        if self.offstage_frames > self.stall_frames:
            r -= self.stall
        to_opponent = np.array([prev[1]["x"] - prev[0]["x"], prev[1]["y"] - prev[0]["y"]], dtype=np.float64)
        moved = np.array([cur[0]["x"] - prev[0]["x"], cur[0]["y"] - prev[0]["y"]], dtype=np.float64)
        distance = np.linalg.norm(to_opponent)
        if not lost and distance > 1e-3 and np.linalg.norm(moved) < MAX_STEP:
            r += self.approach * float(moved @ to_opponent) / distance
        return float(r)
