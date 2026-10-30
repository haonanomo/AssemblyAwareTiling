"""Load and query precomputed IK constraints for sequence environments.

The environment packs these tables into JAX arrays for batched legality checks
during policy training and inference. IK solutions are generated offline by
tiling_rl.sequence.ik_batched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

StateVec = np.ndarray


def resolve_ik_table_path(board_path, suffix="_ik_table.npz"):
    """Find a table in ik_tables/, falling back to legacy adjacent files."""
    board_path = Path(board_path)
    name = f"{board_path.stem}{suffix}"
    table_path = board_path.parent / "ik_tables" / name
    legacy_path = board_path.with_name(name)
    if table_path.is_file() or not legacy_path.is_file():
        return table_path
    return legacy_path


class IKLookupTable:
    """Query precomputed grasp samples and one/two-robot collision constraints.

    This class reads IK tables; it does not solve inverse kinematics or plan
    motion between samples. State entries are 0=removed, 1=free, and 2=held.
    """
    def __init__(self, npz_path: str | Path):
        self.path = str(Path(npz_path).resolve())
        with np.load(self.path, allow_pickle=False) as data:
            self.board = data["board"].astype(np.int32) if "board" in data else None
            self.grid_size = float(data["grid_size"][0]) if "grid_size" in data else 0.05
            self.scene_offset = data["scene_offset"].astype(np.float32) if "scene_offset" in data else np.zeros((3,), dtype=np.float32)
            self.piece_ids = data["piece_ids"].astype(np.int32)
            self.piece_to_col = {int(pid): int(i) for i, pid in enumerate(self.piece_ids)}
            self.q = data["q"].astype(np.float32)
            self.sample_piece_id = data["sample_piece_id"].astype(np.int32)
            self.sample_robot_id = data["sample_robot_id"].astype(np.int32)
            self.collision_mask = data["collision_mask"].astype(bool)
            self.robot1_sample_ids = data.get("robot1_sample_ids", np.where(self.sample_robot_id == 1)[0]).astype(np.int32)
            self.robot2_sample_ids = data.get("robot2_sample_ids", np.where(self.sample_robot_id == 2)[0]).astype(np.int32)
            self.pair_collision = data.get(
                "pair_collision",
                np.zeros((len(self.robot1_sample_ids), len(self.robot2_sample_ids)), dtype=np.uint8),
            ).astype(bool)
        self.global_to_local = {
            1: {int(g): int(i) for i, g in enumerate(self.robot1_sample_ids)},
            2: {int(g): int(i) for i, g in enumerate(self.robot2_sample_ids)},
        }
        self.by_piece_robot: Dict[Tuple[int, int], np.ndarray] = {}
        for pid in self.piece_ids:
            for rid in (1, 2):
                ids = np.where((self.sample_piece_id == int(pid)) & (self.sample_robot_id == rid))[0].astype(np.int32)
                self.by_piece_robot[(int(pid), rid)] = ids
        self._valid_cache: Dict[Tuple[int, int, Tuple[int, ...]], np.ndarray] = {}

    def get_valid_samples(self, pid: int, robot_id: int, active_pids: Sequence[int]) -> np.ndarray:
        active_tuple = tuple(sorted(int(p) for p in active_pids if int(p) != int(pid)))
        key = (int(pid), int(robot_id), active_tuple)
        if key in self._valid_cache:
            return self._valid_cache[key]

        ids = self.by_piece_robot.get((int(pid), int(robot_id)), np.zeros((0,), dtype=np.int32))
        if ids.shape[0] == 0:
            self._valid_cache[key] = ids
            return ids

        cols = [self.piece_to_col[p] for p in active_tuple if p in self.piece_to_col]
        if len(cols) == 0:
            out = ids
        else:
            ok = ~np.any(self.collision_mask[ids][:, cols], axis=1)
            out = ids[ok]

        self._valid_cache[key] = out
        return out

    def one_hold_feasible(self, pid: int, active_pids: Sequence[int]) -> bool:
        return (
            self.get_valid_samples(pid, 1, active_pids).shape[0] > 0
            or self.get_valid_samples(pid, 2, active_pids).shape[0] > 0
        )

    def two_hold_feasible(self, pid_a: int, pid_b: int, active_pids: Sequence[int]) -> bool:
        return self.find_pair_assignment(pid_a, pid_b, active_pids) != (-1, -1)

    def find_pair_assignment(
        self,
        pid_a: int,
        pid_b: int,
        active_pids: Sequence[int],
    ) -> Tuple[int, int]:
        out = self._pair_assignment(pid_a, 1, pid_b, 2, active_pids)
        if out != (-1, -1):
            return out
        return self._pair_assignment(pid_a, 2, pid_b, 1, active_pids)

    def _pair_assignment(
        self,
        pid_a: int,
        robot_a: int,
        pid_b: int,
        robot_b: int,
        active_pids: Sequence[int],
    ) -> Tuple[int, int]:
        if robot_a == robot_b:
            return -1, -1

        sa = self.get_valid_samples(pid_a, robot_a, active_pids)
        sb = self.get_valid_samples(pid_b, robot_b, active_pids)
        if sa.shape[0] == 0 or sb.shape[0] == 0:
            return -1, -1

        if robot_a == 1:
            s1_global, s2_global = sa, sb
        else:
            s1_global, s2_global = sb, sa

        r1_local = np.array([self.global_to_local[1].get(int(x), -1) for x in s1_global], dtype=np.int32)
        r2_local = np.array([self.global_to_local[2].get(int(x), -1) for x in s2_global], dtype=np.int32)

        keep1 = r1_local >= 0
        keep2 = r2_local >= 0
        s1_global = s1_global[keep1]
        s2_global = s2_global[keep2]
        r1_local = r1_local[keep1]
        r2_local = r2_local[keep2]

        if r1_local.shape[0] == 0 or r2_local.shape[0] == 0:
            return -1, -1

        if self.pair_collision.shape[0] == 0 or self.pair_collision.shape[1] == 0:
            return int(s1_global[0]), int(s2_global[0])

        ok_pair = ~self.pair_collision[np.ix_(r1_local, r2_local)]
        pair_ids = np.argwhere(ok_pair)
        if pair_ids.shape[0] == 0:
            return -1, -1

        i, j = pair_ids[0]
        return int(s1_global[int(i)]), int(s2_global[int(j)])

    def state_feasible(self, state: StateVec) -> bool:
        held = (np.where(state == 2)[0] + 1).astype(np.int32)
        if held.shape[0] == 0:
            return True
        if held.shape[0] > 2:
            return False

        active_pids = (np.where(state > 0)[0] + 1).astype(np.int32).tolist()
        if held.shape[0] == 1:
            return self.one_hold_feasible(int(held[0]), active_pids)
        return self.two_hold_feasible(int(held[0]), int(held[1]), active_pids)

    def choose_assignment(self, state: StateVec) -> Tuple[int, int]:
        held = (np.where(state == 2)[0] + 1).astype(np.int32)
        active_pids = (np.where(state > 0)[0] + 1).astype(np.int32).tolist()
        if held.shape[0] == 0:
            return -1, -1

        if held.shape[0] == 1:
            pid = int(held[0])
            for rid in (1, 2):
                samples = self.get_valid_samples(pid, rid, active_pids)
                if samples.shape[0] > 0:
                    return (int(samples[0]), -1) if rid == 1 else (-1, int(samples[0]))
            return -1, -1

        pid_a, pid_b = int(held[0]), int(held[1])
        return self.find_pair_assignment(pid_a, pid_b, active_pids)
