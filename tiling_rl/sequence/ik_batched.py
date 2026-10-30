"""Precompute sequence IK tables and inspect robot configurations.

Run explicitly with ``python -m tiling_rl.sequence.ik_batched precompute_dir``.
Sequence training loads the resulting tables without invoking this module.
Robot YAML, URDF, mesh, and solver resources live in the project robot directory.
"""

from __future__ import annotations

import argparse
import sys
import time
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
from skimage.measure import label

from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Cuboid, Scene
from curobo.types import ContentPath, GoalToolPose, Pose
from curobo.viewer import ViserVisualizer
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg

PROJECT_DIR = Path(__file__).resolve().parents[2]
ROBOT_DIR = PROJECT_DIR / "robot"


def load_robot_config(robot_yml):
    """Resolve robot resources relative to the supplied YAML file."""
    path = Path(robot_yml).resolve()
    config = yaml.safe_load(path.read_text())
    kin = config["robot_cfg"]["kinematics"]
    for key in ("urdf_path", "asset_root_path"):
        kin[key] = str((path.parent / kin[key]).resolve())
    return config


def robot_content_path(robot_yml):
    kin = load_robot_config(robot_yml)["robot_cfg"]["kinematics"]
    return ContentPath(
        robot_config_file=str(Path(robot_yml).resolve()),
        robot_urdf_absolute_path=kin["urdf_path"],
        robot_asset_absolute_path=kin["asset_root_path"],
    )


QUATS_24 = [
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),

    (np.sqrt(2) / 2,  np.sqrt(2) / 2, 0, 0),
    (np.sqrt(2) / 2, -np.sqrt(2) / 2, 0, 0),
    (np.sqrt(2) / 2, 0,  np.sqrt(2) / 2, 0),
    (np.sqrt(2) / 2, 0, -np.sqrt(2) / 2, 0),
    (np.sqrt(2) / 2, 0, 0,  np.sqrt(2) / 2),
    (np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2),

    (0.5,  0.5,  0.5,  0.5),
    (0.5, -0.5, -0.5, -0.5),
    (0.5,  0.5, -0.5, -0.5),
    (0.5, -0.5,  0.5,  0.5),
    (0.5, -0.5,  0.5, -0.5),
    (0.5,  0.5, -0.5,  0.5),

    (0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0),
    (0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 0),
    (0, np.sqrt(2) / 2, 0, np.sqrt(2) / 2),
    (0, np.sqrt(2) / 2, 0, -np.sqrt(2) / 2),
    (0, 0, np.sqrt(2) / 2, np.sqrt(2) / 2),
    (0, 0, np.sqrt(2) / 2, -np.sqrt(2) / 2),

    (0, -np.sqrt(2) / 2, np.sqrt(2) / 2, 0),
    (0, 0, -np.sqrt(2) / 2, np.sqrt(2) / 2),
]

def normalize_piece_ids(board: np.ndarray):
    out = board.copy()
    pids = piece_ids_from_board(board)
    mapping = {old: new for new, old in enumerate(pids, start=1)}
    for old, new in mapping.items():
        out[board == old] = new
    return out, mapping


def load_board(path: str | Path) -> np.ndarray:
    board = np.load(path).astype(np.int32)
    if board.ndim != 2:
        raise ValueError(f"Expected 2D board npy, got shape={board.shape}")

    num_zero = int(np.sum(board == 0))
    if num_zero > 0:
        print(
            f"[VIRTUAL-0] board contains {num_zero} zero cells; "
            f"keep as virtual ignored region for IK",
            flush=True,
        )

    board, _ = normalize_piece_ids(board)
    board = center_board_content(board, background_value=-1)
    return board

def board_to_voxels(board: np.ndarray) -> torch.Tensor:
    return torch.tensor(board[None, :, :], dtype=torch.int32)


def piece_ids_from_board(board: np.ndarray) -> List[int]:
    return sorted(int(x) for x in np.unique(board) if x > 0)

def center_board_content(board: np.ndarray, background_value: int = -1) -> np.ndarray:
    board = np.asarray(board, dtype=np.int32)
    coords = np.argwhere(board != background_value)

    if coords.shape[0] == 0:
        return board.copy()

    h, w = board.shape
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    content_h = r1 - r0 + 1
    content_w = c1 - c0 + 1

    target_r0 = h - content_h
    target_c0 = (w - content_w) // 2

    dr = target_r0 - r0
    dc = target_c0 - c0

    out = np.full_like(board, background_value)

    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)

    dst_r0 = src_r0 + dr
    dst_r1 = src_r1 + dr
    dst_c0 = src_c0 + dc
    dst_c1 = src_c1 + dc

    out[dst_r0:dst_r1, dst_c0:dst_c1] = board[src_r0:src_r1, src_c0:src_c1]
    return out

def piece_color(pid: int) -> List[int]:
    rng = np.random.default_rng(int(pid) * 9973)
    return [int(x) for x in rng.integers(40, 240, size=3)]


def voxel_to_world(
    i: int,
    j: int,
    k: int,
    shape: Tuple[int, int, int],
    grid_size: float,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    x = (shape[2] - k - 1) * grid_size - (shape[2] - 1) / 2 * grid_size + scene_offset[0]
    z = (shape[1] - j - 1) * grid_size + grid_size / 2 + scene_offset[2]
    y = i * grid_size - grid_size / 2 * (shape[0] - 1) + scene_offset[1]
    return float(x), float(y), float(z)


def resolve_path_relative_to_yaml(path_value: str, yaml_path: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path(yaml_path).resolve().parent / path).resolve()


def infer_scene_offset_from_dual_yml(dual_yml: str | Path) -> Tuple[float, float, float]:
    dual_yml = Path(dual_yml).resolve()
    data = yaml.safe_load(dual_yml.read_text())
    kin = data["robot_cfg"]["kinematics"]
    urdf_path = resolve_path_relative_to_yaml(kin["urdf_path"], dual_yml)

    root = ET.parse(urdf_path).getroot()
    base_xyz = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        if name in {"arm0_fr3_base_joint", "arm1_fr3_base_joint"}:
            origin = joint.find("origin")
            if origin is not None:
                xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
                base_xyz.append(xyz)

    if len(base_xyz) != 2:
        print("[WARN] Cannot infer dual-arm midpoint from URDF. Use scene offset (0,0,0).")
        return (0.0, 0.0, 0.0)

    base_xyz = np.asarray(base_xyz, dtype=np.float32)
    mid = base_xyz[:, :2].mean(axis=0)
    return (float(mid[0]), float(mid[1]), 0.0)


def get_scene_offset(args) -> Tuple[float, float, float]:
    if getattr(args, "scene_offset_x", None) is not None:
        return (float(args.scene_offset_x), float(args.scene_offset_y), float(args.scene_offset_z))
    return infer_scene_offset_from_dual_yml(args.dual_yml)


def create_scene_from_voxels(
    voxels: torch.Tensor,
    grid_size: float = 0.05,
    include_table: bool = True,
    only_piece_id: int | None = None,
    exclude_piece_id: int | None = None,
    exclude_voxel_ijk: Tuple[int, int, int] | None = None,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    highlight_piece_id: int | None = None,
    highlight_target_voxel_ijk: Tuple[int, int, int] | None = None,
) -> Scene:
    cuboids = []
    shape = tuple(int(x) for x in voxels.shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                v = int(voxels[i, j, k].item())
                if v <= 0:
                    continue
                if only_piece_id is not None and v != only_piece_id:
                    continue
                if exclude_piece_id is not None and v == exclude_piece_id:
                    continue
                if exclude_voxel_ijk is not None and (int(i), int(j), int(k)) == tuple(exclude_voxel_ijk):
                    continue

                if highlight_piece_id is not None and v == int(highlight_piece_id):
                    if (
                        highlight_target_voxel_ijk is not None
                        and (int(i), int(j), int(k)) == tuple(highlight_target_voxel_ijk)
                    ):
                        color = [255, 0, 0]       # target voxel: bright red
                    else:
                        color = [255, 80, 80]   # same-piece forbidden voxels: light red/pink
                else:
                    color = piece_color(v)

                x, y, z = voxel_to_world(i, j, k, shape, grid_size, scene_offset)
                cuboids.append(
                    Cuboid(
                        name=f"v_{i}_{j}_{k}_p{v}",
                        dims=[grid_size, grid_size, grid_size],
                        pose=[x, y, z, 1, 0, 0, 0.0],
                        color=color,
                    )
                )

    if include_table:
        table_thickness = 0.05
        cuboids = [
            Cuboid(
                name="table",
                dims=[2.4, 2.4, table_thickness],
                pose=[
                    scene_offset[0],
                    scene_offset[1],
                    scene_offset[2] - table_thickness / 2.0,
                    1,
                    0,
                    0,
                    0.0,
                ],
            ),
            *cuboids,
        ]

    return Scene(cuboid=cuboids)


def create_table_scene(
    scene_offset: Tuple[float, float, float],
    table_size: float = 2.4,
    table_thickness: float = 0.05,
) -> Scene:
    return Scene(
        cuboid=[
            Cuboid(
                name="table",
                dims=[table_size, table_size, table_thickness],
                pose=[
                    scene_offset[0],
                    scene_offset[1],
                    scene_offset[2] - table_thickness / 2.0,
                    1,
                    0,
                    0,
                    0.0,
                ],
            )
        ]
    )


def check_scene_collision_from_q(
    checker: RobotCollisionChecker,
    q_batch: torch.Tensor,
    threshold: float = 0.0,
) -> np.ndarray:
    if q_batch.ndim == 2:
        q_batch = q_batch.unsqueeze(1)

    kin_state = checker.get_kinematics(q_batch)
    robot_spheres = kin_state.robot_spheres
    batch_size = int(robot_spheres.shape[0])
    horizon = int(robot_spheres.shape[1])
    num_spheres = int(robot_spheres.shape[2])

    # Direct distance queries require explicit collision-buffer initialization.
    cost = checker.collision_cost

    # Set batch/horizon first: update_num_spheres() reuses these dimensions.
    if hasattr(cost, "setup_batch_tensors"):
        cost.setup_batch_tensors(batch_size, horizon)
    if hasattr(cost, "update_num_spheres"):
        cost.update_num_spheres(num_spheres)

    d = checker.get_collision_distance(kin_state)

    if not torch.is_tensor(d):
        d = torch.as_tensor(d, device=q_batch.device)

    if d.ndim == 0:
        collision = d > threshold
    else:
        reduce_dims = tuple(range(1, d.ndim))
        collision = torch.any(d > threshold, dim=reduce_dims)

    return collision.detach().cpu().numpy().reshape(-1).astype(bool)


def make_goal_poses_for_piece(
    voxels: torch.Tensor,
    piece_id: int,
    grid_size: float,
    z_offset: float,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Pose, np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(int(x) for x in voxels.shape)

    positions = []
    quaternions = []
    voxel_ijk = []
    quat_ids = []

    coords = torch.nonzero(voxels == int(piece_id), as_tuple=False).cpu().numpy()
    for i, j, k in coords:
        x, y, z = voxel_to_world(int(i), int(j), int(k), shape, grid_size, scene_offset)
        z += float(z_offset)
        for qid, quat in enumerate(QUATS_24):
            positions.append([x, y, z])
            quaternions.append(list(quat))
            voxel_ijk.append([int(i), int(j), int(k)])
            quat_ids.append(qid)

    if len(positions) == 0:
        empty_pose = Pose(
            position=torch.empty((0, 3), device="cuda", dtype=torch.float32),
            quaternion=torch.empty((0, 4), device="cuda", dtype=torch.float32),
        )
        return empty_pose, np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    goal_pose = Pose(
        position=torch.tensor(positions, device="cuda", dtype=torch.float32),
        quaternion=torch.tensor(quaternions, device="cuda", dtype=torch.float32),
    )
    return (
        goal_pose,
        np.asarray(voxel_ijk, dtype=np.int32),
        np.asarray(quat_ids, dtype=np.int32),
        np.asarray(positions, dtype=np.float32),
    )


def solve_ik_samples(
    voxels: torch.Tensor,
    piece_id: int,
    robot_id: int,
    robot_yml: str,
    grid_size: float,
    z_offset: float,
    max_batch_size: int,
    num_seeds: int,
    use_scene_collision: bool,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Dict[str, np.ndarray]:
    goal_poses, voxel_ijk, quat_ids, target_positions = make_goal_poses_for_piece(
        voxels=voxels,
        piece_id=piece_id,
        grid_size=grid_size,
        z_offset=z_offset,
        scene_offset=scene_offset,
    )

    n_goals = int(goal_poses.shape[0])
    if n_goals == 0:
        return {
            "q": np.empty((0, 7), dtype=np.float32),
            "voxel_ijk": np.empty((0, 3), dtype=np.int32),
            "quat_id": np.empty((0,), dtype=np.int32),
            "target_pos": np.empty((0, 3), dtype=np.float32),
            "target_quat": np.empty((0, 4), dtype=np.float32),
        }

    # Constrain IK against the table; cache piece collisions for later removal states.
    scene_model = create_table_scene(scene_offset) if use_scene_collision else {}

    config = InverseKinematicsCfg.create(
        robot=load_robot_config(robot_yml),
        optimizer_configs=[str(ROBOT_DIR / "lbfgs_ik.yml")],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=scene_model,
        use_cuda_graph=True,
        max_batch_size=max_batch_size,
        num_seeds=num_seeds,
    )
    config.scene_collision_cfg.use_warp_collision = True

    ik_solver = InverseKinematics(config)
    target_link = ik_solver.tool_frames[0]

    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False

    result = ik_solver.solve_pose(
        GoalToolPose.from_poses({target_link: goal_poses}, num_goalset=1)
    )

    success = result.success.detach().cpu().numpy().astype(bool).reshape(-1)
    if not np.any(success):
        return {
            "q": np.empty((0, 7), dtype=np.float32),
            "voxel_ijk": np.empty((0, 3), dtype=np.int32),
            "quat_id": np.empty((0,), dtype=np.int32),
            "target_pos": np.empty((0, 3), dtype=np.float32),
            "target_quat": np.empty((0, 4), dtype=np.float32),
        }

    q = result.js_solution[result.success].position[:, :7].detach().cpu().numpy().astype(np.float32)
    target_quat = np.asarray(QUATS_24, dtype=np.float32)[quat_ids[success]]

    return {
        "q": q,
        "voxel_ijk": voxel_ijk[success],
        "quat_id": quat_ids[success],
        "target_pos": target_positions[success],
        "target_quat": target_quat,
    }



def make_goal_poses_for_all_pieces(
    voxels: torch.Tensor,
    piece_ids: List[int],
    grid_size: float,
    z_offset: float,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Tuple[Pose, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(int(x) for x in voxels.shape)

    positions = []
    quaternions = []
    piece_ids_out = []
    voxel_ijk = []
    quat_ids = []

    for piece_id in piece_ids:
        coords = torch.nonzero(voxels == int(piece_id), as_tuple=False).cpu().numpy()
        for i, j, k in coords:
            x, y, z = voxel_to_world(int(i), int(j), int(k), shape, grid_size, scene_offset)
            z += float(z_offset)
            for qid, quat in enumerate(QUATS_24):
                positions.append([x, y, z])
                quaternions.append(list(quat))
                piece_ids_out.append(int(piece_id))
                voxel_ijk.append([int(i), int(j), int(k)])
                quat_ids.append(qid)

    if len(positions) == 0:
        empty_pose = Pose(
            position=torch.empty((0, 3), device="cuda", dtype=torch.float32),
            quaternion=torch.empty((0, 4), device="cuda", dtype=torch.float32),
        )
        return (
            empty_pose,
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )

    goal_pose = Pose(
        position=torch.tensor(positions, device="cuda", dtype=torch.float32),
        quaternion=torch.tensor(quaternions, device="cuda", dtype=torch.float32),
    )
    return (
        goal_pose,
        np.asarray(piece_ids_out, dtype=np.int32),
        np.asarray(voxel_ijk, dtype=np.int32),
        np.asarray(quat_ids, dtype=np.int32),
        np.asarray(positions, dtype=np.float32),
        np.asarray(quaternions, dtype=np.float32),
    )


def solve_ik_samples_batched_for_robot(
    voxels: torch.Tensor,
    piece_ids: List[int],
    robot_id: int,
    robot_yml: str,
    grid_size: float,
    z_offset: float,
    max_batch_size: int,
    num_seeds: int,
    use_scene_collision: bool,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Dict[str, np.ndarray]:
    (
        goal_poses,
        goal_piece_ids,
        voxel_ijk,
        quat_ids,
        target_positions,
        target_quats,
    ) = make_goal_poses_for_all_pieces(
        voxels=voxels,
        piece_ids=piece_ids,
        grid_size=grid_size,
        z_offset=z_offset,
        scene_offset=scene_offset,
    )

    n_goals = int(goal_poses.shape[0])
    if n_goals == 0:
        return {
            "q": np.empty((0, 7), dtype=np.float32),
            "piece_id": np.empty((0,), dtype=np.int32),
            "voxel_ijk": np.empty((0, 3), dtype=np.int32),
            "quat_id": np.empty((0,), dtype=np.int32),
            "target_pos": np.empty((0, 3), dtype=np.float32),
            "target_quat": np.empty((0, 4), dtype=np.float32),
        }

    scene_model = create_table_scene(scene_offset) if use_scene_collision else {}

    config = InverseKinematicsCfg.create(
        robot=load_robot_config(robot_yml),
        optimizer_configs=[str(ROBOT_DIR / "lbfgs_ik.yml")],
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=scene_model,
        use_cuda_graph=True,
        max_batch_size=max_batch_size,
        num_seeds=num_seeds,
    )
    config.scene_collision_cfg.use_warp_collision = True

    ik_solver = InverseKinematics(config)
    target_link = ik_solver.tool_frames[0]

    ik_solver.config.use_lm_seed = False
    ik_solver.config.exit_early = False

    all_q = []
    all_piece_id = []
    all_voxel_ijk = []
    all_quat_id = []
    all_target_pos = []
    all_target_quat = []

    chunk_size = max(1, int(max_batch_size))
    for start in range(0, n_goals, chunk_size):
        end = min(start + chunk_size, n_goals)
        chunk_pose = Pose(
            position=goal_poses.position[start:end],
            quaternion=goal_poses.quaternion[start:end],
        )

        result = ik_solver.solve_pose(
            GoalToolPose.from_poses({target_link: chunk_pose}, num_goalset=1)
        )

        success = result.success.detach().cpu().numpy().astype(bool).reshape(-1)
        if not np.any(success):
            continue

        q = result.js_solution[result.success].position[:, :7].detach().cpu().numpy().astype(np.float32)

        all_q.append(q)
        all_piece_id.append(goal_piece_ids[start:end][success])
        all_voxel_ijk.append(voxel_ijk[start:end][success])
        all_quat_id.append(quat_ids[start:end][success])
        all_target_pos.append(target_positions[start:end][success])
        all_target_quat.append(target_quats[start:end][success])

    if len(all_q) == 0:
        return {
            "q": np.empty((0, 7), dtype=np.float32),
            "piece_id": np.empty((0,), dtype=np.int32),
            "voxel_ijk": np.empty((0, 3), dtype=np.int32),
            "quat_id": np.empty((0,), dtype=np.int32),
            "target_pos": np.empty((0, 3), dtype=np.float32),
            "target_quat": np.empty((0, 4), dtype=np.float32),
        }

    return {
        "q": np.concatenate(all_q, axis=0).astype(np.float32),
        "piece_id": np.concatenate(all_piece_id, axis=0).astype(np.int32),
        "voxel_ijk": np.concatenate(all_voxel_ijk, axis=0).astype(np.int32),
        "quat_id": np.concatenate(all_quat_id, axis=0).astype(np.int32),
        "target_pos": np.concatenate(all_target_pos, axis=0).astype(np.float32),
        "target_quat": np.concatenate(all_target_quat, axis=0).astype(np.float32),
    }

def compute_sample_piece_collision(
    q: np.ndarray,
    robot_yml: str,
    voxels: torch.Tensor,
    piece_ids: List[int],
    sample_piece_ids: np.ndarray,
    grid_size: float,
    batch_size: int = 4096,
    scene_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision_distance_threshold: float = 0.0,
) -> np.ndarray:
    n_samples = int(q.shape[0])
    n_pieces = len(piece_ids)
    collision = np.zeros((n_samples, n_pieces), dtype=np.bool_)

    if n_samples == 0:
        return collision

    q_torch = torch.tensor(q, device="cuda", dtype=torch.float32)

    for p_col, obstacle_pid in enumerate(piece_ids):
        scene = create_scene_from_voxels(
            voxels,
            grid_size=grid_size,
            include_table=False,
            only_piece_id=int(obstacle_pid),
            scene_offset=scene_offset,
        )
        config = RobotCollisionCheckerCfg.load_from_config(
            robot_config=load_robot_config(robot_yml),
            scene_model=scene,
            self_collision_activation_distance=0.01,
            collision_activation_distance=0.01,
        )
        checker = RobotCollisionChecker(config)

        collision_chunks = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            joint_positions = q_torch[start:end].unsqueeze(1)
            collision_now = check_scene_collision_from_q(
                checker,
                joint_positions,
                threshold=collision_distance_threshold,
            )
            collision_chunks.append(collision_now)

        collision_all = np.concatenate(collision_chunks, axis=0)
        collision[:, p_col] = collision_all

        # Contact/collision with the target piece itself should not invalidate a grasp.
        collision[sample_piece_ids == obstacle_pid, p_col] = False

    return collision


def compute_sample_table_collision(
    q: np.ndarray,
    robot_yml: str,
    scene_offset: Tuple[float, float, float],
    batch_size: int = 4096,
    collision_distance_threshold: float = 0.0,
) -> np.ndarray:
    n_samples = int(q.shape[0])
    table_collision = np.zeros((n_samples,), dtype=np.bool_)
    if n_samples == 0:
        return table_collision

    table_thickness = 0.05
    scene = Scene(
        cuboid=[
            Cuboid(
                name="table",
                dims=[2.4, 2.4, table_thickness],
                pose=[
                    scene_offset[0],
                    scene_offset[1],
                    scene_offset[2] - table_thickness / 2.0,
                    1,
                    0,
                    0,
                    0.0,
                ],
            )
        ]
    )

    config = RobotCollisionCheckerCfg.load_from_config(
        robot_config=load_robot_config(robot_yml),
        scene_model=scene,
        self_collision_activation_distance=0.01,
        collision_activation_distance=0.01,
    )
    checker = RobotCollisionChecker(config)

    q_torch = torch.tensor(q, device="cuda", dtype=torch.float32)
    collision_chunks = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        joint_positions = q_torch[start:end].unsqueeze(1)
        collision_now = check_scene_collision_from_q(
            checker,
            joint_positions,
            threshold=collision_distance_threshold,
        )
        collision_chunks.append(collision_now)

    table_collision[:] = np.concatenate(collision_chunks, axis=0)
    return table_collision


def compute_sample_same_piece_collision(
    q: np.ndarray,
    robot_yml: str,
    voxels: torch.Tensor,
    sample_piece_ids: np.ndarray,
    sample_voxel_ijk: np.ndarray,
    grid_size: float,
    scene_offset: Tuple[float, float, float],
    batch_size: int = 4096,
    collision_distance_threshold: float = 0.0,
) -> np.ndarray:
    n_samples = int(q.shape[0])
    same_piece_collision = np.zeros((n_samples,), dtype=np.bool_)
    if n_samples == 0:
        return same_piece_collision

    q_torch = torch.tensor(q, device="cuda", dtype=torch.float32)

    groups = {}
    for idx, (pid, ijk) in enumerate(zip(sample_piece_ids, sample_voxel_ijk)):
        key = (int(pid), int(ijk[0]), int(ijk[1]), int(ijk[2]))
        groups.setdefault(key, []).append(idx)

    for (pid, i, j, k), inds_list in groups.items():
        inds_np = np.asarray(inds_list, dtype=np.int64)
        scene = create_scene_from_voxels(
            voxels,
            grid_size=grid_size,
            include_table=False,
            only_piece_id=int(pid),
            exclude_voxel_ijk=(int(i), int(j), int(k)),
            scene_offset=scene_offset,
        )

        if len(scene.cuboid) == 0:
            continue

        config = RobotCollisionCheckerCfg.load_from_config(
            robot_config=load_robot_config(robot_yml),
            scene_model=scene,
            self_collision_activation_distance=0.01,
            collision_activation_distance=0.01,
        )
        checker = RobotCollisionChecker(config)

        collision_chunks = []
        for start in range(0, inds_np.shape[0], batch_size):
            end = min(start + batch_size, inds_np.shape[0])
            ids = inds_np[start:end]
            joint_positions = q_torch[ids].unsqueeze(1)
            collision_now = check_scene_collision_from_q(
                checker,
                joint_positions,
                threshold=collision_distance_threshold,
            )
            collision_chunks.append(collision_now)

        same_piece_collision[inds_np] = np.concatenate(collision_chunks, axis=0)

    return same_piece_collision


def compute_pair_collision(
    q0: np.ndarray,
    q1: np.ndarray,
    dual_robot_yml: str,
    batch_size: int = 8192,
) -> np.ndarray:
    n0, n1 = int(q0.shape[0]), int(q1.shape[0])
    pair_collision = np.zeros((n0, n1), dtype=np.bool_)

    if n0 == 0 or n1 == 0:
        return pair_collision

    config = RobotCollisionCheckerCfg.load_from_config(
        robot_config=load_robot_config(dual_robot_yml),
        scene_model={},
        self_collision_activation_distance=0.01,
        collision_activation_distance=0.01,
    )
    checker = RobotCollisionChecker(config)
    checker.collision_constraint = None

    q0_t = torch.tensor(q0, device="cuda", dtype=torch.float32)
    q1_t = torch.tensor(q1, device="cuda", dtype=torch.float32)

    total_pairs = n0 * n1
    for flat_start in range(0, total_pairs, batch_size):
        flat_end = min(flat_start + batch_size, total_pairs)
        ids = torch.arange(flat_start, flat_end, device="cuda")
        i = torch.div(ids, n1, rounding_mode="floor")
        j = ids % n1
        joint_positions = torch.cat([q0_t[i], q1_t[j]], dim=1).unsqueeze(1)
        valid = checker.validate(joint_positions)
        pair_collision.reshape(-1)[flat_start:flat_end] = ~valid.detach().cpu().numpy().reshape(-1).astype(bool)

    return pair_collision


def precompute_ik_table(args):
    t0 = time.perf_counter()

    board = load_board(args.board_npy)
    piece_ids = piece_ids_from_board(board)
    voxels = board_to_voxels(board)

    robot_yml = {
        1: str(Path(args.robot1_yml).resolve()),
        2: str(Path(args.robot2_yml).resolve()),
    }
    dual_yml = str(Path(args.dual_yml).resolve())
    scene_offset = get_scene_offset(args)

    all_q = []
    all_piece_id = []
    all_robot_id = []
    all_voxel_ijk = []
    all_quat_id = []
    all_target_pos = []
    all_target_quat = []

    per_robot_q = {1: [], 2: []}
    per_robot_global_ids = {1: [], 2: []}

    print(f"[INFO] board={args.board_npy} shape={board.shape} pieces={piece_ids}")
    print(f"[INFO] robot1={robot_yml[1]}")
    print(f"[INFO] robot2={robot_yml[2]}")
    print(f"[INFO] dual={dual_yml}")
    print(f"[INFO] scene_offset={scene_offset}")

    for rid in [1, 2]:
        total_goals = 0
        for pid in piece_ids:
            n_vox = int(np.sum(board == pid))
            goals = n_vox * len(QUATS_24)
            total_goals += goals
            print(f"[IK-BATCH] piece={pid} voxels={n_vox} robot={rid} goals={goals}", flush=True)

        print(
            f"[IK-BATCH] robot={rid} pieces={len(piece_ids)} total_goals={total_goals} "
            f"max_batch_size={args.max_batch_size} num_seeds={args.num_seeds}",
            flush=True,
        )
        out = solve_ik_samples_batched_for_robot(
            voxels=voxels,
            piece_ids=piece_ids,
            robot_id=rid,
            robot_yml=robot_yml[rid],
            grid_size=args.grid_size,
            z_offset=args.z_offset,
            max_batch_size=args.max_batch_size,
            num_seeds=args.num_seeds,
            use_scene_collision=args.use_scene_collision,
            scene_offset=scene_offset,
        )
        q = out["q"]
        n = int(q.shape[0])
        print(f"[IK-BATCH] robot={rid} success={n}", flush=True)

        if n == 0:
            continue

        global_start = len(all_q)
        global_ids = np.arange(global_start, global_start + n, dtype=np.int32)

        all_q.extend(q)
        all_piece_id.extend(out["piece_id"])
        all_robot_id.extend([rid] * n)
        all_voxel_ijk.extend(out["voxel_ijk"])
        all_quat_id.extend(out["quat_id"])
        all_target_pos.extend(out["target_pos"])
        all_target_quat.extend(out["target_quat"])

        per_robot_q[rid].append(q)
        per_robot_global_ids[rid].append(global_ids)

    if len(all_q) == 0:
        raise RuntimeError("No IK sample was found for any piece/robot.")

    q_all = np.asarray(all_q, dtype=np.float32)
    sample_piece_id = np.asarray(all_piece_id, dtype=np.int32)
    sample_robot_id = np.asarray(all_robot_id, dtype=np.int32)
    sample_voxel_ijk = np.asarray(all_voxel_ijk, dtype=np.int32)
    sample_quat_id = np.asarray(all_quat_id, dtype=np.int32)
    sample_target_pos = np.asarray(all_target_pos, dtype=np.float32)
    sample_target_quat = np.asarray(all_target_quat, dtype=np.float32)

    print(f"[INFO] total_raw_samples={q_all.shape[0]}", flush=True)

    table_collision_raw = np.zeros((q_all.shape[0],), dtype=np.bool_)
    same_piece_collision_raw = np.zeros((q_all.shape[0],), dtype=np.bool_)

    for rid in [1, 2]:
        ids = np.where(sample_robot_id == rid)[0]
        if ids.shape[0] == 0:
            continue

        print(f"[TABLE] robot={rid} samples={ids.shape[0]}", flush=True)
        table_collision_raw[ids] = compute_sample_table_collision(
            q=q_all[ids],
            robot_yml=robot_yml[rid],
            scene_offset=scene_offset,
            batch_size=args.collision_batch_size,
            collision_distance_threshold=args.collision_distance_threshold,
        )

        print(f"[SELF-PIECE] robot={rid} samples={ids.shape[0]}", flush=True)
        same_piece_collision_raw[ids] = compute_sample_same_piece_collision(
            q=q_all[ids],
            robot_yml=robot_yml[rid],
            voxels=voxels,
            sample_piece_ids=sample_piece_id[ids],
            sample_voxel_ijk=sample_voxel_ijk[ids],
            grid_size=args.grid_size,
            scene_offset=scene_offset,
            batch_size=args.collision_batch_size,
            collision_distance_threshold=args.collision_distance_threshold,
        )

    keep = (~table_collision_raw) & (~same_piece_collision_raw)
    print(
        f"[FILTER] keep={int(np.sum(keep))}/{q_all.shape[0]} | "
        f"table_collision={int(np.sum(table_collision_raw))} | "
        f"same_piece_collision={int(np.sum(same_piece_collision_raw))}",
        flush=True,
    )

    q_all = q_all[keep]
    sample_piece_id = sample_piece_id[keep]
    sample_robot_id = sample_robot_id[keep]
    sample_voxel_ijk = sample_voxel_ijk[keep]
    sample_quat_id = sample_quat_id[keep]
    sample_target_pos = sample_target_pos[keep]
    sample_target_quat = sample_target_quat[keep]
    table_collision = table_collision_raw[keep]
    same_piece_collision = same_piece_collision_raw[keep]

    if q_all.shape[0] == 0:
        raise RuntimeError("All IK samples were filtered by table/same-piece collision.")

    print(f"[INFO] total_kept_samples={q_all.shape[0]}", flush=True)

    collision_mask = np.zeros((q_all.shape[0], len(piece_ids)), dtype=np.bool_)
    for rid in [1, 2]:
        ids = np.where(sample_robot_id == rid)[0]
        if ids.shape[0] == 0:
            continue
        print(f"[PIECE-COLLISION-MASK] robot={rid} samples={ids.shape[0]} vs pieces={len(piece_ids)}", flush=True)
        collision_mask[ids] = compute_sample_piece_collision(
            q=q_all[ids],
            robot_yml=robot_yml[rid],
            voxels=voxels,
            piece_ids=piece_ids,
            sample_piece_ids=sample_piece_id[ids],
            grid_size=args.grid_size,
            batch_size=args.collision_batch_size,
            scene_offset=scene_offset,
            collision_distance_threshold=args.collision_distance_threshold,
        )

    robot1_ids = np.where(sample_robot_id == 1)[0].astype(np.int32)
    robot2_ids = np.where(sample_robot_id == 2)[0].astype(np.int32)

    if args.no_pair_collision:
        pair_collision = np.zeros((robot1_ids.shape[0], robot2_ids.shape[0]), dtype=np.bool_)
    else:
        print(f"[PAIR] robot1_samples={robot1_ids.shape[0]} robot2_samples={robot2_ids.shape[0]}", flush=True)
        pair_collision = compute_pair_collision(
            q0=q_all[robot1_ids],
            q1=q_all[robot2_ids],
            dual_robot_yml=dual_yml,
            batch_size=args.pair_batch_size,
        )

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        board=board.astype(np.int32),
        piece_ids=np.asarray(piece_ids, dtype=np.int32),
        q=q_all,
        sample_piece_id=sample_piece_id,
        sample_robot_id=sample_robot_id,
        sample_voxel_ijk=sample_voxel_ijk,
        sample_quat_id=sample_quat_id,
        sample_target_pos=sample_target_pos,
        sample_target_quat=sample_target_quat,
        collision_mask=collision_mask.astype(np.uint8),
        table_collision=table_collision.astype(np.uint8),
        same_piece_collision=same_piece_collision.astype(np.uint8),
        robot1_sample_ids=robot1_ids,
        robot2_sample_ids=robot2_ids,
        pair_collision=pair_collision.astype(np.uint8),
        grid_size=np.asarray([args.grid_size], dtype=np.float32),
        z_offset=np.asarray([args.z_offset], dtype=np.float32),
        scene_offset=np.asarray(scene_offset, dtype=np.float32),
    )

    dt = time.perf_counter() - t0
    print(f"[DONE] saved {out_path}")
    print(f"[DONE] samples={q_all.shape[0]} time={dt:.2f}s")


def precompute_ik_dir(args):
    """Process every board, then raise if any file failed; always save the log."""
    board_dir = Path(args.board_dir)
    out_dir = Path(args.out_dir) if args.out_dir != "" else board_dir / "ik_tables"

    if not board_dir.exists() or not board_dir.is_dir():
        raise ValueError(f"Invalid board_dir: {board_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(board_dir.glob("*.npy"))
    if len(npy_files) == 0:
        raise ValueError(f"No .npy files found in {board_dir}")

    print(f"[BATCH] board_dir={board_dir.resolve()}", flush=True)
    print(f"[BATCH] out_dir={out_dir.resolve()}", flush=True)
    print(f"[BATCH] files={len(npy_files)}", flush=True)

    t_all = time.perf_counter()
    logs = []

    for idx, npy_path in enumerate(npy_files, start=1):
        t0 = time.perf_counter()
        out_npz = out_dir / f"{npy_path.stem}_ik_table.npz"

        print("=" * 100, flush=True)
        print(f"[BATCH {idx}/{len(npy_files)} START] {npy_path.name}", flush=True)
        print(f"[BATCH] table_out={out_npz}", flush=True)

        run_args = SimpleNamespace(**vars(args))
        run_args.board_npy = str(npy_path)
        run_args.out_npz = str(out_npz)

        try:
            precompute_ik_table(run_args)
            ok = True
            err = ""
        except Exception as e:
            ok = False
            err = str(e)
            print(f"[BATCH {idx}/{len(npy_files)} ERROR] {npy_path.name}: {e}", flush=True)

        dt = time.perf_counter() - t0
        logs.append(
            {
                "input_file": npy_path.name,
                "input_stem": npy_path.stem,
                "ik_table_npz": out_npz.name,
                "success": bool(ok),
                "error": err,
                "runtime_sec": float(dt),
            }
        )

        print(
            f"[BATCH {idx}/{len(npy_files)} DONE] {npy_path.stem} | "
            f"success={ok} | time={dt:.2f}s",
            flush=True,
        )

    total_dt = time.perf_counter() - t_all
    summary_path = out_dir / "ik_precompute_log.json"
    import json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "board_dir": str(board_dir.resolve()),
                "out_dir": str(out_dir.resolve()),
                "files_total": len(logs),
                "files_success": int(sum(1 for x in logs if x["success"])),
                "files_error": int(sum(1 for x in logs if not x["success"])),
                "runtime_sec_total": float(total_dt),
                "files": logs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 100, flush=True)
    print(f"[BATCH DONE] total_time={total_dt:.2f}s", flush=True)
    print(f"[BATCH DONE] saved log to {summary_path}", flush=True)
    failed = sum(not item["success"] for item in logs)
    if failed:
        raise RuntimeError(f"IK precomputation failed for {failed}/{len(logs)} boards; see {summary_path}")


def sample_valid_for_active_pieces(
    table: Dict[str, np.ndarray],
    sample_idx: int,
    active_piece_ids: List[int],
) -> bool:
    piece_ids = table["piece_ids"].astype(np.int32).tolist()
    p_to_col = {pid: i for i, pid in enumerate(piece_ids)}
    target_pid = int(table["sample_piece_id"][sample_idx])
    collision_mask = table["collision_mask"].astype(bool)

    for pid in active_piece_ids:
        if pid == target_pid:
            continue
        col = p_to_col.get(int(pid), None)
        if col is not None and collision_mask[sample_idx, col]:
            return False
    return True


def launch_viewer(args):
    data = dict(np.load(args.table_npz, allow_pickle=False))
    board = data["board"].astype(np.int32)
    grid_size = float(data["grid_size"][0]) if "grid_size" in data else args.grid_size
    scene_offset = tuple(float(x) for x in data["scene_offset"]) if "scene_offset" in data else get_scene_offset(args)

    dual_yml = str(Path(args.dual_yml).resolve())

    viser_viz = ViserVisualizer(
        content_path=robot_content_path(dual_yml),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=True,
        add_robot_to_scene=True,
    )

    config = KinematicsCfg.from_content_path(robot_content_path(dual_yml))
    robot = Kinematics(config)
    default_joint_state = robot.default_joint_state

    board_voxels = board_to_voxels(board)

    def refresh_obstacle_scene(
        highlight_piece_id: int | None = None,
        highlight_target_voxel_ijk: Tuple[int, int, int] | None = None,
    ):
        for name in ["obstacles", "/obstacles"]:
            try:
                viser_viz._server.scene.remove_by_name(name)
            except Exception:
                pass

        scene = create_scene_from_voxels(
            board_voxels,
            grid_size=grid_size,
            include_table=True,
            scene_offset=scene_offset,
            highlight_piece_id=highlight_piece_id,
            highlight_target_voxel_ijk=highlight_target_voxel_ijk,
        )
        viser_viz.add_scene(scene)

    refresh_obstacle_scene()

    q = data["q"].astype(np.float32)
    sample_piece_id = data["sample_piece_id"].astype(np.int32)
    sample_robot_id = data["sample_robot_id"].astype(np.int32)
    sample_voxel_ijk = data["sample_voxel_ijk"].astype(np.int32)
    sample_quat_id = data["sample_quat_id"].astype(np.int32)
    collision_mask = data["collision_mask"].astype(bool)
    table_collision = data["table_collision"].astype(bool) if "table_collision" in data else np.zeros((q.shape[0],), dtype=bool)
    same_piece_collision = data["same_piece_collision"].astype(bool) if "same_piece_collision" in data else np.zeros((q.shape[0],), dtype=bool)
    piece_ids = data["piece_ids"].astype(np.int32).tolist()

    piece_voxels = {}
    voxel_local_id = {}
    for pid in piece_ids:
        vox = sample_voxel_ijk[sample_piece_id == pid]
        if vox.shape[0] == 0:
            uniq = np.empty((0, 3), dtype=np.int32)
        else:
            uniq = np.unique(vox, axis=0)
            order = np.lexsort((uniq[:, 2], uniq[:, 1], uniq[:, 0]))
            uniq = uniq[order]
        piece_voxels[int(pid)] = uniq
        for local_idx, ijk in enumerate(uniq):
            voxel_local_id[(int(pid), int(ijk[0]), int(ijk[1]), int(ijk[2]))] = int(local_idx)

    print(f"[VIEW] samples={q.shape[0]} pieces={piece_ids}")
    print("[VIEW] open http://localhost:%d" % args.port)

    def show_sample(sample_idx: int):
        sample_idx = int(np.clip(sample_idx, 0, q.shape[0] - 1))
        joint_state = default_joint_state.clone()
        if int(sample_robot_id[sample_idx]) == 1:
            joint_state.position[:7] = torch.tensor(q[sample_idx], device=joint_state.position.device, dtype=joint_state.position.dtype)
        else:
            joint_state.position[7:14] = torch.tensor(q[sample_idx], device=joint_state.position.device, dtype=joint_state.position.dtype)

        viser_viz.set_joint_state(joint_state=joint_state)

        cols = np.where(collision_mask[sample_idx])[0]
        collide_pids = [piece_ids[c] for c in cols]
        pid = int(sample_piece_id[sample_idx])
        ijk = sample_voxel_ijk[sample_idx]

        refresh_obstacle_scene(
            highlight_piece_id=pid,
            highlight_target_voxel_ijk=(int(ijk[0]), int(ijk[1]), int(ijk[2])),
        )

        local_vid = voxel_local_id.get((pid, int(ijk[0]), int(ijk[1]), int(ijk[2])), -1)
        print(
            f"[SAMPLE] idx={sample_idx} robot={int(sample_robot_id[sample_idx])} "
            f"piece={pid} local_voxel={local_vid} voxel_ijk={ijk.tolist()} "
            f"quat={int(sample_quat_id[sample_idx])} collide_pieces={collide_pids} "
            f"table_collision={bool(table_collision[sample_idx])} "
            f"same_piece_collision={bool(same_piece_collision[sample_idx])}",
            flush=True,
        )

    def available_local_voxel_ids(pid: int) -> List[str]:
        voxels_for_piece = piece_voxels.get(int(pid), np.empty((0, 3), dtype=np.int32))
        return [str(i) for i in range(len(voxels_for_piece))]

    def available_quat_ids(rid: int, pid: int, local_vid: int) -> List[str]:
        voxels_for_piece = piece_voxels.get(int(pid), np.empty((0, 3), dtype=np.int32))
        if local_vid < 0 or local_vid >= len(voxels_for_piece):
            return []
        ijk = voxels_for_piece[local_vid]
        mask = (
            (sample_robot_id == int(rid))
            & (sample_piece_id == int(pid))
            & np.all(sample_voxel_ijk == ijk, axis=1)
        )
        qids = sorted(int(x) for x in np.unique(sample_quat_id[mask]))
        return [str(x) for x in qids]

    def set_dropdown_options(dropdown_obj, options: List[str]):
        if len(options) == 0:
            options = ["None"]
        try:
            dropdown_obj.options = options
        except Exception:
            pass
        try:
            dropdown_obj.value = options[0]
        except Exception:
            pass

    try:
        server = viser_viz._server

        sample_options = [str(i) for i in range(q.shape[0])]
        dropdown = server.gui.add_dropdown("global_sample_id", options=sample_options, initial_value="0")
        button = server.gui.add_button("show global sample")

        @button.on_click
        def _(_event):
            show_sample(int(dropdown.value))

        robot_dropdown = server.gui.add_dropdown("robot_id", options=["1", "2"], initial_value="1")
        piece_dropdown = server.gui.add_dropdown("piece_id", options=[str(p) for p in piece_ids], initial_value=str(piece_ids[0]))

        init_pid = int(piece_dropdown.value)
        init_voxel_options = available_local_voxel_ids(init_pid)
        if len(init_voxel_options) == 0:
            init_voxel_options = ["None"]

        local_voxel_dropdown = server.gui.add_dropdown(
            "piece_local_voxel_id",
            options=init_voxel_options,
            initial_value=init_voxel_options[0],
        )

        init_q_options = available_quat_ids(int(robot_dropdown.value), init_pid, int(local_voxel_dropdown.value) if local_voxel_dropdown.value != "None" else -1)
        if len(init_q_options) == 0:
            init_q_options = ["None"]
        quat_dropdown = server.gui.add_dropdown("quat_id_existing_for_this_voxel", options=init_q_options, initial_value=init_q_options[0])

        def refresh_voxel_and_quat():
            pid = int(piece_dropdown.value)
            voxel_options = available_local_voxel_ids(pid)
            set_dropdown_options(local_voxel_dropdown, voxel_options)
            local_vid = int(local_voxel_dropdown.value) if local_voxel_dropdown.value != "None" else -1
            q_options = available_quat_ids(int(robot_dropdown.value), pid, local_vid)
            set_dropdown_options(quat_dropdown, q_options)

        def refresh_quat_only():
            pid = int(piece_dropdown.value)
            local_vid = int(local_voxel_dropdown.value) if local_voxel_dropdown.value != "None" else -1
            q_options = available_quat_ids(int(robot_dropdown.value), pid, local_vid)
            set_dropdown_options(quat_dropdown, q_options)

        try:
            @robot_dropdown.on_update
            def _(_event):
                refresh_quat_only()

            @piece_dropdown.on_update
            def _(_event):
                refresh_voxel_and_quat()

            @local_voxel_dropdown.on_update
            def _(_event):
                refresh_quat_only()
        except Exception as e:
            print(f"[WARN] This viser version may not support dropdown update callbacks: {e}", flush=True)

        refresh_voxel_and_quat()

        select_button = server.gui.add_button("show robot/piece/local_voxel/existing_quat")

        @select_button.on_click
        def _(_event):
            rid = int(robot_dropdown.value)
            pid = int(piece_dropdown.value)
            if local_voxel_dropdown.value == "None" or quat_dropdown.value == "None":
                print(f"[VIEW] no kept sample for robot={rid}, piece={pid} under current selection", flush=True)
                return
            local_vid = int(local_voxel_dropdown.value)
            qid = int(quat_dropdown.value)

            voxels_for_piece = piece_voxels.get(pid, np.empty((0, 3), dtype=np.int32))
            if local_vid < 0 or local_vid >= len(voxels_for_piece):
                print(
                    f"[VIEW] piece={pid} only has local voxel ids 0..{len(voxels_for_piece)-1}; "
                    f"got local_voxel={local_vid}",
                    flush=True,
                )
                return

            ijk = voxels_for_piece[local_vid]
            mask = (
                (sample_robot_id == rid)
                & (sample_piece_id == pid)
                & (sample_quat_id == qid)
                & np.all(sample_voxel_ijk == ijk, axis=1)
            )
            inds = np.where(mask)[0]
            if inds.shape[0] == 0:
                valid_qids = available_quat_ids(rid, pid, local_vid)
                print(
                    f"[VIEW] no kept sample for robot={rid}, piece={pid}, "
                    f"local_voxel={local_vid}, voxel_ijk={ijk.tolist()}, quat={qid}; "
                    f"available_quats={valid_qids}",
                    flush=True,
                )
                return
            show_sample(int(inds[0]))

        print("[VIEW] GUI ready: choose either global_sample_id or robot/piece/local_voxel/existing_quat.")
        print("[VIEW] quat_id dropdown is filtered to kept samples for the selected robot/piece/local voxel.")
        print("[VIEW] Highlight: target voxel = bright red; other voxels in the same piece = light red/pink.")
    except Exception as e:
        print(f"[WARN] Could not create viser GUI widgets: {e}")
        print("[VIEW] Falling back to terminal input.")
        show_sample(0)
        while True:
            s = input("sample id> ").strip()
            if s.lower() in {"q", "quit", "exit"}:
                break
            if s:
                show_sample(int(s))
        return

    show_sample(0)
    while True:
        time.sleep(1.0)



def launch_sequence_viewer(args):
    seq_data = dict(np.load(args.sequence_npz, allow_pickle=False))
    ik_table_path = str(seq_data["ik_table_npz"][0])
    table = dict(np.load(ik_table_path, allow_pickle=False))

    board = seq_data["full_board"].astype(np.int32)
    states = seq_data["states"].astype(np.int32)
    robot_sample_ids = seq_data["robot_sample_ids"].astype(np.int32)
    action_type = seq_data["action_type"] if "action_type" in seq_data else np.asarray([], dtype="U8")
    action_pid = seq_data["action_pid"].astype(np.int32) if "action_pid" in seq_data else np.asarray([], dtype=np.int32)

    if "ik_q" in seq_data:
        q = seq_data["ik_q"].astype(np.float32)
        print("[SEQ] using embedded IK q snapshot from sequence npz")
    else:
        q = table["q"].astype(np.float32)
        print("[WARN] sequence npz has no embedded IK q snapshot; loading q from ik_table_npz path.")

    if "ik_table_grid_size" in seq_data:
        grid_size = float(seq_data["ik_table_grid_size"][0])
    else:
        grid_size = float(table["grid_size"][0]) if "grid_size" in table else args.grid_size

    if "ik_table_scene_offset" in seq_data:
        scene_offset = tuple(float(x) for x in seq_data["ik_table_scene_offset"])
    else:
        scene_offset = tuple(float(x) for x in table["scene_offset"]) if "scene_offset" in table else get_scene_offset(args)

    if robot_sample_ids.size > 0:
        max_sid = int(robot_sample_ids.max())
        if max_sid >= q.shape[0]:
            print(
                f"[WARN] robot_sample_ids max={max_sid} but q has only {q.shape[0]} rows. "
                "This sequence/table pair is inconsistent.",
                flush=True,
            )

    dual_yml = str(Path(args.dual_yml).resolve())

    viser_viz = ViserVisualizer(
        content_path=robot_content_path(dual_yml),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=False,
        visualize_robot_spheres=True,
        add_robot_to_scene=True,
    )

    config = KinematicsCfg.from_content_path(robot_content_path(dual_yml))
    robot = Kinematics(config)
    default_joint_state = robot.default_joint_state

    def board_from_state(state):
        out = board.copy()
        for pid in range(1, len(state) + 1):
            if state[pid - 1] == 0:
                out[out == pid] = -1
        return out

    def show_step(step_idx: int):
        step_idx = int(np.clip(step_idx, 0, states.shape[0] - 1))
        state = states[step_idx]
        step_board = board_from_state(state)

        try:
            viser_viz._server.scene.remove_by_name("obstacles")
        except Exception:
            pass
        scene = create_scene_from_voxels(
            board_to_voxels(step_board),
            grid_size=grid_size,
            include_table=True,
            scene_offset=scene_offset,
        )
        viser_viz.add_scene(scene)

        joint_state = default_joint_state.clone()
        s1, s2 = int(robot_sample_ids[step_idx, 0]), int(robot_sample_ids[step_idx, 1])
        if s1 >= 0:
            joint_state.position[:7] = torch.tensor(q[s1], device=joint_state.position.device, dtype=joint_state.position.dtype)
        if s2 >= 0:
            joint_state.position[7:14] = torch.tensor(q[s2], device=joint_state.position.device, dtype=joint_state.position.dtype)
        viser_viz.set_joint_state(joint_state=joint_state)

        held = (np.where(state == 2)[0] + 1).astype(int).tolist()
        if step_idx == 0:
            act_desc = "init"
        else:
            act_desc = f"{str(action_type[step_idx - 1])} {int(action_pid[step_idx - 1])}"
        print(
            f"[SEQ] step={step_idx}/{states.shape[0]-1} action={act_desc} "
            f"held={held} robot_samples=({s1}, {s2})",
            flush=True,
        )

    print(f"[SEQ] sequence={args.sequence_npz}")
    print(f"[SEQ] ik_table={ik_table_path}")
    print(f"[SEQ] steps={states.shape[0]}")
    print("[SEQ] open http://localhost:%d" % args.port)

    try:
        server = viser_viz._server
        step_options = [str(i) for i in range(states.shape[0])]
        step_dropdown = server.gui.add_dropdown("sequence_step", options=step_options, initial_value="0")
        show_button = server.gui.add_button("show sequence step")

        @show_button.on_click
        def _(_event):
            show_step(int(step_dropdown.value))

        play_button = server.gui.add_button("play sequence once")

        @play_button.on_click
        def _(_event):
            for step in range(states.shape[0]):
                show_step(step)
                time.sleep(args.step_delay)

        print("[SEQ] GUI ready: choose sequence_step or click play sequence once.")
    except Exception as e:
        print(f"[WARN] Could not create sequence GUI widgets: {e}")
        print("[SEQ] Falling back to automatic playback.")
        while True:
            for step in range(states.shape[0]):
                show_step(step)
                time.sleep(args.step_delay)

    show_step(0)
    while True:
        time.sleep(1.0)


def add_precompute_args(p):
    p.add_argument("--board_npy", type=str, default=str(PROJECT_DIR / "dataset/sequence/train/heart_maxcover_best.npy"))
    p.add_argument("--out_npz", type=str, default=str(PROJECT_DIR / "results/ik/heart_maxcover_best_ik_table.npz"))
    p.add_argument("--robot1_yml", type=str, default=str(ROBOT_DIR / "fr3_franka_hand0_optimized.yml"))
    p.add_argument("--robot2_yml", type=str, default=str(ROBOT_DIR / "fr3_franka_hand1_optimized.yml"))
    p.add_argument("--dual_yml", type=str, default=str(ROBOT_DIR / "franka_dual.yml"))
    p.add_argument("--grid_size", type=float, default=0.05)
    p.add_argument("--z_offset", type=float, default=0.0)
    p.add_argument("--max_batch_size", type=int, default=512)
    p.add_argument("--num_seeds", type=int, default=100)
    p.add_argument("--collision_batch_size", type=int, default=16384)
    p.add_argument("--collision_distance_threshold", type=float, default=0.0)
    p.add_argument("--pair_batch_size", type=int, default=65536)
    p.add_argument("--use_scene_collision", dest="use_scene_collision", action="store_true")
    p.add_argument("--no_scene_collision", dest="use_scene_collision", action="store_false")
    p.set_defaults(use_scene_collision=True)
    p.add_argument("--no_pair_collision", action="store_true", default=False)
    p.add_argument("--scene_offset_x", type=float, default=None)
    p.add_argument("--scene_offset_y", type=float, default=0.0)
    p.add_argument("--scene_offset_z", type=float, default=0.0)


def build_arg_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    p = sub.add_parser("precompute")
    add_precompute_args(p)

    b = sub.add_parser("precompute_dir")
    add_precompute_args(b)
    b.add_argument("--board_dir", type=str, default=str(PROJECT_DIR / "dataset/sequence/train"))
    b.add_argument("--out_dir", type=str, default="",
                   help='default: <board_dir>/ik_tables')

    v = sub.add_parser("view")
    v.add_argument("--table_npz", type=str, default=str(PROJECT_DIR / "dataset/sequence/train/structure_house_mask_16_maxcover_best_ik_table.npz"))
    v.add_argument("--dual_yml", type=str, default=str(ROBOT_DIR / "franka_dual.yml"))
    v.add_argument("--grid_size", type=float, default=0.05)
    v.add_argument("--port", type=int, default=8080)
    v.add_argument("--scene_offset_x", type=float, default=None)
    v.add_argument("--scene_offset_y", type=float, default=0.0)
    v.add_argument("--scene_offset_z", type=float, default=0.0)

    s = sub.add_parser("view_sequence")
    s.add_argument("--sequence_npz", type=str, default=str(PROJECT_DIR / "dataset/sequence/cat_maxcover_best_ik_sequence.npz"))
    s.add_argument("--dual_yml", type=str, default=str(ROBOT_DIR / "franka_dual.yml"))
    s.add_argument("--grid_size", type=float, default=0.05)
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--step_delay", type=float, default=0.8)
    s.add_argument("--scene_offset_x", type=float, default=None)
    s.add_argument("--scene_offset_y", type=float, default=0.0)
    s.add_argument("--scene_offset_z", type=float, default=0.0)

    return parser


def main():
    parser = build_arg_parser()

    if len(sys.argv) == 1:
        args = parser.parse_args(["precompute_dir"])
    else:
        args = parser.parse_args()

    if args.mode == "precompute":
        precompute_ik_table(args)
    elif args.mode == "precompute_dir":
        try:
            precompute_ik_dir(args)
        except RuntimeError as exc:
            parser.exit(1, f"ERROR: {exc}\n")
    elif args.mode == "view":
        launch_viewer(args)
    elif args.mode == "view_sequence":
        launch_sequence_viewer(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
