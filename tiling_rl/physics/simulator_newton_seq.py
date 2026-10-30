"""Newton stability simulation for sequence boards."""

import numpy as np
import warp as wp
import matplotlib.cm as cm
import jax.numpy as jnp

from skimage.measure import label
from newton.viewer import ViewerGL

viewer = None

def is_piece_stable(state_init, state_new, pos_tol=0.1, ang_tol=5, check_mask=None):
    num_bodies = state_init.shape[0]

    if check_mask is None:
        check_mask = np.ones(num_bodies, dtype=bool)
    else:
        check_mask = np.asarray(check_mask, dtype=bool)

    for body_id in range(num_bodies):
        if not check_mask[body_id]:
            continue

        s0 = np.array(state_init[body_id, :])
        s1 = np.array(state_new[body_id, :])

        pos0, pos1 = s0[:3], s1[:3]
        dist = np.linalg.norm(pos1 - pos0)

        q0, q1 = s0[3:], s1[3:]
        dot_product = np.clip(np.abs(np.dot(q0, q1)), 0.0, 1.0)
        angular_dist = 2 * np.arccos(dot_product)

        if dist > pos_tol or angular_dist > np.radians(ang_tol):
            return False

    return True

def check_stability_newton(boards, held_piece_ids=None, render: bool = True, keep_run=False, device=None):
    global viewer
    boards = np.array(boards)
    if boards.ndim == 2:
        boards = boards[None, ...]

    if held_piece_ids is None:
        held_piece_ids = [set() for _ in range(len(boards))]
    else:
        if len(held_piece_ids) != len(boards):
            raise ValueError("held_piece_ids should have the same batch size as boards.")
        held_piece_ids = [
            {int(pid) for pid in hs if int(pid) >= 0}
            for hs in held_piece_ids
        ]

    try:
        import newton
    except ImportError:
        print("Please ensure 'newton' library is available.")
        return False

    batch, height, width = boards.shape
    if batch == 0:
        return jnp.array([], dtype=jnp.bool_)

    builder = newton.ModelBuilder()
    ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.1)
    shape_idx = builder.add_ground_plane(cfg=ground_cfg)

    shape_colors = {}
    shape_colors[shape_idx] = cm.plasma(0)[:3]

    world_held_body_indices = []

    for board_idx in range(batch):
        builder.begin_world()

        dynamic_cfg = builder.ShapeConfig(mu=0.1, density=1000.0)
        held_cfg = builder.ShapeConfig(mu=0.1, density=0.0)

        board = boards[board_idx, :, :].astype(np.int32)

        held_set = set(held_piece_ids[board_idx])
        if np.any(board == 0):
            held_set.add(0)

        if board.size == 0 or not np.any(board >= 0):
            dummy_cfg = builder.ShapeConfig(mu=0.1, density=1000.0)
            dummy = builder.add_body(label="dummy")
            shape_idx = builder.add_shape_box(
                body=dummy,
                hx=0.5, hy=0.5, hz=0.5,
                cfg=dummy_cfg,
                xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
            )
            shape_colors[shape_idx] = cm.plasma(0)[:3]
            world_held_body_indices.append(set())

        else:
            piece_ids = sorted(int(x) for x in np.unique(board) if int(x) >= 0)
            body_map = {}
            uid_to_local = {}

            for local_idx, uid in enumerate(piece_ids):
                body_map[uid] = builder.add_body(label=f"piece_{uid}")
                uid_to_local[uid] = local_idx

            held_body_indices = {
                uid_to_local[uid]
                for uid in held_set
                if uid in uid_to_local
            }
            world_held_body_indices.append(held_body_indices)

            ground_offset = -0.5
            xy_offset = width / 2
            denom = max(len(piece_ids), 1)

            color_rank = {uid: k for k, uid in enumerate(piece_ids)}

            for r in range(board.shape[0]):
                for c in range(board.shape[1]):
                    uid = int(board[r, c])
                    if uid < 0:
                        continue

                    rgba = cm.plasma((color_rank[uid] + 1) / (denom + 1))
                    y = float(c) - xy_offset
                    z = float(board.shape[0] - r - 1) - ground_offset

                    cfg = held_cfg if uid in held_set else dynamic_cfg
                    shape_idx = builder.add_shape_box(
                        body=body_map[uid],
                        hx=0.4999,
                        hy=0.4999,
                        hz=0.4999,
                        cfg=cfg,
                        xform=wp.transform(
                            wp.vec3(0, y, z),
                            wp.quat_identity(),
                        ),
                    )
                    shape_colors[shape_idx] = rgba[:3]

        builder.end_world()

    if device is None:
        model = builder.finalize()
    else:
        model = builder.finalize(device=device)


    state_0 = model.state()
    state_1 = model.state()

    contacts = model.contacts()
    solver = newton.solvers.SolverXPBD(model, iterations=15, enable_restitution=True, rigid_contact_relaxation=0.8)
    control = model.control()

    if render:
        if viewer is None:
            viewer = ViewerGL()
            viewer.set_model(model)
        else:
            viewer.model = model
        viewer.update_shape_colors(shape_colors)
        viewer.set_camera(pos=wp.vec3(30, 0, 30), pitch=-50, yaw=180)

    dt = 0.005
    initial_positions = state_0.body_q.numpy().copy()
    for step in range(300):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt=dt)
        state_0, state_1 = state_1, state_0
        body_qd = state_0.body_qd.numpy()
        body_qd[:, 0] = 0.0
        body_qd[:, 4:] = 0.0

        if step%10==0 and render:
            viewer.begin_frame(step * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

    body_start = model.body_world_start.numpy()
    body_end = body_start[1:]
    body_start = body_start[:-1]
    current_positions = state_0.body_q.numpy()
    stable = []
    for world_id in range(model.world_count):
        start = body_start[world_id]
        end = body_end[world_id]
        initial_q = initial_positions[start:end, :]
        current_q = current_positions[start:end, :]
        world_num_bodies = end - start
        held_body_indices = world_held_body_indices[world_id]
        check_mask = np.ones(world_num_bodies, dtype=bool)
        for local_body_idx in held_body_indices:
            if 0 <= local_body_idx < world_num_bodies:
                check_mask[local_body_idx] = False
        stable.append(
            is_piece_stable(
                initial_q,
                current_q,
                pos_tol=0.05,
                ang_tol=5,
                check_mask=check_mask
            )
        )

    if render and keep_run:
        while True:
            viewer.begin_frame(step * dt)
            viewer.log_state(state_0)
            viewer.end_frame()

    return jnp.array(stable, dtype=jnp.bool_)
