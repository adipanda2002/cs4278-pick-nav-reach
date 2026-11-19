"""Run a simple-controller trial and archive its authoritative measurements."""
import argparse
import gzip
import hashlib
import json
import os
import signal
from pathlib import Path
from time import monotonic
import traceback
import zipfile

import numpy as np
import pybullet as p
import trimesh

from pick_nav_reach_env import PickNavReachEnv
from simple_controller.controller import Controller
from simple_controller.pick import plan_pick
from simple_controller.native import Native, PALM_POLICY


ROOT = Path(__file__).resolve().parent


class TrialTimeout(TimeoutError):
    pass


def time_limit(_signal, _frame):
    raise TrialTimeout("Per-case wall-clock budget exhausted")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, content):
    path.write_text(json.dumps(content, allow_nan=False, indent=2) + "\n")


def run(object_index, seed, output, plan_only):
    started = monotonic()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    config_path = ROOT / "simple_controller/config.json"
    config = json.loads(config_path.read_text())
    sources = [ROOT / name for name in ("run_simple.py", "pick_nav_reach_env.py",
                                        "maze_utils.py", "utils.py", "keyboard_control.py")]
    sources.extend(sorted((ROOT / "simple_controller").glob("*.py")))
    sources.append(config_path)
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    write_json(output / "source.json", source_hashes)
    with zipfile.ZipFile(output / "sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.relative_to(ROOT))
    write_json(output / "config.json", config)
    env = None
    previous_alarm = signal.signal(signal.SIGALRM, time_limit)
    signal.alarm(config["wall_limit_seconds"])
    try:
        env = PickNavReachEnv(seed=seed, object_idx=object_index, headless=True,
                              maze_path=output / "maze.urdf")
        scene_assets = {path for path in (ROOT / "assets/fetch").rglob("*") if path.is_file()}
        scene_assets.update(path for path in env.object_asset.parent.iterdir() if path.is_file())
        import pybullet_data
        data = Path(pybullet_data.getDataPath())
        for relative in ("plane.urdf", "plane100.obj", "table/table.urdf",
                         "table/table.obj", "table/table.mtl", "table/table.png"):
            scene_assets.add(data / relative)
        native = Native(env)
        geometry_path = output / "palm_geometry.json"
        write_json(geometry_path, native.palm_support.record)
        write_json(output / "scene.json", {
            "body_ids": {"plane": env.plane_id, "robot": env.robot_id, "table": env.table_id,
                         "object": env.object_id, "maze": env.maze_id, "goal": env.goal_marker_id},
            "hand_link": native.hand, "finger_links": sorted(native.fingers), "palm_policy": PALM_POLICY,
            "palm_geometry_sha256": file_hash(geometry_path),
            "object_index": object_index, "object_asset": str(env.object_asset.relative_to(ROOT)),
            "seed": seed, "maze_sha256": file_hash(output / "maze.urdf"),
            "cube_positions": env.cube_positions.tolist(), "goal_position": env.goal_pos.tolist(),
            "scene_assets": {os.path.relpath(path, ROOT) if path.is_relative_to(ROOT)
                             else "pybullet_data/" + str(path.relative_to(data)): file_hash(path)
                             for path in sorted(scene_assets)},
            "joint_indices": env.joint_indices, "joint_lower": env.joint_lower.tolist(),
            "joint_upper": env.joint_upper.tolist(), "substeps": env.substeps,
            "actuator_position_gains": [.3] * 13 + [.03] * 2,
            "actuator_max_forces": [env.max_force] * 15,
            "physics_parameters": p.getPhysicsEngineParameters(),
            "dynamics": {
                label: {str(link): p.getDynamicsInfo(body, link)
                        for link in range(-1, p.getNumJoints(body))}
                for label, body in (("robot", env.robot_id), ("object", env.object_id),
                                    ("table", env.table_id), ("maze", env.maze_id),
                                    ("plane", env.plane_id))},
            "pybullet_api_version": p.getAPIVersion(),
            "numpy_version": np.__version__, "trimesh_version": trimesh.__version__,
        })
        np.savez_compressed(output / "surface.npz", points=env.object_canonical_pc,
                            normals=env.object_canonical_normals)
        if plan_only:
            with gzip.open(output / "trace.jsonl.gz", "wt", encoding="utf-8", newline="") as trace:
                controller = Controller(env, config, trace)
                notes = {}
                # A static explicit surrogate at the URDF's launch orientation.
                # This mode advances zero physics and carries no trial score.
                n = controller.native
                with n.planning():
                    pos, quat = p.getBasePositionAndOrientation(n.object)
                    bottom = p.getAABB(n.object)[0][2]
                    p.resetBasePositionAndOrientation(n.object,
                        (pos[0], pos[1], pos[2] - bottom + controller.table_height), quat)
                    planned = plan_pick(n, env._get_obs(), controller.table_height, config, notes)
                result = {"termination": "planning_only", "planned": planned is not None,
                          "planning": notes, "calls": 0, "ticks": 0}
        else:
            with gzip.open(output / "trace.jsonl.gz", "wt", encoding="utf-8", newline="") as trace:
                result = Controller(env, config, trace).run()
        result["elapsed_seconds"] = monotonic() - started
        write_json(output / "result.json", result)
        return result
    except Exception:
        write_json(output / "error.json", {
            "traceback": traceback.format_exc(),
            "elapsed_seconds": monotonic() - started,
            "calls": env.step_count if env is not None else 0,
            "ticks": env.tick_count if env is not None else 0,
        })
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if env is not None:
            p.disconnect(env.pb_physics_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.object, args.seed, args.output, args.plan_only)))
