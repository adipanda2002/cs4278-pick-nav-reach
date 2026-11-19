"""Independently check archived collision vertices and measured pose projections."""

from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np


SCHEMA = "palm_collision_vertices_v1"


def rotation(quaternion):
    x, y, z, w = quaternion
    scale = 2 / (x*x + y*y + z*z + w*w)
    return np.array([
        [1-scale*(y*y+z*z), scale*(x*y-z*w), scale*(x*z+y*w)],
        [scale*(x*y+z*w), 1-scale*(x*x+z*z), scale*(y*z-x*w)],
        [scale*(x*z-y*w), scale*(y*z+x*w), 1-scale*(x*x+y*y)],
    ])


def vertices(values):
    points = np.asarray(values, dtype=float)
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) >= 4
    assert np.isfinite(points).all()
    return points


def source_vertices(mesh):
    if mesh.suffix.lower() == ".obj":
        return vertices([line.split()[1:4] for line in mesh.read_text().splitlines() if line.startswith("v ")])
    assert mesh.suffix.lower() == ".stl"
    import trimesh
    return vertices(trimesh.load_mesh(mesh, process=False).vertices)


def validate_role(p, root, scene, data, body, link, urdf, link_name):
    assert (data["body"], data["link"], data["link_name"], data["urdf"]) == (body, link, link_name, urdf)
    assert data["vertex_frame"] == "link_inertial_com"
    node = ET.parse(root / urdf).find(f"./link[@name='{link_name}']")
    (collision,) = node.findall("collision")
    mesh = collision.find("geometry/mesh")
    mesh_path = (root / urdf).parent / mesh.attrib["filename"]
    assert mesh_path.relative_to(root).as_posix() == data["mesh"]
    for name in (urdf, data["mesh"]):
        assert sha256((root/name).read_bytes()).hexdigest() == scene["scene_assets"][name]
    (shape,) = p.getCollisionShapeData(body, link)
    assert shape[2] == p.GEOM_MESH
    assert Path(shape[4].decode()).resolve() == mesh_path.resolve()
    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    assert tuple(scale) == tuple(data["mesh_scale"]) == shape[3]
    assert (list(shape[5]), list(shape[6])) == (data["collision_origin_com_m"], data["collision_origin_com_xyzw"])
    dynamics = p.getDynamicsInfo(body, link)
    assert tuple(data["link_to_com_m"]) == dynamics[3]
    assert tuple(data["link_to_com_xyzw"]) == dynamics[4]

    origin = collision.find("origin")
    offset = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    euler = np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    com_from_link = rotation(dynamics[4]).T
    com_from_collision = com_from_link @ rotation(p.getQuaternionFromEuler(euler))
    collision_origin = com_from_link @ (offset - dynamics[3])
    np.testing.assert_allclose(com_from_collision, rotation(shape[6]), atol=1e-7, rtol=0.)
    np.testing.assert_allclose(collision_origin, shape[5], atol=1e-7, rtol=0.)
    expected_source = (source_vertices(mesh_path) * scale) @ com_from_collision.T + collision_origin
    source = vertices(data["source_vertices_com_m"])
    np.testing.assert_allclose(source, expected_source, atol=1e-12, rtol=0.)
    exposed = vertices(data["native_vertices_com_m"])
    count, points = p.getMeshData(body, link, flags=p.MESH_DATA_SIMULATION_MESH)[:2]
    assert len(exposed) == count
    np.testing.assert_array_equal(exposed, points)
    np.testing.assert_allclose([source.min(axis=0), source.max(axis=0)],
                               [exposed.min(axis=0), exposed.max(axis=0)], atol=2.5e-5, rtol=0.)
    return np.vstack((exposed, source))


def verify_geometry(folder, root, scene):
    """Check the original read-only mesh extraction against the pinned scene."""
    raw = (folder / "palm_geometry.json").read_bytes()
    assert sha256(raw).hexdigest() == scene["palm_geometry_sha256"]
    data = json.loads(raw)
    assert data["schema"] == SCHEMA
    assert data["palm_pose_frame"] == "world_urdf_link"
    assert data["object_pose_frame"] == "world_base_com"
    if data["status"] == "unsupported":
        assert set(data) <= {"schema", "status", "palm_pose_frame", "object_pose_frame", "reason"}
        return None
    assert data["status"] == "ready"

    import pybullet as p
    from pick_nav_reach_env import PickNavReachEnv

    def trap(*args, **kwargs):
        raise AssertionError("Offline geometry verification cannot step")

    previous = Path.cwd()
    with TemporaryDirectory(prefix="pnr-audit038-") as scratch:
        try:
            os.chdir(root)
            with patch.object(p, "stepSimulation", trap), patch.object(PickNavReachEnv, "step", trap):
                env = PickNavReachEnv(scene["seed"], scene["object_index"], headless=True,
                                      maze_path=Path(scratch) / "maze.urdf")
                try:
                    assert sha256(env.maze_path.read_bytes()).hexdigest() == scene["maze_sha256"]
                    assert (env.robot_id, env.object_id) == (scene["body_ids"]["robot"], scene["body_ids"]["object"])
                    assert p.getPhysicsEngineParameters() == scene["physics_parameters"]
                    palm, obj = data["palm"], data["object"]
                    a = validate_role(p, root, scene, palm, env.robot_id, scene["hand_link"],
                                      "assets/fetch/fetch.urdf", "gripper_link")
                    b = validate_role(p, root, scene, obj, env.object_id, -1,
                                      scene["object_asset"], p.getBodyInfo(env.object_id)[0].decode())
                    assert env.step_count == env.tick_count == 0
                finally:
                    p.disconnect(env.pb_physics_client)
        finally:
            os.chdir(previous)
    return a, b, palm["link_to_com_m"], palm["link_to_com_xyzw"]


def measured_gap(geometry, row, world_normal):
    if geometry is None:
        return None
    n = np.asarray(world_normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all():
        return None
    length = np.linalg.norm(n)
    if abs(length-1.) > 1e-8:
        return None
    n = n / length
    a, b, offset, orientation = geometry
    hand_rot = rotation(row["hand_xyzw"])
    com_rot = hand_rot @ rotation(orientation)
    object_rot = rotation(row["object_xyzw"])
    # Relative world centres avoid cancellation from the robot's large world X.
    centre = np.asarray(row["hand_position"]) + hand_rot @ offset - row["object_position"]
    palm_support = np.min((a @ com_rot.T + centre) @ n)
    object_support = np.max((b @ object_rot.T) @ n)
    return float(palm_support - object_support)
