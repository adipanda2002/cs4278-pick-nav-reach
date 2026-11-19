"""Support-plane certificates for the measured palm and payload collision geometry."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "palm_collision_vertices_v1"
FACE_AXIS_LIMIT = 64
# Provenance tolerances only. The certificate uses both complete vertex sets
# and compares their signed gap with zero without a clearance tolerance.
FRAME_ATOL_M = 1e-7
MESH_BOUND_ATOL_M = 2.5e-5


def rotation(xyzw):
    return np.asarray(p.getMatrixFromQuaternion(xyzw)).reshape(3, 3)


def finite_vertices(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError("Invalid collision vertices")
    return vertices


def source_mesh_vertices(path):
    if path.suffix.lower() == ".obj":
        with path.open() as mesh:
            return finite_vertices([line.split()[1:4] for line in mesh if line.startswith("v ")])
    if path.suffix.lower() == ".stl":
        return finite_vertices(trimesh.load_mesh(path, process=False).vertices)
    raise ValueError("Unsupported rigid collision mesh format")


def collision_mesh_record(body, link, urdf, link_name):
    link_node = ET.parse(urdf).find(f"./link[@name='{link_name}']")
    collisions = [] if link_node is None else link_node.findall("collision")
    metadata = p.getCollisionShapeData(body, link)
    if len(collisions) != 1 or len(metadata) != 1 or metadata[0][2] != p.GEOM_MESH:
        raise ValueError("Expected one source rigid mesh matched to native collision metadata")
    collision = collisions[0]
    mesh = collision.find("geometry/mesh")
    if mesh is None:
        raise ValueError("Expected a source collision mesh")
    source_path = urdf.parent / mesh.attrib["filename"]
    native_shape = metadata[0]
    if Path(native_shape[4].decode()).resolve() != source_path.resolve():
        raise ValueError("Native collision mesh differs from the URDF")
    scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0.) or not np.array_equal(
            scale, native_shape[3]):
        raise ValueError("Native collision mesh scale differs from the URDF")

    dynamics = p.getDynamicsInfo(body, link)
    link_to_com_m, link_to_com_xyzw = dynamics[3:5]
    origin = collision.find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
        raise ValueError("Invalid URDF collision origin")
    com_from_link = rotation(link_to_com_xyzw).T
    source_com_xyz = com_from_link @ (xyz - np.asarray(link_to_com_m))
    source_com_rotation = com_from_link @ rotation(p.getQuaternionFromEuler(rpy))
    native_xyz, native_xyzw = native_shape[5:7]
    if not (np.allclose(source_com_xyz, native_xyz, rtol=0., atol=FRAME_ATOL_M)
            and np.allclose(source_com_rotation, rotation(native_xyzw), rtol=0., atol=FRAME_ATOL_M)):
        raise ValueError("Source and native collision frames disagree")

    source = (source_mesh_vertices(source_path) * scale) @ source_com_rotation.T + source_com_xyz
    count, exposed = p.getMeshData(body, link, flags=p.MESH_DATA_SIMULATION_MESH)[:2]
    exposed = finite_vertices(exposed)
    if count != len(exposed):
        raise ValueError("Native vertex count differs from the reported count")
    if not np.allclose([source.min(axis=0), source.max(axis=0)],
                       [exposed.min(axis=0), exposed.max(axis=0)], rtol=0., atol=MESH_BOUND_ATOL_M):
        raise ValueError("Source and exposed native mesh bounds disagree")
    return {
        "body": body, "link": link, "link_name": link_name,
        "urdf": urdf.relative_to(ROOT).as_posix(), "mesh": source_path.relative_to(ROOT).as_posix(),
        "mesh_scale": scale.tolist(), "vertex_frame": "link_inertial_com",
        "link_to_com_m": list(link_to_com_m), "link_to_com_xyzw": list(link_to_com_xyzw),
        "collision_origin_com_m": list(native_xyz), "collision_origin_com_xyzw": list(native_xyzw),
        "native_vertices_com_m": exposed.tolist(), "source_vertices_com_m": source.tolist(),
    }


def source_face_axes(triangles, scale, com_from_collision):
    """Return signed unit axes in palm inertial COM coordinates, in lexicographic order."""
    triangles = np.asarray(triangles, dtype=float) * scale
    if (triangles.ndim != 3 or triangles.shape[1:] != (3, 3)
            or not np.isfinite(triangles).all()):
        raise ValueError("Invalid source collision triangles")
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    axes = cross @ com_from_collision.T
    lengths = np.linalg.norm(axes, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0.)
    axes = axes[valid] / lengths[valid, None]
    return np.unique(np.vstack((axes, -axes)), axis=0)


def support_gap(palm, obj, palm_link_pose, object_com_pose, normal_world, link_to_com):
    """Signed palm-min minus object-max projection in world metres."""
    normal = np.asarray(normal_world, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        return None
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or abs(length - 1.) > 1e-8:
        return None
    normal = normal / length
    palm_pos, palm_xyzw = palm_link_pose
    obj_pos, obj_xyzw = object_com_pose
    com_offset, com_xyzw = link_to_com
    for position, quaternion in (palm_link_pose, object_com_pose, link_to_com):
        position, quaternion = np.asarray(position), np.asarray(quaternion)
        if position.shape != (3,) or quaternion.shape != (4,):
            return None
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
            return None
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm == 0.:
            return None
    world_from_link = rotation(palm_xyzw)
    world_from_palm_com = world_from_link @ rotation(com_xyzw)
    world_palm_com = np.asarray(palm_pos) + world_from_link @ com_offset
    world_from_object_com = rotation(obj_xyzw)
    value = ((world_palm_com - np.asarray(obj_pos)) @ normal
             + np.min(palm @ (world_from_palm_com.T @ normal))
             - np.max(obj @ (world_from_object_com.T @ normal)))
    return float(value) if np.isfinite(value) else None


class PalmSupport:
    def __init__(self, env, hand):
        self.face_axes_com = np.empty((0, 3))
        self.record = {"schema": SCHEMA, "status": "unsupported",
                       "palm_pose_frame": "world_urdf_link", "object_pose_frame": "world_base_com"}
        if hand != 20:
            return
        try:
            palm = collision_mesh_record(env.robot_id, hand, ROOT / "assets/fetch/fetch.urdf", "gripper_link")
            obj = collision_mesh_record(env.object_id, -1, env.object_asset,
                                        p.getBodyInfo(env.object_id)[0].decode())
        except (ValueError, OSError, KeyError, TypeError, ET.ParseError, p.error) as error:
            self.record["reason"] = str(error)
            return
        self.record.update(status="ready", palm=palm, object=obj)
        self.palm = np.concatenate((palm["native_vertices_com_m"], palm["source_vertices_com_m"]))
        self.obj = np.concatenate((obj["native_vertices_com_m"], obj["source_vertices_com_m"]))
        self.link_to_com = palm["link_to_com_m"], palm["link_to_com_xyzw"]
        try:
            mesh = trimesh.load_mesh(ROOT / palm["mesh"], process=False)
            self.face_axes_com = source_face_axes(
                mesh.triangles, palm["mesh_scale"], rotation(palm["collision_origin_com_xyzw"]))
        except (ValueError, OSError, TypeError, AttributeError):
            # Additional source faces are optional. Retain the native-axis proof.
            pass

    def gap(self, palm_link_pose, object_com_pose, normal_world):
        if self.record["status"] != "ready":
            return None
        return support_gap(self.palm, self.obj, palm_link_pose, object_com_pose, normal_world,
                           self.link_to_com)

    def face_certificate(self, palm_link_pose, object_com_pose, normal_world, query_gaps):
        """Return the first complete separator within the bounded face search."""
        if not len(self.face_axes_com):
            return None
        world_from_com = rotation(palm_link_pose[1]) @ rotation(self.link_to_com[1])
        world_axes = self.face_axes_com @ world_from_com.T
        choices = np.argsort(-(world_axes @ normal_world), kind="stable")[:FACE_AXIS_LIMIT]
        for index in choices:
            direction = world_axes[index]
            if index not in query_gaps:
                query_gaps[index] = self.gap(palm_link_pose, object_com_pose, direction)
            gap = query_gaps[index]
            if gap is not None and gap >= 0.:
                return int(index), direction.tolist(), gap
        return None
