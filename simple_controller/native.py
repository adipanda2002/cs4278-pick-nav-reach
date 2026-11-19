"""Native geometry, state-preserving planning queries, and fixed-base IK."""
from contextlib import contextmanager
from itertools import product

import numpy as np
import pybullet as p

from .palm_support import FACE_AXIS_LIMIT, PalmSupport


EDGE_RESOLUTION = np.array([.022, .022, .05, .012, .05, .05] + [.055]*7 + [.005]*2)


PALM_POLICY = {
    "policy_id": "palm_geometry_v3",
    "source_face_axis_limit": FACE_AXIS_LIMIT,
    "source_face_axis_order": "alignment_to_own_normal_stable_lexicographic_com",
    "source_face_axis_basis": "signed_scaled_palm_source_triangle_normals",
    "geometry_normal_x_max_exclusive": 0.,
    "max_penetration": .0005,
    "front_x_min": -.03502512,
    "normal_x_max": -.707,
}


def intended_palm(distance, point_local, normal_local, held):
    return bool(held and distance >= -PALM_POLICY["max_penetration"]
                and point_local[0] >= PALM_POLICY["front_x_min"]
                and normal_local[0] <= PALM_POLICY["normal_x_max"])


def forward_palm_axis(object_link, candidate):
    if object_link != -1:
        return False
    point = np.asarray(candidate.get("point_local"), dtype=float)
    normal_local = np.asarray(candidate.get("normal_local"), dtype=float)
    normal_world = np.asarray(candidate.get("normal_on_object"), dtype=float)
    if not all(v.shape == (3,) and np.isfinite(v).all() for v in (point, normal_local, normal_world)):
        return False
    return bool(point[0] >= PALM_POLICY["front_x_min"]
                and normal_local[0] < PALM_POLICY["geometry_normal_x_max_exclusive"]
                and abs(np.linalg.norm(normal_world) - 1.) <= 1e-8)


def rotation(quaternion):
    return np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)


def rotation_error(target, current):
    transform = target @ current.T
    skew = np.array([transform[2, 1] - transform[1, 2],
                     transform[0, 2] - transform[2, 0],
                     transform[1, 0] - transform[0, 1]]) / 2
    angle = np.arccos(np.clip((np.trace(transform) - 1) / 2, -1, 1))
    magnitude = np.linalg.norm(skew)
    if magnitude > 1e-6:
        return skew * angle / magnitude
    if angle > 1.:
        _, vectors = np.linalg.eigh((transform + transform.T) / 2)
        return vectors[:, -1] * angle
    return skew


class Native:
    def __init__(self, env):
        self.env = env
        self.robot = env.robot_id
        self.object = env.object_id
        self.obstacles = (env.plane_id, env.table_id, env.maze_id)
        self.lower = env.joint_lower.astype(float)
        self.upper = env.joint_upper.astype(float)
        self.arm = np.array([3, 6, 7, 8, 9, 10, 11, 12])
        self.wrap = self.upper == 314
        infos = [p.getJointInfo(self.robot, i) for i in range(p.getNumJoints(self.robot))]
        names = {info[12]: info[0] for info in infos}
        self.hand = names[b"gripper_link"]
        if not hasattr(env, "_palm_support"):
            env._palm_support = PalmSupport(env, self.hand)
        self.palm_support = env._palm_support
        self.fingers = {names[b"r_gripper_finger_link"], names[b"l_gripper_finger_link"]}
        self.support = {names[name] for name in (b"base_link", b"r_wheel_link", b"l_wheel_link")}
        if len(env.joint_indices) != 15:
            raise RuntimeError("Unexpected Fetch joint layout")
        cluster = {-1: -1}
        adjacent = set()
        for info in infos:
            i, parent = info[0], info[16]
            cluster[i] = cluster[parent] if info[2] == p.JOINT_FIXED else i
            if cluster[i] != cluster[parent]:
                adjacent.add(frozenset((cluster[i], cluster[parent])))
        self.links = [i for i in cluster if p.getCollisionShapeData(self.robot, i)]
        self.self_pairs = [(a,b) for k,a in enumerate(self.links) for b in self.links[k+1:]
                           if cluster[a] != cluster[b]
                           and frozenset((cluster[a], cluster[b])) not in adjacent]

    @contextmanager
    def planning(self):
        state = p.saveState()
        try:
            yield
        finally:
            p.restoreState(state)
            p.removeState(state)

    def q(self):
        return np.array([s[0] for s in p.getJointStates(self.robot, self.env.joint_indices)])

    def joint_positions_and_velocities(self):
        states = np.asarray([state[:2] for state in
                             p.getJointStates(self.robot, self.env.joint_indices)])
        return states[:, 0].copy(), states[:, 1].copy()

    def set(self, q):
        for index, value in zip(self.env.joint_indices, q):
            p.resetJointState(self.robot, index, float(value))

    def hand_pose(self):
        state = p.getLinkState(self.robot, self.hand, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def delta(self, start, end):
        values = end - start
        values[self.wrap] = (values[self.wrap] + np.pi) % (2 * np.pi) - np.pi
        return values

    def ik(self, seed, position, orientation, iterations=85):
        q = seed.copy()
        zeros = [0.] * len(q)
        for _ in range(iterations):
            self.set(q)
            location, quaternion = self.hand_pose()
            error = np.r_[position - location, rotation_error(orientation, rotation(quaternion))]
            if np.linalg.norm(error[:3]) < .002 and np.linalg.norm(error[3:]) < .025:
                return q
            linear, angular = p.calculateJacobian(self.robot, self.hand, (0.,0.,0.),
                                                  q.tolist(), zeros, zeros)
            jacobian = np.r_[linear, angular][:, self.arm]
            jacobian[3:] *= .45
            error[3:] *= .45
            update = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .0025*np.eye(6), error)
            q[self.arm] = np.clip(q[self.arm] + np.clip(update, -.16, .16),
                                  self.lower[self.arm], self.upper[self.arm])
        return None

    def payload_transform(self):
        hand, quaternion = self.hand_pose()
        inverse = p.invertTransform(hand, quaternion)
        return p.multiplyTransforms(*inverse, *p.getBasePositionAndOrientation(self.object))

    def position_payload(self, relative):
        world = p.multiplyTransforms(*self.hand_pose(), *relative)
        p.resetBasePositionAndOrientation(self.object, *world)

    def palm_geometry(self, point):
        origin, quaternion = self.hand_pose()
        local = rotation(quaternion).T
        return {
            "distance": float(point[8]),
            "point_on_palm": list(point[5]),
            "normal_on_object": list(point[7]),
            "point_local": (local @ (np.asarray(point[5]) - origin)).tolist(),
            "normal_local": (local @ point[7]).tolist(),
        }

    def violations(self, q, fingers=False, lifted=False, *, held_palm=False, palm_candidates=None, finger_contact_callback=None, finger_contact_witnesses=None, overlap_witnesses=None):
        result = []
        contact_records = []
        if (not np.all(np.isfinite(q)) or np.any(q < self.lower - .001)
                or np.any(q > self.upper + .001)):
            result.append(("joint_limit",))
        boxes = {link: np.array(p.getAABB(self.robot, link)) for link in self.links}
        for a, b in self.self_pairs:
            aa, bb = boxes[a], boxes[b]
            if (aa[0] <= bb[1]).all() and (bb[0] <= aa[1]).all():
                self_hits = p.getClosestPoints(self.robot, self.robot, 0., linkIndexA=a, linkIndexB=b)
                if overlap_witnesses is not None:
                    overlap_witnesses.extend({'kind':'self','point':list(point)} for point in self_hits)
                if self_hits:
                    result.append(("self", a, b))
        # getClosestPoints is independent of the URDF self-collision flag and
        # remains valid after hypothetical state changes during planning.
        for body in (*self.obstacles, self.object):
            points = p.getClosestPoints(self.robot, body, 0.)
            if overlap_witnesses is not None:
                overlap_witnesses.extend({'kind':'robot_scene','point':list(point)} for point in points)
            palm_records = ([(point[4], self.palm_geometry(point)) for point in points if point[3] == self.hand]
                            if body == self.object else [])
            palm_index = -1
            axis_gaps = {}
            face_gaps = {}
            measured_poses = None
            for point in points:
                if (finger_contact_callback is not None and not fingers
                        and body == self.object and point[3] in self.fingers):
                    record = finger_contact_callback(point)
                    contact_records.append(record)
                    if finger_contact_witnesses is not None:
                        finger_contact_witnesses.append(record)
                if body == self.env.plane_id and point[3] in self.support:
                    continue
                if fingers and body == self.object and point[3] in self.fingers:
                    continue
                if body == self.object and point[3] == self.hand:
                    palm_index += 1
                    object_link, candidate = palm_records[palm_index]
                    accepted = object_link == -1 and intended_palm(
                        candidate["distance"], candidate["point_local"], candidate["normal_local"], held_palm)
                    if not accepted and held_palm and forward_palm_axis(object_link, candidate):
                        support = getattr(self, "palm_support", None)
                        if support is not None and measured_poses is None:
                            measured_poses = self.hand_pose(), p.getBasePositionAndOrientation(self.object)
                        order = [palm_index] + [
                            index for index, (link, record) in enumerate(palm_records)
                            if index != palm_index and forward_palm_axis(link, record)
                        ]
                        gap, selected = None, None
                        for axis_index in order:
                            axis = palm_records[axis_index][1]["normal_on_object"]
                            if axis_index not in axis_gaps:
                                axis_gaps[axis_index] = (None if support is None
                                                         else support.gap(*measured_poses, axis))
                            trial = axis_gaps[axis_index]
                            if axis_index == palm_index and trial is not None:
                                gap, selected = trial, axis_index
                            if trial is not None and trial >= 0.:
                                gap, selected, accepted = trial, axis_index, True
                                break
                        candidate.update(support_gap_m=gap, support_axis_index=selected,
                                         support_axis_world=None if selected is None else
                                         palm_records[selected][1]["normal_on_object"])
                        if not accepted and support is not None:
                            face = support.face_certificate(
                                *measured_poses, candidate["normal_on_object"], face_gaps)
                            if face is not None:
                                face_index, face_axis, face_gap = face
                                candidate.update(face_axis_index=face_index, face_axis_world=face_axis,
                                                 face_gap_m=face_gap)
                                accepted = True
                    if palm_candidates is not None:
                        palm_candidates.append({**candidate, "accepted": accepted, "object_link": point[4]})
                    if accepted:
                        continue
                result.append(("robot_scene", body, point[3], point[4], point[8]))
        payload_obstacles = self.obstacles if lifted else (self.env.plane_id, self.env.maze_id)
        for body in payload_obstacles:
            payload_points = p.getClosestPoints(self.object, body, 0.)
            if overlap_witnesses is not None:
                overlap_witnesses.extend({'kind':'payload_scene','point':list(point)} for point in payload_points)
            for point in payload_points:
                result.append(("payload_scene", body, point[4], point[8]))
        if finger_contact_callback is not None:
            from .early_contact import admit_only_finger_contacts
            return admit_only_finger_contacts(result, contact_records)
        return result

    def contacts(self):
        contacts = {str(i): [] for i in sorted(self.fingers)}
        palm_contacts = []
        for point in p.getContactPoints(self.robot, self.object):
            if point[3] == self.hand and point[4] == -1:
                palm_contacts.append({**self.palm_geometry(point), "normal_force": float(point[9])})
            if point[3] in self.fingers and point[9] > 0. and point[8] <= .001:
                contacts[str(point[3])].append([point[8], point[9], list(point[5]), list(point[6])])
        return contacts, palm_contacts

    def edge(self, start, target, *, fingers=False, payload=None, lifted=False, held_palm=False):
        if held_palm and (payload is None or not fingers):
            raise ValueError("Held palm planning requires an explicit rigid payload and finger contact context")
        delta = self.delta(start, target)
        samples = max(1, int(np.ceil(np.max(np.abs(delta) / EDGE_RESOLUTION))))
        for fraction in np.linspace(0., 1., samples + 1):
            q = start + fraction * delta
            self.set(q)
            if payload is not None:
                self.position_payload(payload)
            if self.violations(q, fingers=fingers, lifted=lifted, held_palm=held_palm):
                return False
        return True


    def pregrasp_aperture_corridor(self, start, target, aperture_low, aperture_high):
        """Sample the measured/open-finger rectangle along an unloaded body edge."""
        delta = self.delta(start, target)
        count = max(1, int(np.ceil(np.max(np.abs(delta[:-2]) / EDGE_RESOLUTION[:-2]))))
        values = [np.linspace(low, high, 1 + int(np.ceil((high - low) / .001)))
                  for low, high in zip(aperture_low, aperture_high)]
        pairs = list(product(*values))
        for fraction in np.linspace(0., 1., count + 1):
            q = start + fraction * delta
            for aperture in pairs:
                q[-2:] = aperture
                self.set(q)
                if self.violations(q):
                    return False
        return True
