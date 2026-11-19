"""Drive the supplied environment through one scored physical attempt."""
from collections import deque
import json

import numpy as np
import pybullet as p

from .motion import differential_open_target
from .loaded_heading import heading_carry
from .native import Native, rotation, rotation_error
from .navigation import footprint_blocker, route
from .pick import plan_pick


LOADED_PHASES = ("lift", "hold", "retract", "navigate")


class StopTrial(RuntimeError):
    pass


class EarlyContactReady(Exception):
    """Internalcall-boundary event, never a claimedpickup or scoringexception."""


class Controller:
    def __init__(self, env, config, trace):
        self.env, self.config, self.trace = env, config, trace
        self.native = Native(env)
        self.physics_dt = p.getPhysicsEngineParameters()["fixedTimeStep"]
        env.tick_observer = self.observe
        self.phase = "settle"
        self.table_height = float(p.getAABB(env.table_id)[1][2])
        self.initial_pose = None
        self.initial_bottom = None
        self.closure_command = None
        self.extra_lift_active = False
        self.pick = None
        self.arrival = None
        self.hold_calls = 0
        self.held_call = False
        self.last = None
        self.guard = None
        self.target_pose = None
        self.extra = {"table_height": self.table_height}

    def observe(self, target, substep):
        n, e = self.native, self.env
        q = n.q()
        pos, quat = p.getBasePositionAndOrientation(n.object)
        bottom = p.getAABB(n.object)[0][2]
        contacts, palm_contacts = n.contacts()
        supported = bool(p.getContactPoints(n.object, e.table_id))
        finger_contact = all(contacts.values())
        guarding_fingers = self.phase in ("close", "lift", "hold", "retract", "navigate")
        carrying = self.extra_lift_active or self.phase in ("hold", "retract", "navigate")
        closure_confirmed = self.closure_command is not None
        palm_candidates = []
        contact_records, overlaps = [], []
        callback = None
        if self.phase == 'approach':
            def callback(point):
                classifier = getattr(self,'_early_contact_classifier',None)
                if classifier is None:
                    from .early_contact import FingerPads
                    classifier = FingerPads(n)
                    self._early_contact_classifier = classifier
                return classifier.classify(point,contacts)
        violations = n.violations(q, fingers=guarding_fingers, lifted=carrying,
                                  held_palm=closure_confirmed and finger_contact,
                                  palm_candidates=palm_candidates,
                                  finger_contact_callback=callback,
                                  finger_contact_witnesses=contact_records if callback else None,
                                  overlap_witnesses=overlaps if callback else None)
        if contact_records and not violations and all(x['accepted'] for x in contact_records):
            if not getattr(self,'early_contact_pending',False):
                event={'call':e.step_count,'tick':e.tick_count,'substep':substep,
                       'links':sorted({x['link'] for x in contact_records})}
                self.extra.setdefault('early_contact_events',[]).append(event)
            self.early_contact_pending = True
        if carrying and substep == e.substeps and (not finger_contact or supported):
            violations.append(("lost_hold",))
        eligible = (self.initial_pose is not None and finger_contact and not supported
                    and bottom - self.table_height >= .1 and bottom - self.initial_bottom >= .1
                    and np.linalg.norm(np.array(pos) - self.initial_pose) >= .1)
        if violations and self.guard is None:
            self.guard = violations[0]
        distance = float(np.linalg.norm(np.array(pos) - e.goal_pos))
        hand_pos, hand_quat = n.hand_pose()
        row = {"call": e.step_count, "tick": e.tick_count, "substep": substep,
               "phase": self.phase, "target": target.tolist(), "q": q.tolist(),
               "object_position": pos, "object_xyzw": quat, "object_bottom": bottom,
               "finger_contacts": contacts, "object_supported": supported,
               "closure_confirmed": closure_confirmed, "palm_candidates": palm_candidates,
               "palm_contacts": palm_contacts,
               "hand_position": hand_pos.tolist(), "hand_xyzw": hand_quat.tolist(),
               "goal_distance": distance, "lift_eligible": bool(eligible),
               "violations": violations}
        if getattr(self,'early_contact_pending',False):
            row['early_contact_pending'] = True
            row['early_contact_finger_candidates'] = contact_records
            row['early_contact_native_overlaps'] = overlaps
        self.trace.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.last = row
        if substep != e.substeps:
            return
        self.held_call = finger_contact and not supported
        self.hold_calls = self.hold_calls + 1 if eligible and not self.guard else 0
        if self.pick is None and self.hold_calls >= 50:
            self.pick = {"call": e.step_count, "tick": e.tick_count}
        if self.pick and self.phase == "navigate" and self.held_call and not self.guard and distance <= .1:
            self.arrival = {"call": e.step_count, "tick": e.tick_count, "distance": distance}

    def step(self, q, *, allow_approach_feedback=True):
        if self.pick is None and self.env.step_count >= self.config["pickup_call_limit"]:
            raise StopTrial("pickup_call_limit")
        if self.pick is not None and self.env.step_count - self.pick["call"] >= self.config["postpickup_call_limit"]:
            raise StopTrial("postpickup_call_limit")
        if self.phase in LOADED_PHASES:
            if self.closure_command is None:
                raise RuntimeError("Loaded motion requires a bilateral closure command")
            q = q.copy()
            q[-2:] = self.closure_command
        elif self.phase == "pregrasp":
            q = self.pregrasp_finger_feedback(q)
        elif self.phase == "approach" and allow_approach_feedback:
            q = self.pregrasp_finger_feedback(q, require_near_stop=False)
        self.env.step(q)
        if self.guard:
            raise StopTrial("physical_guard")
        if self.phase == 'approach' and getattr(self,'early_contact_pending',False):
            raise EarlyContactReady()

    def pregrasp_finger_feedback(self, target, *, require_near_stop=True):
        n = self.native
        measured, velocity = n.joint_positions_and_velocities()
        lower, upper = n.lower[-2:], n.upper[-2:]
        nominal = np.clip(target[-2:], lower, upper)
        predicted = measured[-2:] + 2 * self.physics_dt * velocity[-2:]
        correction = differential_open_target(nominal, predicted, lower, upper,
                                              require_near_stop=require_near_stop)
        if correction is None:
            return target
        adjusted = target.copy()
        adjusted[-2:] = correction
        apertures = np.stack((measured[-2:], nominal, correction,
                              np.clip(predicted, lower, upper)))
        aperture_low, aperture_high = apertures.min(axis=0), apertures.max(axis=0)
        with n.planning():
            corridor_free = n.pregrasp_aperture_corridor(measured, adjusted, aperture_low, aperture_high)
            edge_free = corridor_free and n.edge(measured, adjusted)
        log_key = "open_finger_feedback" if require_near_stop else "approach_open_finger_feedback"
        self.extra.setdefault(log_key, []).append({
            "call": self.env.step_count + 1, "measured": measured[-2:].tolist(),
            "velocity": velocity[-2:].tolist(), "prediction": predicted.tolist(),
            "nominal": nominal.tolist(), "adjusted": correction.tolist(),
            "aperture_low": aperture_low.tolist(), "aperture_high": aperture_high.tolist(),
            "sampled_corridor_free": corridor_free, "adjusted_edge_free": edge_free,
            "issued": (correction if edge_free else nominal).tolist(),
        })
        return adjusted if edge_free else target

    def follow(self, target, phase, *, continuous=False):
        self.phase = phase
        start = self.native.q()
        if phase in LOADED_PHASES:
            target = target.copy()
            target[-2:] = self.closure_command
        delta = self.native.delta(start, target)
        spacing = np.array([.012, .012, .037, .007, .04, .04] + [.04]*7 + [.005]*2)
        # Preserve the legacy body cadence. Only dispatch replaces the
        # interpolated finger samples with the original closure command.
        count = max(1, int(np.ceil(np.max(np.abs(delta) / spacing))))
        for fraction in np.linspace(0., 1., count + 1)[1:]:
            self.step(start + delta * fraction)
            if self.arrival:
                return
        if continuous:
            return
        tolerance = np.array([.005, .005, .02, .004, .02, .02] + [.02]*7 + [.002]*2)
        # In grasped phases the closed finger targets lie beneath the material
        # contact surfaces, so convergence here concerns the other joints.
        if phase in LOADED_PHASES:
            tolerance[-2:] = np.inf
        # Opening targets finger motion from the settled configuration. Native
        # geometry guards remain active for every joint throughout this phase.
        if phase == "open":
            tolerance[:-2] = np.inf
        for _ in range(self.config["waypoint_wait_calls"]):
            if (np.abs(self.native.delta(self.native.q(), target)) <= tolerance).all():
                return
            self.step(target)
            if self.arrival:
                return
        if phase == "approach":
            return self.finish_approach(target, tolerance)
        raise StopTrial(f"{phase}_convergence")

    def _approach_finger_correction(self, current, target, tolerance):
        """Propose an opposed opening correction after recovery stalls."""
        n = self.native
        error = n.delta(target, current)
        fingers = error[-2:]
        nominal = target[-2:]
        if (np.any(np.abs(error[:-2]) > tolerance[:-2])
                or not np.all(np.abs(fingers) > tolerance[-2:])
                or fingers[0] * fingers[1] >= 0.
                or nominal[0] != nominal[1]):
            return None
        if (self.last is None or any(self.last["finger_contacts"].values())
                or self.last["palm_contacts"]):
            return None

        _, velocity = n.joint_positions_and_velocities()
        predicted = current[-2:] + 2 * self.physics_dt * velocity[-2:]
        if not np.all(fingers * (predicted - nominal) > 0.):
            return None
        correction = differential_open_target(nominal, predicted, n.lower[-2:], n.upper[-2:],
                                              require_near_stop=False)
        if correction is None:
            return None
        return correction, predicted, velocity[-2:]

    def finish_approach(self, target, tolerance):
        """Continue a lagging unloaded waypoint under the original tolerances."""
        n = self.native

        def residual(q):
            return float(np.max(np.abs(n.delta(q, target)) / tolerance))

        current = n.q()
        error = residual(current)
        best = error
        stagnant = 0
        attempt = {"entry_call": self.env.step_count, "entry_error": error,
                   "target": target.tolist(), "tolerance": tolerance.tolist(), "steps": []}
        self.extra.setdefault("approach_recovery", []).append(attempt)
        while error > 1.:
            command = target
            low = np.minimum(current[-2:], target[-2:])
            high = np.maximum(current[-2:], target[-2:])
            feedback = self._approach_finger_correction(current, target, tolerance) if stagnant else None
            if feedback is not None:
                correction, predicted, velocity = feedback
                command = target.copy()
                command[-2:] = correction
                apertures = np.stack((low, high, correction,
                                      np.clip(predicted, n.lower[-2:], n.upper[-2:])))
                low, high = apertures.min(axis=0), apertures.max(axis=0)
            with n.planning():
                corridor_free = n.pregrasp_aperture_corridor(current, command, low, high)
                edge_free = corridor_free and n.edge(current, command)
            evidence = {"previous_call": self.env.step_count, "error_before": error,
                        "aperture_low": low.tolist(), "aperture_high": high.tolist(),
                        "sampled_corridor_free": corridor_free, "sampled_edge_free": edge_free}
            if feedback is not None:
                evidence["finger_feedback"] = {
                    "measured": current[-2:].tolist(), "velocity": velocity.tolist(),
                    "prediction": predicted.tolist(), "command": correction.tolist(),
                    "dispatched": False,
                }
            attempt["steps"].append(evidence)
            if not edge_free:
                attempt["stop"] = "native_block"
                raise StopTrial("approach_convergence")
            try:
                self.step(command, allow_approach_feedback=False)
            except StopTrial as exc:
                evidence["after_call"] = self.env.step_count
                attempt["stop"] = str(exc)
                raise
            finally:
                if feedback is not None:
                    evidence["finger_feedback"]["dispatched"] = (
                        self.env.step_count > evidence["previous_call"])
            current = n.q()
            error = residual(current)
            if error <= best - self.config["approach_progress_fraction"]:
                best, stagnant = error, 0
            else:
                stagnant += 1
            evidence.update(after_call=self.env.step_count, error_after=error,
                            nonimproving_calls=stagnant)
            if error > 1. and stagnant >= self.config["approach_stagnation_calls"]:
                attempt["stop"] = "no_progress"
                raise StopTrial("approach_convergence")
        attempt["stop"] = "converged"

    def extend_lift(self, required_bottom):
        n = self.native
        evidence = []
        self.extra["measured_lift"] = evidence
        stagnant = 0
        preferred_bottom = required_bottom + self.config["lift_extension_clearance"]
        self.extra_lift_active = True
        try:
            while self.pick is None:
                before = self.last["object_bottom"]
                if not self.held_call:
                    raise StopTrial("strict_pick_not_qualified")
                start = n.q()
                location, quaternion = n.hand_pose()
                orientation = rotation(quaternion)
                relative = n.payload_transform()
                target = None
                requested = self.config["lift_extension_step"]
                attempt = {"call": self.env.step_count, "bottom_before": before,
                           "required_bottom": required_bottom, "preferred_bottom": preferred_bottom,
                           "queries": []}
                evidence.append(attempt)
                with n.planning():
                    for rise in (requested, requested / 2):
                        candidate = n.ik(start, location + [0., 0., rise], orientation)
                        if candidate is not None:
                            candidate[-2:] = start[-2:]
                        if candidate is None:
                            attempt["queries"].append([rise, "ik"])
                        elif not n.edge(start, candidate, fingers=True, payload=relative,
                                        lifted=True, held_palm=True):
                            attempt["queries"].append([rise, "edge"])
                        else:
                            attempt["queries"].append([rise, "planned"])
                            target = candidate
                            break
                if target is None and before < required_bottom:
                    raise StopTrial("measured_lift_unreachable")
                if target is not None:
                    self.follow(target, "lift")
                after = self.last["object_bottom"]
                attempt["bottom_after"] = after
                if self.pick:
                    return
                stagnant = stagnant + 1 if after - before < .001 else 0
                # The clearance reserve is a motion target. If the arm cannot
                # make progress toward it, hold at any already qualifying height.
                if after >= preferred_bottom or (after >= required_bottom
                                                  and (target is None or stagnant >= 2)):
                    self.phase = "hold"
                    hold_target = start if target is None else target
                    for _ in range(self.config["hold_calls"]):
                        self.step(hold_target)
                        if self.pick:
                            return
                        if self.last["object_bottom"] < required_bottom:
                            break
                    if self.last["object_bottom"] >= required_bottom:
                        raise StopTrial("strict_pick_not_qualified")
                if stagnant >= 2:
                    raise StopTrial("measured_lift_no_progress")
        finally:
            self.extra_lift_active = False

    @staticmethod
    def goal_base_xy(object_position, base_xy, goal):
        shift = goal[:2] - object_position[:2]
        projected = object_position + np.r_[shift, 0.]
        if np.linalg.norm(projected - goal) > .1:
            return None
        return base_xy + shift

    def choose_carry_prefix(self, current, base_xy, prefixes, certify_route):
        ranked = []
        for path in prefixes:
            self.native.set(path[-1])
            hand, _ = self.native.hand_pose()
            reach = float(np.linalg.norm(hand[:2] - base_xy))
            ranked.append((reach, path))
        ranked.sort(key=lambda item: (item[0], -len(item[1])))
        # The already held configuration is a geometric fallback of last resort.
        ranked.append((None, ()))
        notes = {"candidates": len(ranked), "checked": [], "realized_edges": []}
        self.extra["carry_prefix"] = notes
        for reach, path in ranked:
            endpoint = path[-1] if path else current
            certified = certify_route(endpoint)
            attempt = {"segments": len(path), "hand_reach": reach,
                       "route_certified": certified is not None}
            notes["checked"].append(attempt)
            if certified is not None:
                waypoints, obj = certified
                attempt["waypoints"] = len(waypoints)
                return path, waypoints, obj, "prefix" if path else "unretracted"
        return None

    def certify_terminal_reach(self, endpoint, relative, notes):
        """Check one modeled terminal correction before selecting direct carry."""
        n = self.native
        n.set(endpoint)
        n.position_payload(relative)
        modeled_object = np.array(p.getBasePositionAndOrientation(n.object)[0])
        error = self.env.goal_pos - modeled_object
        distance = np.linalg.norm(error)
        notes["modeled_distance_before"] = float(distance)
        if distance <= .1:
            return True
        location, xyzw = n.hand_pose()
        orientation = rotation(xyzw)
        requested = min(self.config["goal_reach_step"], distance)
        for reach in (requested, requested / 2):
            n.set(endpoint)
            n.position_payload(relative)
            attempt = {"reach": reach}
            notes["queries"].append(attempt)
            target = n.ik(endpoint, location + error * reach / distance, orientation)
            if target is None:
                attempt["status"] = "ik"
                continue
            target[-2:] = endpoint[-2:]
            if not n.edge(endpoint, target, fingers=True, payload=relative,
                          lifted=True, held_palm=True):
                attempt["status"] = "edge"
                continue
            after = np.array(p.getBasePositionAndOrientation(n.object)[0])
            remaining = float(np.linalg.norm(self.env.goal_pos - after))
            attempt.update(status="within_tolerance" if remaining <= .1 else "outside_tolerance",
                           modeled_distance_after=remaining)
            if remaining <= .1:
                return True
        return False

    def plan_carry(self, current, relative):
        """Original route admission, called only inside state-restoring planning."""
        n, e = self.native, self.env
        hand, xyzw = n.hand_pose()
        orient = rotation(xyzw)
        base_origin = np.array([-2., 0.])
        start = current[:2] + base_origin
        object_pos = np.array(p.getBasePositionAndOrientation(n.object)[0])
        desired_z = hand[2] + .9 - object_pos[2]
        table = np.array(p.getAABB(e.table_id))

        def certify_route(q, terminal_notes=None):
            n.set(q)
            n.position_payload(relative)
            obj = np.array(p.getBasePositionAndOrientation(n.object)[0])
            goal = self.goal_base_xy(obj, start, e.goal_pos)
            if goal is None:
                if terminal_notes is None:
                    return None
                goal = start + e.goal_pos[:2] - obj[:2]
            boxes = np.array([p.getAABB(n.robot, i) for i in n.links]
                             + [p.getAABB(n.object)])
            footprint = boxes[:, :, :2] - start
            maze_blocked = footprint_blocker(e.cube_positions[:, :2], footprint)
            table_footprint = footprint[boxes[:, 0, 2] <= table[1, 2] + .005]

            def blocked(a, b):
                if maze_blocked(a, b):
                    return True
                extent = table_footprint + np.array([np.minimum(a,b), np.maximum(a,b)])
                return bool(np.any(np.all((extent[:, 0, :] <= table[1,:2])
                                           & (table[0,:2] <= extent[:, 1,:]), axis=-1)))

            waypoints = route(start, goal, blocked)
            if waypoints is None:
                return None
            previous = q
            valid = True
            for point in waypoints:
                next_q = q.copy()
                next_q[:2] = point - base_origin
                if not n.edge(previous, next_q, fingers=True, payload=relative, lifted=True, held_palm=True):
                    valid = False
                    break
                previous = next_q
            if not valid:
                return None
            if terminal_notes is not None:
                terminal_notes["route_waypoints"] = len(waypoints)
                if not self.certify_terminal_reach(previous, relative, terminal_notes):
                    return None
            return waypoints, obj

        carries = []
        prefixes = []
        with n.planning():
            with n.planning():
                direct = certify_route(current)
            direct_kind = "direct"
            if direct is None and self.goal_base_xy(object_pos, start, e.goal_pos) is None:
                terminal = {"height_residual": float(abs(object_pos[2] - e.goal_pos[2])),
                            "queries": []}
                self.extra["direct_terminal"] = terminal
                with n.planning():
                    direct = certify_route(current, terminal)
                terminal["certified"] = direct is not None
                direct_kind = "direct_terminal"
            if direct is not None:
                waypoints, obj = direct
                carries.append(((), waypoints, obj, direct_kind))
            else:
                # Retract toward the torso so the loaded silhouette fits 1 m aisles.
                # Fixed yaw preserves hand orientation and minimizes inertial load.
                for xy in ([.42, .0], [.49, 0.], [.46, -.12], [.46, .12], [.55, 0.]):
                    for height in (desired_z, hand[2]):
                        q = current.copy()
                        path = []
                        for fraction in np.linspace(0., 1., 6)[1:]:
                            end = np.r_[start + xy, height]
                            q_next = n.ik(q, hand + fraction * (end - hand), orient, 55)
                            if q_next is None or not n.edge(q, q_next, fingers=True, payload=relative, lifted=True, held_palm=True):
                                break
                            q = q_next
                            path.append(q)
                            if len(path) < 5:
                                prefixes.append(tuple(point.copy() for point in path))
                        if len(path) != 5:
                            continue
                        certified = certify_route(q)
                        if certified is not None:
                            waypoints, obj = certified
                            carries.append((path, waypoints, obj, "full"))
                            break
                    if carries:
                        break
            if not carries:
                fallback = self.choose_carry_prefix(current, start, prefixes, certify_route)
                if fallback is not None:
                    carries.append(fallback)
        return carries[0] if carries else None

    def carry_route(self):
        n, e = self.native, self.env
        current = n.q()
        relative = n.payload_transform()
        with n.planning():
            selected = self.plan_carry(current, relative)
        if selected is None:
            with n.planning():
                selected = heading_carry(self, current, relative)
        if selected is None:
            from .isolated_se2 import query
            notes={};self.extra['isolated_se2']=notes
            path=query(n,self.config,notes)
            if path is not None:
                from .se2_execute import execute
                execute(self,path)
                return
        if selected is None:
            raise StopTrial("no_safe_carry_route")
        path, _, modeled_object, kind = selected
        base_origin = np.array([-2., 0.])
        table = np.array(p.getAABB(e.table_id))
        self.extra["carry_modeled_object"] = modeled_object.tolist()
        self.extra["carry_route_admission"] = kind
        for target in path:
            if kind in ("prefix", "heading_prefix"):
                actual = n.q()
                target = target.copy()
                target[-2:] = actual[-2:]
                relative = n.payload_transform()
                with n.planning():
                    free = n.edge(actual, target, fingers=True, payload=relative,
                                  lifted=True, held_palm=True)
                if kind == "heading_prefix":
                    self.extra["loaded_heading"]["realized_edges"].append(free)
                else:
                    self.extra["carry_prefix"]["realized_edges"].append(free)
                if not free:
                    raise StopTrial("carry_prefix_edge_blocked")
            self.follow(target, "retract")
        # Rebuild the footprint from actual settled fingers/object before driving.
        q = n.q()
        relative = n.payload_transform()
        start = q[:2] + base_origin
        obj = np.array(p.getBasePositionAndOrientation(n.object)[0])
        boxes = np.array([p.getAABB(n.robot, i) for i in n.links] + [p.getAABB(n.object)])
        footprints = boxes[:, :, :2] - start
        maze_blocked = footprint_blocker(e.cube_positions[:, :2], footprints)
        low = footprints[boxes[:, 0, 2] <= table[1, 2] + .005]

        def blocked(a, b):
            if maze_blocked(a, b):
                return True
            extent = low + np.array([np.minimum(a,b), np.maximum(a,b)])
            return bool(np.any(np.all((extent[:, 0, :] <= table[1,:2])
                                       & (table[0,:2] <= extent[:, 1,:]), axis=-1)))

        goal = self.goal_base_xy(obj, start, e.goal_pos)
        if goal is None:
            terminal_allowed = kind == "direct_terminal" or (
                kind == "heading_prefix"
                and self.extra["loaded_heading"].get("selected_original_kind") == "direct_terminal")
            if not terminal_allowed:
                raise StopTrial("measured_carry_height_out_of_goal")
            goal = start + e.goal_pos[:2] - obj[:2]
        waypoints = route(start, goal, blocked)
        if waypoints is None:
            raise StopTrial("no_measured_carry_route")
        self.extra["route_waypoints"] = [point.tolist() for point in waypoints]
        for point in waypoints:
            previous = n.q()
            target = q.copy()
            target[:2] = point - base_origin
            relative = n.payload_transform()
            with n.planning():
                free = n.edge(previous, target, fingers=True, payload=relative, lifted=True, held_palm=True)
            if not free:
                raise StopTrial("carry_edge_blocked")
            self.follow(target, "navigate", continuous=True)
            if self.arrival:
                return
        for _ in range(self.config["waypoint_wait_calls"]):
            self.step(target)
            if self.arrival:
                return
        self.finish_goal()
        if not self.arrival:
            raise StopTrial("goal_not_held_within_tolerance")

    def finish_goal(self):
        n = self.native
        attempts = []
        self.extra["final_goal_reach"] = attempts
        while self.held_call and not self.arrival:
            error = self.env.goal_pos - self.last["object_position"]
            distance = np.linalg.norm(error)
            if distance <= .1:
                self.extra["final_goal_stop"] = "within_tolerance"
                return
            start = n.q()
            location, quaternion = n.hand_pose()
            orientation = rotation(quaternion)
            relative = n.payload_transform()
            requested = min(self.config["goal_reach_step"], distance)
            attempt = {"call": self.env.step_count, "goal_error_before": error.tolist(),
                       "distance_before": float(distance), "queries": []}
            attempts.append(attempt)
            target = None
            with n.planning():
                for reach in (requested, requested / 2):
                    candidate = n.ik(start, location + error * (reach / distance), orientation)
                    if candidate is None:
                        attempt["queries"].append([reach, "ik"])
                        continue
                    candidate[-2:] = start[-2:]
                    if not n.edge(start, candidate, fingers=True, payload=relative,
                                  lifted=True, held_palm=True):
                        attempt["queries"].append([reach, "edge"])
                        continue
                    attempt["queries"].append([reach, "planned"])
                    target = candidate
                    break
            if target is None:
                self.extra["final_goal_stop"] = "unreachable"
                return
            try:
                self.follow(target, "navigate")
            except StopTrial as exc:
                self.extra["final_goal_stop"] = str(exc)
                raise
            after = self.last["goal_distance"]
            attempt["distance_after"] = after
            if not self.arrival and distance - after < .001:
                self.extra["final_goal_stop"] = "measured_no_progress"
                return
        self.extra["final_goal_stop"] = "arrival" if self.arrival else "not_held"

    def settle_for_planning(self, command):
        def speeds():
            linear, angular = p.getBaseVelocity(self.native.object)
            return float(np.linalg.norm(linear)), float(np.linalg.norm(angular))

        linear, angular = speeds()
        extra_wait = linear > self.config["settle_trigger_linear_speed"]
        notes = {"activated": extra_wait, "reference_call": self.env.step_count,
                 "initial_speeds": [linear, angular], "waits": []}
        self.extra["perception_settling"] = notes
        notes["exit_reason"] = "waiting" if extra_wait else "inactive"
        if extra_wait:
            window_calls = self.config["settle_pose_window_calls"]
            observations = deque(maxlen=window_calls)
            canonical = np.asarray(self.env.object_canonical_pc, dtype=float)
            stable = pose_streak = 0
            while True:
                self.step(command)
                linear, angular = speeds()
                below = (linear <= self.config["settle_linear_speed"]
                         and angular <= self.config["settle_angular_speed"])
                stable = stable + 1 if below else 0
                world = canonical @ rotation(self.last["object_xyzw"]).T + self.last["object_position"]
                observations.append(world)
                extent = None
                if len(observations) == window_calls:
                    extent = float(max(np.linalg.norm(points - world, axis=1).max()
                                       for points in observations))
                    pose_streak = pose_streak + 1 if extent <= self.config["settle_pose_tolerance"] else 0
                notes["waits"].append({"call": self.env.step_count, "linear_speed": linear,
                                       "angular_speed": angular, "stable_calls": stable,
                                       "pose_extent_m": extent, "pose_stable_calls": pose_streak})
                if stable >= self.config["settle_stable_calls"]:
                    notes["exit_reason"] = "velocity"
                    break
                if pose_streak >= self.config["settle_pose_stable_calls"]:
                    notes["exit_reason"] = "pose_window"
                    break
        notes["planning_call"] = self.env.step_count
        notes["planning_object_position"] = self.last["object_position"]
        notes["planning_object_xyzw"] = self.last["object_xyzw"]

    def run(self):
        n, e = self.native, self.env
        stop = "held_goal"
        try:
            q = n.q()
            for _ in range(self.config["settle_calls"]):
                self.step(q)
            self.initial_pose = np.array(self.last["object_position"])
            self.initial_bottom = self.last["object_bottom"]
            self.settle_for_planning(q)
            notes = {}
            self.extra["pick_planning"] = notes
            plan = plan_pick(n, e._get_obs(), self.table_height, self.config, notes)
            if plan is None:
                raise StopTrial("no_safe_approach")
            self.target_pose = (plan.target_position, plan.target_rotation)
            q = n.q()
            q[-2:] = plan.staging[-2:]
            self.follow(q, "open")
            with n.planning():
                if not n.edge(n.q(), plan.staging):
                    raise StopTrial("measured_staging_blocked")
            self.follow(plan.staging, "stage_arm")
            self.follow(plan.base, "base_to_table")
            self.follow(plan.pregrasp, "pregrasp")
            contact_q = None
            try:
                for target in plan.approach:
                    self.follow(target, "approach")
            except EarlyContactReady:
                contact_q = n.q().copy()
                self.early_contact_pending = False
                self.extra['early_closure']={'boundary_call':e.step_count,'boundary_tick':e.tick_count,
                                             'body_anchor_q':contact_q.tolist(),
                                             'interrupted_remaining_approach':True}
            hand, orientation = n.hand_pose()
            self.extra["approach_position_error"] = float(np.linalg.norm(hand - plan.target_position))
            self.extra["approach_angle_error"] = float(np.linalg.norm(
                rotation_error(plan.target_rotation, rotation(orientation))))
            target = plan.approach[-1].copy() if contact_q is None else contact_q.copy()
            target[-2:] = 0.
            self.phase = "close"
            consecutive = 0
            for _ in range(self.config["close_calls"]):
                self.step(target)
                consecutive = consecutive + 1 if all(self.last["finger_contacts"].values()) else 0
                if consecutive >= self.config["closure_contact_calls"]:
                    break
            if consecutive < self.config["closure_contact_calls"]:
                raise StopTrial("no_bilateral_closure")
            self.closure_command = target[-2:].copy()
            if contact_q is not None:
                from .isolated_contact_lift import query_lift
                lift_notes={};self.extra['early_contact_lift']=lift_notes
                path=query_lift(n,{'initial_position':self.initial_pose,'initial_bottom':self.initial_bottom},self.config,lift_notes)
                if path is None:
                    raise StopTrial('early_contact_lift_unreachable')
                for target in path:
                    self.follow(target,'lift')
                    if not all(self.last['finger_contacts'].values()):
                        raise StopTrial('lift_contact_lost')
            else:
                relative = n.payload_transform()
                actual = n.q()
                safe = True
                with n.planning():
                    for target in plan.lift:
                        target[-2:] = actual[-2:]
                        if not n.edge(actual, target, fingers=True, payload=relative, held_palm=True):
                            safe = False
                            break
                        actual = target
                if not safe:
                    raise StopTrial("lift_edge_blocked")
                for target in plan.lift:
                    self.follow(target, "lift")
                    if not all(self.last["finger_contacts"].values()):
                        raise StopTrial("lift_contact_lost")
            self.phase = "hold"
            for _ in range(self.config["hold_calls"]):
                self.step(target)
                if self.pick:
                    break
            required_bottom = max(self.table_height, self.initial_bottom) + .1
            if contact_q is None and self.pick is None and self.held_call and self.last["object_bottom"] < required_bottom:
                self.extend_lift(required_bottom)
            if self.pick is None or self.pick["call"] > self.config["pickup_call_limit"]:
                raise StopTrial("strict_pick_not_qualified")
            self.carry_route()
        except StopTrial as exc:
            stop = str(exc)
        return {"termination": stop, "guard": self.guard, "calls": e.step_count,
                "ticks": e.tick_count, "pickup": self.pick, "arrival": self.arrival,
                "settled_position": self.initial_pose.tolist() if self.initial_pose is not None else None,
                "settled_bottom": self.initial_bottom, **self.extra}
