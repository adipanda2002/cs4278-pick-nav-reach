"""Bounded loaded-heading fallback after original carry admission fails.
World coordinates/metres; Bullet XYZW; q base slots0,1 worldXY−[-2,0],
slot2 yaw radians. Slots3: (including measured fingers) fixed during turns.
All native setters are invoked by the caller inside Native.planning().
"""
from copy import deepcopy
import numpy as np
import pybullet as p
from .native import rotation


def quarter_turn_order(hand_xy, base_xy, goal_xy):
    """Try the goal-side radial turn first; both signed90° headings bounded."""
    radial = np.asarray(hand_xy) - base_xy
    goal = np.asarray(goal_xy) - base_xy
    cross = radial[0] * goal[1] - radial[1] * goal[0]
    first = -1 if cross < 0 else 1
    return first, -first


def heading_carry(controller, current, relative):
    """Return a completely checked height/yaw/original-plan chain or None.

    The footprint's two world-axis principal extents identify the dimension
    exchange. The unchanged fixed-heading planner decides whether that exchange
    actually opens a route; no object identity or assumed corridor width enters.
    """
    n, e = controller.native, controller.env
    planner_keys = ("carry_prefix", "direct_terminal")
    original_notes = {key: deepcopy(controller.extra[key]) for key in planner_keys if key in controller.extra}

    def restore_original_notes():
        for key in planner_keys:
            if key in original_notes:
                controller.extra[key] = deepcopy(original_notes[key])
            else:
                controller.extra.pop(key, None)

    notes = {"trigger": "original_no_safe_carry_route", "attempts": [],
             "realized_edges": [], "original_planner_notes": original_notes}
    controller.extra["loaded_heading"] = notes
    n.set(current)
    n.position_payload(relative)
    hand, quat = n.hand_pose()
    obj = np.asarray(p.getBasePositionAndOrientation(n.object)[0])
    dz = float(e.goal_pos[2] - obj[2])
    notes["world_z_correction"] = dz
    raised = n.ik(current, hand + np.array([0., 0., dz]), rotation(quat))
    if raised is None:
        notes["status"] = "height_ik"
        return None
    raised[-2:] = current[-2:]
    if not n.edge(current, raised, fingers=True, payload=relative,
                  lifted=True, held_palm=True):
        notes["status"] = "height_edge"
        return None
    n.set(raised)
    n.position_payload(relative)
    boxes = np.asarray([p.getAABB(n.robot, i) for i in n.links] + [p.getAABB(n.object)])
    extent = boxes[:, 1, :2].max(axis=0) - boxes[:, 0, :2].min(axis=0)
    notes["footprint_extent_xy"] = extent.tolist()
    notes["major_world_axis"] = int(np.argmax(extent))
    base = raised[:2] + np.array([-2., 0.])
    order = quarter_turn_order(n.hand_pose()[0][:2], base, e.goal_pos[:2])
    for sign in order:
        attempt = {"yaw_delta": sign * np.pi / 2}
        notes["attempts"].append(attempt)
        with n.planning():
            n.set(raised)
            n.position_payload(relative)
            turned = raised.copy()
            turned[2] += attempt["yaw_delta"]
            if not n.edge(raised, turned, fingers=True, payload=relative,
                          lifted=True, held_palm=True):
                attempt["status"] = "turn_edge"
                continue
            n.set(turned)
            n.position_payload(relative)
            for key in planner_keys:
                controller.extra.pop(key, None)
            selected = controller.plan_carry(turned, relative)
            attempt["planner_notes"] = {key: deepcopy(controller.extra[key]) for key in planner_keys if key in controller.extra}
            if selected is None:
                attempt["status"] = "no_route"
                restore_original_notes()
                continue
            retract, waypoints, modeled_object, kind = selected
            path = [raised.copy(), turned.copy(), *[point.copy() for point in retract]]
            previous = current
            valid = True
            for target in path:
                if not n.edge(previous, target, fingers=True, payload=relative,
                              lifted=True, held_palm=True):
                    valid = False
                    break
                previous = target
            if valid:
                endpoint = previous.copy()
                for xy in waypoints:
                    target = endpoint.copy()
                    target[:2] = np.asarray(xy) - np.array([-2., 0.])
                    if not n.edge(previous, target, fingers=True, payload=relative,
                                  lifted=True, held_palm=True):
                        valid = False
                        break
                    previous = target
            terminal = {"queries": []}
            if valid:
                valid = controller.certify_terminal_reach(previous, relative, terminal)
            attempt.update(status="certified" if valid else "complete_chain_blocked",
                           original_kind=kind, terminal=terminal)
            if valid:
                notes.update(status="certified", prefix_segments=len(path),
                             route_waypoints=len(waypoints), selected_original_kind=kind)
                return path, waypoints, modeled_object, "heading_prefix"
            restore_original_notes()
    notes["status"] = "no_certificate"
    return None
