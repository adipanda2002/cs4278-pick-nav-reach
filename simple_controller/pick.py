"""Build one checked approach from gripper poses and joint-space segments."""
from dataclasses import dataclass
from itertools import product

import numpy as np

from .geometry import grasps, top_insertion_depths


@dataclass
class PickPlan:
    staging: np.ndarray
    base: np.ndarray
    pregrasp: np.ndarray
    approach: list
    target_position: np.ndarray
    target_rotation: np.ndarray
    lift: list
    family: str


def _central_low_top_order(candidates, cloud_median_xy):
    """Promote three existing candidates by world-XY distance in metres."""
    original = [
        (rank, grasp, float(np.linalg.norm(grasp.center[:2] - cloud_median_xy)))
        for rank, grasp in enumerate(candidates, start=1)
    ]
    promoted = sorted(original, key=lambda row: (row[2], row[0]))[:3]
    promoted_ranks = {rank for rank, _, _ in promoted}
    return promoted + [row for row in original if row[0] not in promoted_ranks]


def _legacy_plan_pick(native, observation, table_height, config, notes):
    start = native.q()
    base_origin = np.array([-2., 0.])
    options = grasps(observation["object_pc"], observation["object_normals"],
                     table_height, start[:2] + base_origin, config["grasp_limit"])
    notes["proposals"] = {family: sum(option.family == family for option in options)
                          for family in ("top", "side", "rim")}
    low_top = grasps(observation["object_pc"], observation["object_normals"], table_height,
                     start[:2] + base_origin, config["low_top_limit"],
                     low_top_open=config["low_top_open_finger_position"])
    notes["low_top_proposals"] = len(low_top)
    notes["checks"] = {}
    seeds = (
        [.16, 0., -.68, 0., 1.8, 0., -1.12, 0.],
        [.16, .3, -.4, -1.48, 1.5, 0., -.5, 1.2],
        [.24, -.35, -.45, 1.48, 1.5, 0., -.5, -1.2],
    )
    with native.planning():
        # Preserve every legacy approach before searching bounded, lower top contacts.
        for support_aware, bank in ((False, options), (True, low_top)):
            candidates = [(rank, grasp, None) for rank, grasp in enumerate(bank, start=1)]
            if not candidates:
                continue
            if support_aware:
                notes["legacy_checks"] = notes["checks"]
                notes["checks"] = {}
                median_xy = np.median(observation["object_pc"], axis=0)[:2]
                candidates = _central_low_top_order(bank, median_xy)
                notes["low_top_order"] = {
                    "basis": "world_xy_distance_to_observed_cloud_coordinate_median",
                    "cloud_median_world_xy_m": median_xy.tolist(),
                    "original_ranks": [rank for rank, _, _ in candidates],
                }
            opening = config["low_top_open_finger_position" if support_aware else "open_finger_position"]
            open_q = start.copy()
            open_q[-2:] = opening
            staging = open_q.copy()
            staging[native.arm] = seeds[0]
            if (support_aware and not native.edge(start, open_q)) or not native.edge(open_q, staging):
                notes["checks"]["staging"] = 1
                continue
            for stand_off, offset_y in ((.93, 0.), (.98, -.22), (.98, .22), (.82, 0.)):
                base = staging.copy()
                base[:2] = observation["object_pos"][:2] + [-stand_off, offset_y] - base_origin
                base[2] = np.arctan2(-offset_y, stand_off)
                if not native.edge(staging, base):
                    notes["checks"]["base"] = notes["checks"].get("base", 0) + 1
                    continue
                for (ordered_rank, (original_rank, grasp, offset)), flip in product(
                    enumerate(candidates, start=1), (1., -1.)
                ):
                    orientation = grasp.rotation @ np.diag([1., flip, flip])
                    direction = orientation[:, 0]
                    depths = (top_insertion_depths(grasp.center, orientation, table_height,
                                                   opening)
                              if support_aware else (.012,))
                    for insertion_depth in depths:
                        target = grasp.center - insertion_depth * direction
                        pregrasp = target - .083 * direction
                        for seed in seeds:
                            q_seed = base.copy()
                            q_seed[native.arm] = seed
                            q_pre = native.ik(q_seed, pregrasp, orientation, config["ik_iterations"])
                            if q_pre is None:
                                continue
                            notes["checks"]["ik"] = notes["checks"].get("ik", 0) + 1
                            if not native.edge(base, q_pre):
                                notes["checks"]["pregrasp_collision"] = notes["checks"].get("pregrasp_collision", 0) + 1
                                continue
                            approach = [q_pre]
                            for distance in (.055, .028, 0.):
                                q = native.ik(approach[-1], target - distance * direction, orientation, 45)
                                if q is None or not native.edge(approach[-1], q):
                                    break
                                approach.append(q)
                            if len(approach) != 4:
                                notes["checks"]["approach"] = notes["checks"].get("approach", 0) + 1
                                continue
                            lift = [approach[-1]]
                            for distance in (.035, .075, .118, .149):
                                q = native.ik(lift[-1], target + [0., 0., distance], orientation, 50)
                                if q is None:
                                    break
                                lift.append(q)
                            if len(lift) == 5:
                                notes["selected"] = {"family": grasp.family, "width": grasp.width,
                                                     "score": grasp.score, "center": grasp.center.tolist()}
                                if support_aware:
                                    notes["selected"].update(support_aware=True, insertion_depth=insertion_depth,
                                                             open_finger_position=opening,
                                                             pad_tip_reserve=.0315 - insertion_depth,
                                                             original_bank_rank=original_rank,
                                                             ordered_bank_rank=ordered_rank,
                                                             cloud_median_offset_xy_m=offset)
                                return PickPlan(staging, base, q_pre, approach[1:], target, orientation, lift[1:], grasp.family)
                            notes["checks"]["lift_ik"] = notes["checks"].get("lift_ik", 0) + 1
    return None


def plan_pick(native, observation, table_height, config, notes):
    original = _legacy_plan_pick(native, observation, table_height, config, notes)
    if original is None:
        return None
    from .isolated_chord import improve_contact
    review = {}
    selected = improve_contact(native, observation, original, config, review)
    notes["native_chord"] = review
    return selected
