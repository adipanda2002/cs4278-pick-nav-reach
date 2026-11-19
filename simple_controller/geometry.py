"""World-space antipodal proposals for the Fetch parallel gripper."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grasp:
    center: np.ndarray
    rotation: np.ndarray
    width: float
    family: str
    score: float
    preferred_depth: float | None = None


def top_insertion_depths(center, orientation, table_height, open_position):
    """Preferred and near-boundary axial depths using the open pad projection."""
    down = -orientation[2, 0]
    if down <= 0.:
        return ()
    # The opposing pads span these half extents around the hand origin.
    z_reach = np.abs(orientation[2]) @ [.0315, open_position + .013425, .012]
    depth_at_plane = (table_height - center[2] + z_reach) / down
    preferred = max(.012, depth_at_plane + .002 / down)
    tip = np.nextafter(.0315, 0.)
    if depth_at_plane > tip:
        return ()
    fallback = max(.012, min(tip, depth_at_plane + .0002 / down))
    if preferred <= .0275 and preferred - fallback >= .0005:
        return (preferred, fallback)
    if preferred <= tip:
        return (preferred,) if preferred == fallback else (fallback, preferred)
    return (fallback,)


def grasps(points, normals, table_height, base_xy, limit=96, *, low_top_open=None):
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    # Limit dense pairs to contacts on opposing outward-facing surface patches.
    a, b = np.triu_indices(len(points), 1)
    axis = points[a] - points[b]
    width = np.linalg.norm(axis, axis=1)
    valid = (width >= .005) & (width <= .094)
    a, b, axis, width = a[valid], b[valid], axis[valid], width[valid]
    axis /= width[:, None]
    alignment = np.minimum(np.einsum("ij,ij->i", normals[a], axis),
                           -np.einsum("ij,ij->i", normals[b], axis))
    valid = alignment >= .72
    up = np.array([0., 0., 1.])
    cloud_center = np.median(points, axis=0)
    proposals = {"top": [], "side": [], "rim": []}
    for first, second, closing, opening, quality in zip(
            a[valid], b[valid], axis[valid], width[valid], alignment[valid]):
        center = (points[first] + points[second]) / 2
        toward = np.r_[center[:2] - base_xy, 0.]
        toward /= np.linalg.norm(toward) + 1e-12
        families = (("top", -up),) if low_top_open is not None else (("top", -up), ("side", toward))
        for family, direction in families:
            approach = direction - np.dot(direction, closing) * closing
            norm = np.linalg.norm(approach)
            if norm < .88:
                continue
            approach /= norm
            if family == "side" and abs(closing[2]) > .65:
                family = "rim"
            if family == "top":
                low = center[2] - table_height < .038
                if low != (low_top_open is not None):
                    continue
            if family == "side" and center[2] - table_height < .019:
                continue
            if family == "rim" and center[2] - table_height < .006:
                continue
            # Keep +Y orientation consistent per family. Wrist-flipped variants
            # are generated during IK without changing the chosen contacts.
            normal_axis = np.cross(approach, closing)
            if (normal_axis[2] < 0 if family != "top" else closing[1] < 0):
                closing = -closing
                normal_axis = -normal_axis
            rotation = np.column_stack((approach, closing, normal_axis))
            preferred_depth = None
            if low_top_open is not None:
                depths = top_insertion_depths(center, rotation, table_height, low_top_open)
                if not depths:
                    continue
                preferred_depth = depths[0]
            score = (quality - 2.0 * np.linalg.norm(center - cloud_center)
                     + .65 * opening + .28 * (center[2] - table_height))
            proposals[family].append(Grasp(center, rotation, float(opening), family, float(score),
                                           preferred_depth))

    if low_top_open is None:
        rank_key = lambda item: -item.score
    else:
        # A smaller contact X leaves more of the finger pad past the contacts.
        # Apply the preference before de-duplication and the candidate cap.
        rank_key = lambda item: (item.preferred_depth, -item.score)
    distinct = {}
    for family, options in proposals.items():
        distinct[family] = []
        for proposal in sorted(options, key=rank_key):
            if any(np.linalg.norm(proposal.center - prior.center) < .015
                   and abs(np.dot(proposal.rotation[:, 1], prior.rotation[:, 1])) > .93
                   for prior in distinct[family]):
                continue
            distinct[family].append(proposal)
            if len(distinct[family]) == limit:
                break
    result = []
    for rank in range(limit):
        for family in ("top", "side", "rim"):
            if rank < len(distinct[family]):
                result.append(distinct[family][rank])
            if len(result) == limit:
                return result
    return result
