"""Fixed-yaw carry planning on a quarter-metre lattice."""
from heapq import heappop, heappush
from itertools import product

import numpy as np


def route(start, goal, blocked, spacing=.25):
    """Find an 8-connected XY path; each edge is checked before admission."""
    start = np.asarray(start)
    goal = np.asarray(goal)
    source = tuple(np.rint(start / spacing).astype(int))
    end = tuple(np.rint(goal / spacing).astype(int))
    offsets = [(x,y) for x,y in product((-1,0,1), repeat=2) if x or y]
    leads = [source] + [(source[0]+x, source[1]+y) for x,y in offsets]
    targets = {end} | {(end[0]+x, end[1]+y) for x,y in offsets}
    heap, costs, parents, edges = [], {}, {}, {}

    def xy(node):
        return spacing * np.asarray(node)

    for lead in leads:
        if blocked(start, xy(lead)):
            continue
        cost = np.linalg.norm(start - xy(lead))
        costs[lead], parents[lead] = cost, None
        heappush(heap, (cost + np.linalg.norm(xy(lead) - goal), cost, lead))

    while heap:
        _, cost, node = heappop(heap)
        if cost != costs[node]:
            continue
        point = xy(node)
        if node in targets and not blocked(point, goal):
            result = [goal]
            while node is not None:
                result.append(xy(node))
                node = parents[node]
            return result[::-1]
        for x, y in offsets:
            next_node = node[0] + x, node[1] + y
            next_point = xy(next_node)
            if not (-4.5 <= next_point[0] <= 18.5 and -5.75 <= next_point[1] <= 5.75):
                continue
            next_cost = cost + spacing * np.hypot(x, y)
            if next_cost >= costs.get(next_node, np.inf):
                continue
            key = tuple(sorted((node, next_node)))
            if key not in edges:
                edges[key] = blocked(point, next_point)
            if edges[key]:
                continue
            costs[next_node], parents[next_node] = next_cost, node
            heappush(heap, (next_cost + np.linalg.norm(next_point - goal), next_cost, next_node))
    return None


def footprint_blocker(cubes, rectangles):
    """Conservative translated link/payload AABB broad phase against maze walls."""
    rectangles = np.asarray(rectangles)  # shape (links, 2 bounds, 2 axes)
    centers = np.asarray(cubes)
    wall_lower, wall_upper = centers - .503, centers + .503

    def blocked(start, end):
        swept_lower = rectangles[:, 0, :] + np.minimum(start, end)
        swept_upper = rectangles[:, 1, :] + np.maximum(start, end)
        overlap = ((swept_lower[:, None, :] <= wall_upper)
                   & (wall_lower <= swept_upper[:, None, :]))
        return bool(np.any(np.all(overlap, axis=-1)))

    return blocked
