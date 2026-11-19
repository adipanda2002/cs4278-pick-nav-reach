"""Independent v3 geometry classifier and retained v2 scoring recount."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import zipfile

import numpy as np
import pybullet as p
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.palm_geometry038.geometry_audit import measured_gap, rotation, verify_geometry
from evaluation.palm_geometry038.sibling_axes_v2.recount_palm_geometry038_v2 import (
    GEOMETRY_POLICY as V2_POLICY, contact_frame, digest, distance, open_trace,
)

GEOMETRY_POLICY = {
    **V2_POLICY,
    "policy_id": "palm_geometry_v3",
    "source_face_axis_limit": 64,
    "source_face_axis_order": "alignment_to_own_normal_stable_lexicographic_com",
    "source_face_axis_basis": "signed_scaled_palm_source_triangle_normals",
}


def check_scene_policy(scene):
    policy = scene["palm_policy"]
    assert policy == GEOMETRY_POLICY and all(type(policy[k]) is type(v)
                                             for k, v in GEOMETRY_POLICY.items())
    assert scene["hand_link"] == 20 and scene["finger_links"] == [21, 22]


def world_face_axes(axes, hand_xyzw, com_xyzw):
    # Obtain the exact Bullet rotations for ranking at ties. The support gap
    # below uses the independent quaternion and world-vertex projection.
    hand, com = [np.asarray(p.getMatrixFromQuaternion(q)).reshape(3, 3)
                 for q in (hand_xyzw, com_xyzw)]
    return axes @ (hand @ com).T


def verify_face_bank(folder, root, scene, geometry):
    if geometry is None:
        return np.empty((0, 3))
    palm = json.loads((folder / "palm_geometry.json").read_bytes())["palm"]
    path = root / palm["mesh"]
    assert digest(path.read_bytes()) == scene["scene_assets"][palm["mesh"]]
    mesh = trimesh.load_mesh(path, process=False)
    scaled_vertices = np.asarray(mesh.vertices) * palm["mesh_scale"]
    a, b, c = scaled_vertices[np.asarray(mesh.faces)].transpose(1, 0, 2)
    u, v = (b - a).T, (c - a).T
    cross = np.column_stack((u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]))
    q = palm["collision_origin_com_xyzw"]
    com_from_collision = np.asarray(p.getMatrixFromQuaternion(q)).reshape(3, 3)
    np.testing.assert_allclose(com_from_collision, rotation(q), rtol=0., atol=1e-15)
    com_axes = cross @ com_from_collision.T
    lengths = np.linalg.norm(com_axes, axis=1)
    assert np.isfinite(lengths).all()
    good = lengths > 0.
    unit = com_axes[good] / lengths[good, None]
    bank = np.asarray(sorted({tuple(axis) for axis in unit} | {tuple(-axis) for axis in unit}))
    assert bank.ndim == 2 and bank.shape[1] == 3 and np.isfinite(bank).all()
    np.testing.assert_allclose(np.linalg.norm(bank, axis=1), 1., rtol=0., atol=1e-15)
    return bank


def relative_world_vertices(geometry, row):
    palm, obj, offset, com_xyzw = geometry
    hand = rotation(row["hand_xyzw"])
    palm_center = np.asarray(row["hand_position"]) + hand @ offset - row["object_position"]
    return (palm @ (hand @ rotation(com_xyzw)).T + palm_center,
            obj @ rotation(row["object_xyzw"]).T)


def check_palm(row, scene, geometry, face_bank):
    policy = scene['palm_policy']
    check_scene_policy(scene)
    legacy_incompatible = 0
    assert set(row['finger_contacts']) == {str(i) for i in scene['finger_links']}
    bilateral = all(any(hit[1]>0 and hit[0]<=.001 for hit in hits)
                    for hits in row['finger_contacts'].values())
    candidates = row['palm_candidates']
    frames = [contact_frame(row, hit) for hit in candidates]
    for hit in row['palm_contacts']:
        contact_frame(row, hit)

    def is_front(hit, point):
        return hit['object_link']==-1 and point[0]>=policy['front_x_min']

    eligible = [
        i for i, (hit, (point, normal)) in enumerate(zip(candidates,frames,strict=True))
        if (is_front(hit,point) and normal[0]<policy['geometry_normal_x_max_exclusive']
            and len(hit['normal_on_object'])==3 and all(math.isfinite(x) for x in hit['normal_on_object'])
            and abs(math.sqrt(sum(x*x for x in hit['normal_on_object']))-1.)<=1e-8)]
    held = row['closure_confirmed'] and bilateral
    cache = {}
    world_axes = None
    face_cache = {}
    face_fields = {'face_axis_world', 'face_axis_index', 'face_gap_m'}
    face_count = 0
    support_fields = {'support_gap_m','support_axis_index','support_axis_world'}
    for i, (hit, (point,normal)) in enumerate(zip(candidates,frames,strict=True)):
        legacy = (held and is_front(hit,point) and hit['distance']>=-policy['max_penetration']
                  and normal[0]<=policy['normal_x_max'])
        extra = not legacy and held and i in eligible
        assert set(hit)&support_fields == (support_fields if extra else set()), ('support_certificate_context',row['tick'])
        gap = selected = None
        if extra:
            for axis_index in [i,*[j for j in eligible if j!=i]]:
                if axis_index not in cache:
                    cache[axis_index] = measured_gap(geometry,row,candidates[axis_index]['normal_on_object'])
                candidate_gap = cache[axis_index]
                if axis_index==i and candidate_gap is not None:
                    gap,selected=candidate_gap,axis_index
                if candidate_gap is not None and candidate_gap>=0.:
                    gap,selected=candidate_gap,axis_index
                    break
            expected_axis = candidates[selected]['normal_on_object'] if selected is not None else None
            assert hit['support_axis_index']==selected and hit['support_axis_world']==expected_axis, (
                'support_certificate_axis',row['tick'])
            reported = hit['support_gap_m']
            if gap is None:
                assert reported is None, ('unsupported_certificate',row['tick'])
            else:
                assert reported is not None and abs(gap-reported)<=2e-12, (
                    'support_certificate_gap',row['tick'],gap,reported)
        expected = legacy or (gap is not None and gap>=0.)
        chosen_face = None
        if extra and not expected and len(face_bank):
            if world_axes is None:
                world_axes = world_face_axes(face_bank, row['hand_xyzw'], geometry[3])
                palm_world, object_world = relative_world_vertices(geometry, row)
            align = world_axes @ hit['normal_on_object']
            ranked = sorted(range(len(align)), key=lambda index: (-align[index], index))
            for index in ranked[:policy['source_face_axis_limit']]:
                if index not in face_cache:
                    normal = world_axes[index] / np.linalg.norm(world_axes[index])
                    face_cache[index] = float(np.min(palm_world @ normal) - np.max(object_world @ normal))
                face_gap = face_cache[index]
                if face_gap is not None and face_gap >= 0.:
                    chosen_face = index
                    break
        if chosen_face is None:
            assert not set(hit) & face_fields, ('unexpected_face_witness', row['tick'])
        else:
            assert set(hit) & face_fields == face_fields, ('missing_face_witness', row['tick'])
            assert type(hit['face_axis_index']) is int and hit['face_axis_index'] == chosen_face
            assert distance(hit['face_axis_world'], world_axes[chosen_face]) <= 1e-12
            stored_gap = hit['face_gap_m']
            archived_axis_gap = measured_gap(geometry, row, hit['face_axis_world'])
            assert math.isfinite(stored_gap) and stored_gap >= 0. and archived_axis_gap >= 0.
            assert abs(stored_gap - face_cache[chosen_face]) <= 2e-12
            assert abs(stored_gap - archived_axis_gap) <= 2e-12
            expected = True
            face_count += 1
        assert hit['accepted']==expected, ('palm_classification',row['tick'])
        if hit['accepted']:
            assert row['phase'] in ('lift','hold','retract','navigate')
            legacy_incompatible += int(not legacy)
        else:
            assert any(v[0]=='robot_scene' and v[1]==scene['body_ids']['object']
                       and v[2]==scene['hand_link'] for v in row['violations'])
    return (bool(candidates),sum(hit['accepted'] for hit in candidates),legacy_incompatible,face_count)


def audit(folder, root):
    scene = json.loads((folder/'scene.json').read_text())
    check_scene_policy(scene)
    result = json.loads((folder/'result.json').read_text())
    config = json.loads((folder/'config.json').read_text())
    sources = json.loads((folder/'source.json').read_text())
    assert result['termination'] != 'planning_only'
    with zipfile.ZipFile(folder/'sources.zip') as archive:
        assert set(archive.namelist()) == set(sources)
        for name, expected in sources.items():
            assert digest(archive.read(name)) == expected, ('archived_source_mismatch', name)
    current_sources_match = all((root/name).is_file() and digest((root/name).read_bytes()) == expected
                                for name, expected in sources.items())
    assert digest((folder/'maze.urdf').read_bytes()) == scene['maze_sha256']
    assert scene['substeps'] == 5
    assert scene['actuator_position_gains'] == [.3]*13+[.03]*2
    assert scene['actuator_max_forces'] == [2000000.]*15
    geometry = verify_geometry(folder, root, scene)
    face_bank = verify_face_bank(folder, root, scene, geometry)
    face_samples = 0
    first_face = None
    settled = result['settled_position']
    bottom0 = result['settled_bottom']
    height = result['table_height']
    chain = 0
    first_pick = None
    first_arrival = None
    lost_after_pick = False
    first_guard = None
    first_palm = None
    accepted_palm_samples = 0
    first_legacy_palm_rejection = None
    legacy_incompatible_samples = 0
    count = 0
    previous = None
    raw_hash = hashlib.sha256()
    with open_trace(folder, 'rb') as stream:
        for data in stream:
            raw_hash.update(data)
            row = json.loads(data)
            count += 1
            call = (count-1)//5+1
            substep = (count-1)%5+1
            assert (row['call'],row['tick'],row['substep']) == (call,count,substep)
            assert all(math.isfinite(v) for v in row['q']+row['target']+row['object_position'])
            assert len(row['q']) == len(row['target']) == 15
            if substep > 1:
                assert previous['target'] == row['target']
            if row['violations'] and first_guard is None:
                first_guard = count
            touched, accepted, legacy_incompatible, face_count = check_palm(
                row, scene, geometry, face_bank)
            if face_count and first_face is None:
                first_face = count
            face_samples += face_count
            if touched and first_palm is None:
                first_palm = count
            if legacy_incompatible and first_legacy_palm_rejection is None:
                first_legacy_palm_rejection = count
            accepted_palm_samples += accepted
            legacy_incompatible_samples += legacy_incompatible
            if call == config['settle_calls'] and substep == 5:
                assert row['object_position'] == settled and row['object_bottom'] == bottom0
            if substep == 5:
                contacts = row['finger_contacts']
                bilateral = len(contacts) == 2 and all(any(hit[1]>0 and hit[0]<=.001 for hit in hits) for hits in contacts.values())
                held = bilateral and not row['object_supported']
                eligible = (settled is not None and held and row['object_bottom']-height >= .1
                            and row['object_bottom']-bottom0 >= .1
                            and distance(row['object_position'],settled)>=.1)
                # Settled reference is first established after the settling call.
                if call <= config['settle_calls']:
                    eligible = False
                assert eligible == row['lift_eligible'], ('lift_eligible',call)
                chain = chain+1 if eligible and first_guard is None else 0
                if first_pick is None and chain >= 50:
                    first_pick = {'call':call,'tick':count}
                if first_pick and (not held or first_guard is not None):
                    lost_after_pick = True
                goal_distance = distance(row['object_position'],scene['goal_position'])
                assert abs(goal_distance-row['goal_distance']) < 1e-9
                if (first_pick and row['phase']=='navigate' and held and not lost_after_pick
                        and first_guard is None and goal_distance<=.1 and first_arrival is None):
                    first_arrival = {'call':call,'tick':count}
            previous = row
    assert count == result['ticks'] == 5*result['calls']
    assert first_pick == result['pickup'], ('pickup_mismatch',first_pick,result['pickup'])
    claimed = result['arrival']
    assert (None if claimed is None else {k:claimed[k] for k in ('call','tick')}) == first_arrival
    call_pick = first_pick is not None and first_pick['call']<=3000
    tick_pick = first_pick is not None and first_pick['tick']<=3000
    call_delivery = bool(call_pick and first_arrival and first_arrival['call']-first_pick['call']<=15000)
    both_delivery_tick_windows = bool(call_delivery and first_arrival['tick']<=15000 and first_arrival['tick']-first_pick['tick']<=15000)
    return {'pair':[scene['object_index'],scene['seed']],'status':'PASS','termination':result['termination'],
            'calls':result['calls'],'ticks':count,'pickup':first_pick,'arrival':first_arrival,
            'call_strict_pick':call_pick,'call_held_delivery':call_delivery,
            'tick_strict_pick':tick_pick,'both_tick_delivery':both_delivery_tick_windows,
            'both_timed_stages':bool(tick_pick and both_delivery_tick_windows),
            'whole_call_delivery_window':bool(call_delivery and first_arrival['call']<=15000),
            'first_guard_tick':first_guard,'trace_sha256':raw_hash.hexdigest(),
            'archived_sources_verified':True,'current_sources_match':current_sources_match,
             'palm_classification_checked':True,
            'palm_geometry_provenance_verified':geometry is not None,'palm_policy_id':GEOMETRY_POLICY['policy_id'],
            'first_legacy45_incompatible_palm_tick':first_legacy_palm_rejection,
            'legacy45_incompatible_palm_samples':legacy_incompatible_samples,
            'legacy45_compatible_pick':bool(call_pick and (first_legacy_palm_rejection is None or
                                                              first_legacy_palm_rejection>first_pick['tick'])),
            'legacy45_compatible_delivery':bool(call_delivery and (first_legacy_palm_rejection is None or
                                                                      first_legacy_palm_rejection>first_arrival['tick'])),
            'first_source_face_certificate_tick':first_face,
            'source_face_certificate_samples':face_samples,'source_face_axis_bank_size':len(face_bank),
            'first_palm_candidate_tick':first_palm,'accepted_palm_samples':accepted_palm_samples,
            'finger_only_compatible_pick':bool(call_pick and (first_palm is None or first_palm>first_pick['tick'])),
            'finger_only_compatible_delivery':bool(call_delivery and (first_palm is None or first_palm>first_arrival['tick'])),
            'verification_scope':'independent metric/clock/source recount; native guard decisions recorded by pinned producer'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.folder.resolve(),Path(__file__).resolve().parents[2])
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report))
