"""Independent both-clock recount for the palm_geometry_v2 scene schema."""
import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.palm_geometry038.geometry_audit import measured_gap, verify_geometry

GEOMETRY_POLICY = {
    "policy_id": "palm_geometry_v2",
    "max_penetration": .0005,
    "front_x_min": -.03502512,
    "normal_x_max": -.707,
    "geometry_normal_x_max_exclusive": 0.,
}


def check_scene_policy(scene):
    policy = scene.get('palm_policy')
    assert policy == GEOMETRY_POLICY and all(type(policy[k]) is type(value)
                                                for k,value in GEOMETRY_POLICY.items()), 'palm_policy_schema'
    assert scene.get('hand_link') == 20 and scene.get('finger_links') == [21, 22]


def open_trace(folder, mode="rb"):
    plain, compressed = folder / "trace.jsonl", folder / "trace.jsonl.gz"
    assert plain.exists() != compressed.exists(), ("ambiguous_or_missing_trace", str(folder))
    return plain.open(mode) if plain.exists() else gzip.open(compressed, mode)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b, strict=True)))


def world_to_hand(vector, quaternion):
    x,y,z,w = quaternion
    scale = 2/(x*x+y*y+z*z+w*w)
    matrix = ((1-scale*(y*y+z*z),scale*(x*y-z*w),scale*(x*z+y*w)),
              (scale*(x*y+z*w),1-scale*(x*x+z*z),scale*(y*z-x*w)),
              (scale*(x*z-y*w),scale*(y*z+x*w),1-scale*(x*x+y*y)))
    return [sum(matrix[j][i]*vector[j] for j in range(3)) for i in range(3)]


def contact_frame(row, hit):
    point = world_to_hand([a-b for a,b in zip(hit['point_on_palm'],row['hand_position'])],row['hand_xyzw'])
    normal = world_to_hand(hit['normal_on_object'],row['hand_xyzw'])
    assert distance(point,hit['point_local'])<1e-8 and distance(normal,hit['normal_local'])<1e-8
    return point, normal


def check_palm(row, scene, geometry):
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
        assert hit['accepted']==expected, ('palm_classification',row['tick'])
        if hit['accepted']:
            assert row['phase'] in ('lift','hold','retract','navigate')
            legacy_incompatible += int(not legacy)
        else:
            assert any(v[0]=='robot_scene' and v[1]==scene['body_ids']['object']
                       and v[2]==scene['hand_link'] for v in row['violations'])
    return (bool(candidates),sum(hit['accepted'] for hit in candidates),legacy_incompatible)


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
            touched, accepted, legacy_incompatible = check_palm(row, scene, geometry)
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
            'first_palm_candidate_tick':first_palm,'accepted_palm_samples':accepted_palm_samples,
            'finger_only_compatible_pick':bool(call_pick and (first_palm is None or first_palm>first_pick['tick'])),
            'finger_only_compatible_delivery':bool(call_delivery and (first_palm is None or first_palm>first_arrival['tick'])),
            'verification_scope':'independent metric/clock/source recount; native guard decisions recorded by pinned producer'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.folder.resolve(),Path(__file__).resolve().parents[3])
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report))
