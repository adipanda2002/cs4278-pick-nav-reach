"""Run new contact queries in a zero-step child, never perturb live Bullet.
Live path uses getters/file reads only. State uses world SI / Bullet XYZW;
robot joint indices are native indices, qvel units metres/s or radians/s.
Seed/model index only reconstruct the identical supplied scene, never select
policy parameters or actions. Constructor output must match actual maze/assets.
"""
import argparse
import base64
import hashlib
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from time import monotonic
import numpy as np
import pybullet as p

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def encode(value):
    if isinstance(value,np.ndarray):return {"__array__":value.tolist(),"dtype":str(value.dtype)}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [encode(x) for x in value]
    return value


def decode(value):
    if isinstance(value,dict):
        if set(value)=={"__array__","dtype"}:return np.asarray(value["__array__"],dtype=value["dtype"])
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(x) for x in value]
    return value


def plan_record(plan):
    return {key:encode(getattr(plan,key)) for key in plan.__dataclass_fields__}


def restore_plan(record,plan_type):
    return plan_type(**{key:decode(value) for key,value in record.items()})


def digest(data):return hashlib.sha256(data).hexdigest()


def asset_manifest(env):
    import pybullet_data
    assets={path for path in (ROOT/'assets/fetch').rglob('*') if path.is_file()}
    assets.update(path for path in env.object_asset.parent.iterdir() if path.is_file())
    data=Path(pybullet_data.getDataPath())
    assets.update(data/name for name in ('plane.urdf','plane100.obj','table/table.urdf','table/table.obj','table/table.mtl','table/table.png'))
    return {str(path.resolve()):digest(path.read_bytes()) for path in sorted(assets)}


def snapshot(native,observation,original,config):
    """Getters only: no state save/restore, FK reset, ray or collision query."""
    env=native.env
    labels={name:getattr(env,attr) for name,attr in (('plane','plane_id'),('robot','robot_id'),('table','table_id'),('object','object_id'),('maze','maze_id'),('goal','goal_marker_id'))}
    assert p.getNumBodies()==len(labels)
    bodies={}
    for label,body in labels.items():
        count=p.getNumJoints(body)
        bodies[label]={"id":body,"pose":p.getBasePositionAndOrientation(body),"velocity":p.getBaseVelocity(body),
            "joints":[list(p.getJointState(body,i)[:2]) for i in range(count)],
            "dynamics":[p.getDynamicsInfo(body,i) for i in range(-1,count)]}
    assets=asset_manifest(env);assert len(assets)==94
    maze=env.maze_path.read_bytes()
    return encode({"seed":env.seed,"object_idx":env.object_idx,"object_asset":str(env.object_asset.resolve()),
        "maze_base64":base64.b64encode(maze).decode('ascii'),"maze_sha256":digest(maze),"assets":assets,
        "bodies":bodies,"physics_parameters":p.getPhysicsEngineParameters(),"joint_indices":env.joint_indices,
        "joint_lower":env.joint_lower,"joint_upper":env.joint_upper,"goal_pos":env.goal_pos,
        "cube_positions":env.cube_positions,"canonical_pc":env.object_canonical_pc,"canonical_normals":env.object_canonical_normals,
        "observation":observation,"original_plan":None if original is None else plan_record(original),"config":config,
        "producer_sha256":{name:digest((ROOT/name).read_bytes()) for name in
            ('pick_nav_reach_env.py','simple_controller/native_chord.py','simple_controller/isolated_chord.py','simple_controller/native.py','simple_controller/controller.py')}})


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value

def query(native,config,notes):
    """After both originalcarryplannersNone; getters/files only in live world."""
    started=monotonic();request=snapshot(native,native.env._get_obs(),None,config)
    request['producer_sha256']={str(path.relative_to(ROOT)):digest(path.read_bytes()) for path in (ROOT/'simple_controller').glob('*.py')}
    request['producer_sha256'].update({name:digest((ROOT/name).read_bytes()) for name in ('run_simple.py','pick_nav_reach_env.py','maze_utils.py','utils.py','keyboard_control.py','simple_controller/config.json')})
    payload=json.dumps(request,separators=(',',':'),allow_nan=False).encode();notes.update(status='isolated_se2_review',backend='zero_step_child',snapshot_sha256=digest(payload),child_timeout_seconds=195)
    with tempfile.TemporaryDirectory(prefix='pnr-se2-') as directory:
        folder=Path(directory);input_file=folder/'request.json';output_file=folder/'reply.json';input_file.write_bytes(payload);proc=None
        try:
            with (folder/'child.log').open('wb') as log:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--request',str(input_file),'--output',str(output_file)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                code=proc.wait(timeout=195)
            if code!=0 or not output_file.is_file():notes.update(status='child_error',child_exit_code=code);return None
            reply=decode(json.loads(output_file.read_bytes()))
            if reply['status']!='PASS' or reply['physics_steps']!=0 or reply['snapshot_sha256']!=digest(payload):notes['status']='child_integrity_error';return None
            notes.update(plain(reply['notes']));notes['child_state']=reply['state_verified']
            if notes['status']!='selected':return None
            path=reply['data']['q15_chain']
            assert len(path)>=2 and all(q.shape==(15,) and np.isfinite(q).all() for q in path)
            return path
        except subprocess.TimeoutExpired:notes['status']='child_timeout';return None
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            archive=Path(tempfile.mkdtemp(prefix='isolated_se2_',dir=native.env.maze_path.parent))
            with gzip.open(archive/'request.json.gz','wb') as stream:stream.write(payload)
            for name in ('reply.json','child.log','completed_path.json'):
                if (folder/name).exists():(archive/name).write_bytes((folder/name).read_bytes())
            notes['isolated_query_artifact']=archive.name;notes['isolated_elapsed_seconds']=monotonic()-started


def align_height(n,start,relative,notes):
    """Private native checked prefix. World metres/XYZW; q[3] native joint5.
    0.09m is an action target reserve, never a changed goal tolerance.
    """
    from simple_controller.native import rotation
    n.set(start);n.position_payload(relative)
    obj=np.asarray(p.getBasePositionAndOrientation(n.object)[0]);dz=float(n.env.goal_pos[2]-obj[2])
    record={'initial_height_error_m':dz,'status':'not_needed'};notes['height_prefix']=record
    if abs(dz)<=.1:return start.copy()
    index=n.env.joint_indices[3];info=p.getJointInfo(n.robot,index)
    record.update(native_joint_index=index,joint_type=int(info[2]),joint_axis=list(info[13]),lower_m=float(n.lower[3]),upper_m=float(n.upper[3]))
    if index!=5 or info[2]!=p.JOINT_PRISMATIC:record['status']='unexpected_torso_contract';return None
    hand,quat=n.hand_pose();R=rotation(quat)
    # Actual FK, not just URDF local-axis inference; avoid true-limit violation.
    increment=.001 if start[3]+.001<=n.upper[3] else -.001
    probe=start.copy();probe[3]+=increment
    if probe[3]<n.lower[3]:record['status']='fk_probe_out_of_bounds';return None
    n.set(probe);position,orientation=n.hand_pose();translation=(position-hand)/increment;angular=float(np.max(abs(rotation(orientation)-R)))
    record.update(fk_translation_per_m=translation.tolist(),fk_rotation_matrix_error=angular)
    n.set(start);n.position_payload(relative)
    if not np.allclose(translation,[0.,0.,1.],atol=2e-4,rtol=0.) or angular>1e-6:record['status']='unexpected_torso_FK';return None
    shift=dz-np.copysign(.09,dz);candidate=start.copy();candidate[3]+=shift
    record['torso_target_m']=float(candidate[3]);record['requested_shift_m']=shift
    if n.lower[3]<=candidate[3]<=n.upper[3]:
        free=n.edge(start,candidate,fingers=True,payload=relative,lifted=True,held_palm=True)
        record['torso_edge_clear']=bool(free)
        if free:
            n.set(candidate);n.position_payload(relative);error=float(n.env.goal_pos[2]-p.getBasePositionAndOrientation(n.object)[0][2]);record['torso_height_error_m']=error
            if abs(error)<=.1:record['status']='torso_selected';return candidate.copy()
    else:record['torso_bounds_rejected']=True
    n.set(start);n.position_payload(relative)
    candidate=n.ik(start,hand+np.array([0.,0.,dz]),R)
    record['ik_found']=candidate is not None
    if candidate is None:record['status']='height_ik';return None
    candidate[-2:]=start[-2:]
    if not n.edge(start,candidate,fingers=True,payload=relative,lifted=True,held_palm=True):record['status']='height_ik_edge';return None
    n.set(candidate);n.position_payload(relative);error=float(n.env.goal_pos[2]-p.getBasePositionAndOrientation(n.object)[0][2]);record['ik_height_error_m']=error
    if abs(error)>.1:record['status']='height_ik_residual';return None
    record['status']='ik_selected';return candidate.copy()


def plan_private(n,notes,checkpoint):
    """Zero-step search, finiteparity proof, independentfullNativechain."""
    from simple_controller.native import rotation
    from simple_controller.se2_planner import Planner,Bound,samples
    start=n.q();relative=n.payload_transform();n.position_payload(relative)
    bad=n.violations(start,fingers=True,lifted=True,held_palm=True)
    notes.update(status='review',initial_fullNative_violations=bad,parity=[])
    if bad:notes['status']='initial_native_blocked';return None
    object_pos=np.asarray(p.getBasePositionAndOrientation(n.object)[0])
    original_start=start.copy()
    aligned=align_height(n,start,relative,notes)
    if aligned is None:notes['status']='height_alignment_failed';return None
    start=aligned.copy();n.set(start);n.position_payload(relative)
    prefix_changed=not np.array_equal(original_start,start)
    planner=Planner(n,start,relative,n.env.goal_pos)
    for index,q in enumerate([start,*planner.goals]):
        n.set(q);n.position_payload(relative);violations=n.violations(q,fingers=True,lifted=True,held_palm=True);scene_free=planner.scene(q,usecache=False)
        notes['parity'].append({'index':index,'scene_only_clear':scene_free,'fullNative_clear':not violations,'native_violations':violations})
        if scene_free!=(not violations):notes['status']='scene_native_parity_mismatch';return None
    try:path,reason=planner.search();notes['search_reason']=reason
    except Bound as error:path=planner.completed;notes.update(search_reason=str(error),search_budget_exit=True)
    notes['search']=planner.record()
    if path is None:notes['status']='unresolved';return None
    route_start_index=1 if prefix_changed else 0
    if prefix_changed:path=[original_start.copy(),*[q.copy() for q in path]]
    data={'q15_chain':[q.copy() for q in path],'relative_payload':relative,'goal':n.env.goal_pos};checkpoint.write_text(json.dumps(encode(data),indent=2,allow_nan=False)+'\n')
    certificate={'status':'review','height_prefix_edges':route_start_index,'edges':[],'sample_count':sum(len(samples(a,b,n.wrap)) for a,b in zip(path,path[1:]))};began=monotonic()
    with n.planning():
        for index,(a,b) in enumerate(zip(path,path[1:])):
            if monotonic()-began>90:certificate['status']='UNKNOWN_VERIFY90s';break
            if index>=route_start_index and (not np.array_equal(a[3:],start[3:]) or not np.array_equal(b[3:],start[3:])):raise AssertionError('SE2fixedposturecontract')
            free=n.edge(a,b,fingers=True,payload=relative,lifted=True,held_palm=True);certificate['edges'].append({'index':index,'free':free})
            if not free:certificate.update(status='FAIL_FULL_NATIVE',failed_q=n.q(),violations=n.violations(n.q(),fingers=True,lifted=True,held_palm=True));break
        else:
            n.set(path[-1]);n.position_payload(relative);com=np.asarray(p.getBasePositionAndOrientation(n.object)[0]);distance=float(np.linalg.norm(com-n.env.goal_pos));certificate.update(status='PASS' if distance<=.1 else 'FAIL_GOAL',goal_distance_m=distance,goal_COM_world_m=com)
    certificate['wall_seconds']=monotonic()-began;notes['certificate']=certificate;notes['status']='selected' if certificate['status']=='PASS' else 'certificate_failed'
    return data if notes['status']=='selected' else None


def child(request_file,output_file):
    """Full independent native world; no stepping, including constructor."""
    from pick_nav_reach_env import PickNavReachEnv
    from simple_controller.native import Native,rotation
    from simple_controller.se2_planner import Planner
    from simple_controller.pick import PickPlan
    payload=request_file.read_bytes();request=decode(json.loads(payload));traps=0
    def trap(*args,**kwargs):
        nonlocal traps
        traps+=1;raise AssertionError('Zero-step native child')
    p.stepSimulation=trap;PickNavReachEnv.step=trap
    for name,pin in request['producer_sha256'].items():assert digest((ROOT/name).read_bytes())==pin,name
    for name,pin in request['assets'].items():assert digest(Path(name).read_bytes())==pin,name
    maze_file=output_file.parent/'child_maze.urdf';maze_file.write_bytes(base64.b64decode(request['maze_base64']))
    env=PickNavReachEnv(request['seed'],request['object_idx'],headless=True,maze_path=maze_file)
    try:
        assert env.step_count==env.tick_count==traps==0
        assert digest(env.maze_path.read_bytes())==request['maze_sha256']
        assert str(env.object_asset.resolve())==request['object_asset'] and asset_manifest(env)==request['assets']
        assert encode(p.getPhysicsEngineParameters())==encode(request['physics_parameters'])
        assert env.joint_indices==request['joint_indices'];np.testing.assert_array_equal(env.joint_lower,request['joint_lower']);np.testing.assert_array_equal(env.joint_upper,request['joint_upper'])
        for name in ('goal_pos','cube_positions','canonical_pc','canonical_normals'):
            attr={'canonical_pc':'object_canonical_pc','canonical_normals':'object_canonical_normals'}.get(name,name)
            np.testing.assert_array_equal(getattr(env,attr),request[name])
        labels={name:getattr(env,attr) for name,attr in (('plane','plane_id'),('robot','robot_id'),('table','table_id'),('object','object_id'),('maze','maze_id'),('goal','goal_marker_id'))}
        assert p.getNumBodies()==len(labels)
        max_rotation_readback=0.;visual_only_inertial_quaternions=[]
        for label,body in labels.items():
            saved=request['bodies'][label];assert body==saved['id'] and p.getNumJoints(body)==len(saved['joints'])
            actual_dynamics=encode([p.getDynamicsInfo(body,i) for i in range(-1,p.getNumJoints(body))])
            expected_dynamics=encode(saved['dynamics'])
            visual_only=False
            if label=='goal' and p.getNumJoints(body)==0 and actual_dynamics[0][0]==expected_dynamics[0][0]==0.:
                try:shapes=p.getCollisionShapeData(body,-1)
                except p.error:
                    # This API reports no collision info as an error for the
                    # supplied maximal-coordinate visual marker (shape=-1).
                    assert body==env.goal_marker_id and p.getVisualShapeData(body)
                    shapes=()
                visual_only=not shapes
            if visual_only:
                # Bullet's visual-only maximal-coordinate marker reports an
                # uninitialized inertial quaternion. It has no physical use.
                visual_only_inertial_quaternions.append(label)
                assert all(a==b for i,(a,b) in enumerate(zip(actual_dynamics[0],expected_dynamics[0])) if i!=4)
            else:
                assert actual_dynamics==expected_dynamics,(label,[(i,a,b) for i,(a,b) in enumerate(zip(actual_dynamics,expected_dynamics)) if a!=b])
            p.resetBasePositionAndOrientation(body,*saved['pose']);p.resetBaseVelocity(body,*saved['velocity'])
            for i,(q,qdot) in enumerate(saved['joints']):p.resetJointState(body,i,q,qdot)
            now=p.getBasePositionAndOrientation(body);np.testing.assert_array_equal(now[0],saved['pose'][0])
            error=float(np.max(np.abs(rotation(now[1])-rotation(saved['pose'][1]))));max_rotation_readback=max(max_rotation_readback,error);assert error<=1e-12
            np.testing.assert_array_equal(p.getBaseVelocity(body),saved['velocity'])
            np.testing.assert_array_equal([p.getJointState(body,i)[:2] for i in range(p.getNumJoints(body))],saved['joints'])
        n=Native(env);notes={}
        with n.planning():data=plan_private(n,notes,output_file.parent/'completed_path.json')
        assert env.step_count==env.tick_count==traps==0
        for label,body in labels.items():
            saved=request['bodies'][label];now=p.getBasePositionAndOrientation(body)
            np.testing.assert_array_equal(now[0],saved['pose'][0]);np.testing.assert_allclose(rotation(now[1]),rotation(saved['pose'][1]),atol=1e-12,rtol=0.)
            np.testing.assert_array_equal(p.getBaseVelocity(body),saved['velocity'])
            np.testing.assert_array_equal([p.getJointState(body,i)[:2] for i in range(p.getNumJoints(body))],saved['joints'])
        reply={'status':'PASS','snapshot_sha256':digest(payload),'notes':notes,'data':data,'physics_steps':0,
            'state_verified':{'body_count':len(labels),'all_body_poses_velocities_joint_q_qvel':True,'all94assets_exact':True,'maze_exact':True,'supplied_dynamics_exact':True,'canonical_pc_normals_exact':True,'constructor_steps':0,'max_rotation_matrix_readback':max_rotation_readback,'visual_only_inertial_quaternion_not_compared':visual_only_inertial_quaternions}}
        output_file.write_text(json.dumps(encode(reply),indent=2,allow_nan=False)+'\n')
    finally:p.disconnect(env.pb_physics_client)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();child(args.request,args.output)
