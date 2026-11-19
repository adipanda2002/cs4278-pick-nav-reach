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

def query_lift(native,reference,config,notes):
    """After actual early-contact bilateral closure; live getters/files only."""
    started=monotonic();request=snapshot(native,native.env._get_obs(),None,config)
    request['reference']=encode(reference)
    request['producer_sha256']={str(path.relative_to(ROOT)):digest(path.read_bytes()) for path in (ROOT/'simple_controller').glob('*.py')}
    request['producer_sha256'].update({name:digest((ROOT/name).read_bytes()) for name in ('run_simple.py','pick_nav_reach_env.py','maze_utils.py','utils.py','keyboard_control.py','simple_controller/config.json')})
    payload=json.dumps(request,separators=(',',':'),allow_nan=False).encode();notes.update(status='isolated_contact_lift_review',backend='zero_step_child',snapshot_sha256=digest(payload),child_timeout_seconds=60)
    with tempfile.TemporaryDirectory(prefix='pnr-contact-lift-') as directory:
        folder=Path(directory);input_file=folder/'request.json';output_file=folder/'reply.json';input_file.write_bytes(payload);proc=None
        try:
            with (folder/'child.log').open('wb') as log:
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--request',str(input_file),'--output',str(output_file)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                code=proc.wait(timeout=60)
            if code!=0 or not output_file.is_file():notes.update(status='child_error',child_exit_code=code);return None
            reply=decode(json.loads(output_file.read_bytes()))
            if reply['status']!='PASS' or reply['physics_steps']!=0 or reply['snapshot_sha256']!=digest(payload):notes['status']='child_integrity_error';return None
            notes.update(plain(reply['notes']));notes['child_state']=reply['state_verified']
            if notes['status']!='selected':return None
            path=reply['data']['q15_chain']
            assert len(path)==4 and all(q.shape==(15,) and np.isfinite(q).all() for q in path)
            return path
        except subprocess.TimeoutExpired:notes['status']='child_timeout';return None
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            archive=Path(tempfile.mkdtemp(prefix='isolated_contact_lift_',dir=native.env.maze_path.parent))
            with gzip.open(archive/'request.json.gz','wb') as stream:stream.write(payload)
            for name in ('reply.json','child.log','completed_path.json'):
                if (folder/name).exists():(archive/name).write_bytes((folder/name).read_bytes())
            notes['isolated_query_artifact']=archive.name;notes['isolated_elapsed_seconds']=monotonic()-started


def lift_waypoints(hand):
    """World-Z metre offsets from actual closure, never old grasp endpoint."""
    return [np.asarray(hand)+np.array([0.,0.,z]) for z in (.035,.075,.118,.149)]


def plan_private(n,notes,checkpoint,reference):
    from simple_controller.native import rotation
    start=n.q();anchor,quat=n.hand_pose();anchor=np.asarray(anchor);quat=np.asarray(quat);R=rotation(quat);relative=n.payload_transform()
    targets=lift_waypoints(anchor);path=[];edges=[];actual=start.copy()
    notes.update(status='review',anchor_position_m=anchor.tolist(),anchor_xyzw=quat.tolist(),requested_positions_m=[x.tolist() for x in targets])
    for index,position in enumerate(targets):
        q=n.ik(actual,position,R)
        if q is None:notes.update(status='lift_ik_failed',failed_waypoint=index);return None
        q[-2:]=start[-2:]
        free=n.edge(actual,q,fingers=True,payload=relative,held_palm=True)
        edges.append({'index':index,'free':bool(free)})
        if not free:notes.update(status='lift_edge_blocked',edges=edges,failed_q=n.q().tolist(),violations=n.violations(n.q(),fingers=True,held_palm=True));return None
        path.append(q.copy());actual=q.copy()
    n.set(path[-1]);n.position_payload(relative)
    bottom=float(p.getAABB(n.object)[0][2]);com=np.asarray(p.getBasePositionAndOrientation(n.object)[0]);table=float(p.getAABB(n.env.table_id)[1][2])
    bad=n.violations(actual,fingers=True,lifted=True,held_palm=True)
    qualified=(bottom-table>=.1 and bottom-reference['initial_bottom']>=.1 and np.linalg.norm(com-reference['initial_position'])>=.1)
    notes.update(edges=edges,terminal_violations=bad,modeled_bottom_m=bottom,modeled_COM_world_m=com.tolist(),modeled_lift_eligible=bool(qualified))
    if bad or not qualified:notes['status']='lift_terminal_unqualified';return None
    data={'q15_chain':path,'relative_payload':relative,'anchor_position_m':anchor,'anchor_xyzw':quat}
    checkpoint.write_text(json.dumps(encode(data),indent=2,allow_nan=False)+'\n')
    notes['status']='selected';return data


def child(request_file,output_file):
    """Full independent native world; no stepping, including constructor."""
    from pick_nav_reach_env import PickNavReachEnv
    from simple_controller.native import Native,rotation
    from simple_controller.pick import PickPlan
    payload=request_file.read_bytes();request=decode(json.loads(payload));traps=0
    def alarm(*args):raise TimeoutError('contact_lift50s')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(50)
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
        with n.planning():data=plan_private(n,notes,output_file.parent/'completed_path.json',request['reference'])
        assert env.step_count==env.tick_count==traps==0
        for label,body in labels.items():
            saved=request['bodies'][label];now=p.getBasePositionAndOrientation(body)
            np.testing.assert_array_equal(now[0],saved['pose'][0]);np.testing.assert_allclose(rotation(now[1]),rotation(saved['pose'][1]),atol=1e-12,rtol=0.)
            np.testing.assert_array_equal(p.getBaseVelocity(body),saved['velocity'])
            np.testing.assert_array_equal([p.getJointState(body,i)[:2] for i in range(p.getNumJoints(body))],saved['joints'])
        reply={'status':'PASS','snapshot_sha256':digest(payload),'notes':notes,'data':data,'physics_steps':0,
            'state_verified':{'body_count':len(labels),'all_body_poses_velocities_joint_q_qvel':True,'all94assets_exact':True,'maze_exact':True,'supplied_dynamics_exact':True,'canonical_pc_normals_exact':True,'constructor_steps':0,'max_rotation_matrix_readback':max_rotation_readback,'visual_only_inertial_quaternion_not_compared':visual_only_inertial_quaternions}}
        output_file.write_text(json.dumps(encode(reply),indent=2,allow_nan=False)+'\n')
    finally:signal.alarm(0);p.disconnect(env.pb_physics_client)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();child(args.request,args.output)
