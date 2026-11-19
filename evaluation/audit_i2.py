"""Additive I2 witness audit atop unchanged043 metrics/palm/scene/source audit.
Independent URDF/hand/q transforms and finite-face math; no Bullet state queries.
Recorded Native overlap evidence is checked, not claimed to be a second physical
simulation. Historical raws are never relabelled or source hashes rewritten.
"""
from pathlib import Path
import gzip,json,sys,hashlib,xml.etree.ElementTree as ET
import numpy as np
import pybullet as p

def R(q):return np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
def xyz(node,key,default='0 0 0'):return np.fromstring(node.get(key,default),sep=' ')
def finite(local,normal,half,sign,distance,force):
 return bool(np.isfinite(np.r_[local,normal,half,distance,force]).all() and force>0 and -.0005<=distance<=0 and half[0]-abs(local[0])>1e-6 and half[2]-abs(local[2])>1e-6 and abs(local[1]+sign*half[1])<=1e-6 and abs(np.linalg.norm(normal)-1)<=1e-6 and normal[1]*sign>=1-1e-6)
def hardware(root,scene):
 urdf=ET.parse(Path(root)/'assets/fetch/fetch.urdf');children={}
 for joint in urdf.findall('joint'):children.setdefault(joint.find('parent').get('link'),[]).append(joint)
 childlinks={j.find('child').get('link') for j in urdf.findall('joint')};rootlink=next(x.get('name') for x in urdf.findall('link') if x.get('name') not in childlinks);order=[]
 def visit(name):
  for joint in children.get(name,[]):order.append(joint);visit(joint.find('child').get('link'))
 visit(rootlink);assert order[scene['hand_link']].find('child').get('link')=='gripper_link';frames={}
 for link in scene['finger_links']:
  j=order[link];assert j.get('type')=='prismatic' and j.find('parent').get('link')=='gripper_link';index=scene['joint_indices'].index(link)
  collision=urdf.find("./link[@name='%s']/collision"%j.find('child').get('link'));jo=j.find('origin');co=collision.find('origin');J=R(p.getQuaternionFromEuler(xyz(jo,'rpy')));C=R(p.getQuaternionFromEuler(xyz(co,'rpy')));axis=xyz(j.find('axis'),'xyz');opening=C.T@axis;assert np.allclose(abs(opening),[0,1,0],atol=1e-12,rtol=0.)
  frames[link]=(index,xyz(jo,'xyz'),J,axis,xyz(co,'xyz'),C,xyz(collision.find('geometry/box'),'size')/2,float(np.sign(opening[1])))
 support={i for i,j in enumerate(order) if j.find('child').get('link') in ('base_link','r_wheel_link','l_wheel_link')}
 return frames,support

def check_record(row,rec,scene,frames):
 link=rec['link'];hit=rec['raw_point'];assert hit[1]==scene['body_ids']['robot'] and hit[2]==scene['body_ids']['object'] and hit[3]==link and hit[4]==-1
 index,origin,J,axis,offset,C,half,sign=frames[link];H=R(row['hand_xyzw']);worldR=H@J@C;worldorigin=np.asarray(row['hand_position'])+H@(origin+J@(axis*row['q'][index]+offset))
 local=worldR.T@(np.asarray(hit[5])-worldorigin);normal=worldR.T@hit[7];force=sum(x[1] for x in row['finger_contacts'][str(link)])
 # Logged hand link pose is Bullet float32 readback, producer uses COM doubles.
 np.testing.assert_allclose(local,rec['local_point_m'],atol=1e-6,rtol=0.);np.testing.assert_allclose(normal,rec['local_normalOnB'],atol=1e-6,rtol=0.)
 np.testing.assert_array_equal(half,rec['half_m']);assert sign==rec['opening_sign'] and force==rec['cached_same_link_force_N'] and hit[8]==rec['distance_m']
 accepted=finite(local,normal,half,sign,hit[8],force);assert accepted==rec['accepted'];return accepted

def check_overlap_tuples(row,scene,frames,support):
 records=row['early_contact_finger_candidates'];expected=[];other=[]
 for witness in row['early_contact_native_overlaps']:
  hit=witness['point'];kind=witness['kind']
  if kind=='robot_scene' and hit[2]==scene['body_ids']['plane'] and hit[3] in support:continue
  if kind=='robot_scene' and hit[2]==scene['body_ids']['object'] and hit[3] in frames:
   # EXACT one recorded raw tuple for every offending point; no count-only
   # substitute, unlogged duplicate, or different link/body/depth may disappear.
   assert sum(r['raw_point']==hit for r in records)==1
   expected.append(('robot_scene',hit[2],hit[3],hit[4],hit[8]))
  else:other.append(witness)
 assert len(expected)==len(records)
 assert sorted(expected)==sorted(('robot_scene',r['raw_point'][2],r['raw_point'][3],r['raw_point'][4],r['raw_point'][8]) for r in records)
 return not other

def decode(value):
 if isinstance(value,dict):
  if set(value)=={'__array__','dtype'}:return np.asarray(value['__array__'],dtype=value['dtype'])
  return {k:decode(v) for k,v in value.items()}
 if isinstance(value,list):return [decode(v) for v in value]
 return value

def check_lift_query(folder,root,result,scene,rows):
 notes=result.get('early_contact_lift')
 if not notes:return None
 archive=folder/notes['isolated_query_artifact'];payload=gzip.open(archive/'request.json.gz','rb').read();request=decode(json.loads(payload));assert hashlib.sha256(payload).hexdigest()==notes['snapshot_sha256']
 assert request['producer_sha256']==json.loads((folder/'source.json').read_bytes());assert request['maze_sha256']==scene['maze_sha256'];assert len(request['assets'])==94
 import pybullet_data
 assets={}
 for name,pin in request['assets'].items():
  path=Path(name)
  try:key=path.relative_to(root.resolve()).as_posix()
  except ValueError:key='pybullet_data/'+path.relative_to(Path(pybullet_data.getDataPath())).as_posix()
  assets[key]=pin
 assert assets==scene['scene_assets'];assert request['reference']['initial_bottom']==result['settled_bottom'];np.testing.assert_array_equal(request['reference']['initial_position'],result['settled_position'])
 # Request is obtained after the last ordinary bilateral closure action, not at
 # an old grasp target. qvel/body velocities are retained even though trace has
 # no historical qvel fields.
 preceding=[r for r in rows if r['phase']=='close'];assert preceding;closure=preceding[-1];saved=request['bodies']['robot'];q=[saved['joints'][i][0] for i in scene['joint_indices']];np.testing.assert_array_equal(q,closure['q'])
 if notes['status']=='selected':
  reply=decode(json.loads((archive/'reply.json').read_bytes()));assert reply['status']=='PASS' and reply['physics_steps']==0 and reply['snapshot_sha256']==notes['snapshot_sha256'];assert reply['state_verified']['all94assets_exact'] and reply['state_verified']['all_body_poses_velocities_joint_q_qvel'] and reply['state_verified']['supplied_dynamics_exact'];assert len(reply['data']['q15_chain'])==4;assert len(reply['notes']['edges'])==4 and all(x['free'] for x in reply['notes']['edges']);assert reply['notes']['modeled_lift_eligible'] and not reply['notes']['terminal_violations']
 return {'status':notes['status'],'request_source_scene_assets_refs_actual_closure_verified':True}


def audit(folder,root):
 folder,root=Path(folder).resolve(),Path(root).resolve();sys.path.insert(0,str(root))
 from evaluation.palm_faces043.audit import audit as original
 base=original(folder,root);assert base['current_sources_match'];scene=json.loads((folder/'scene.json').read_bytes());result=json.loads((folder/'result.json').read_bytes());frames,support=hardware(root,scene)
 stream=gzip.open(folder/'trace.jsonl.gz','rt') if (folder/'trace.jsonl.gz').exists() else (folder/'trace.jsonl').open();rows=[json.loads(x) for x in stream];stream.close();events=[];samples=0;pending=[]
 for row in rows:
  if not row.get('early_contact_pending'):continue
  assert row['phase']=='approach';records=row['early_contact_finger_candidates'];accepted=[check_record(row,r,scene,frames) for r in records];samples+=len(records);other_clear=check_overlap_tuples(row,scene,frames,support)
  q=np.asarray(row['q']);joint_bad=bool(np.any(q<np.asarray(scene['joint_lower'])-.001) or np.any(q>np.asarray(scene['joint_upper'])+.001))
  if joint_bad:assert ['joint_limit'] in row['violations']
  if not row['violations']:assert all(accepted) and other_clear and not joint_bad
  if not pending:assert records and all(accepted) and other_clear;events.append({'call':row['call'],'tick':row['tick']})
  else:assert row['tick']==pending[-1]['tick']+1 and row['call']==pending[-1]['call']
  pending.append(row)
 if pending:
  assert pending[-1]['substep']==5 and len(pending)==6-pending[0]['substep']
 if pending and result.get('early_closure'):
  closure=result['early_closure'];boundary=pending[-1];assert closure['body_anchor_q']==boundary['q'];assert closure['boundary_tick']==boundary['tick'];assert closure['boundary_call']==boundary['call']
  first=rows[boundary['tick']];assert first['phase']=='close';q=np.array(boundary['q'],dtype=np.float32).astype(float);q[-2:]=0
  # Action and actual q are float32 at env boundary: wrapped body delta is0.
  q=np.clip(q,np.asarray(scene['joint_lower']),np.asarray(scene['joint_upper']));np.testing.assert_allclose(first['target'],q,atol=1e-12,rtol=0.)
  assert all(x['phase']!='approach' for x in rows[boundary['tick']:])
  notes=result.get('early_contact_lift')
  if notes and notes.get('status')=='selected':
   anchor=rows[next(i for i,r in enumerate(rows) if r['phase']=='lift')-1];np.testing.assert_allclose(notes['anchor_position_m'],anchor['hand_position'],atol=1e-6,rtol=0.);np.testing.assert_allclose(R(notes['anchor_xyzw']),R(anchor['hand_xyzw']),atol=1e-6,rtol=0.)
   np.testing.assert_allclose(np.array(notes['requested_positions_m'])-notes['anchor_position_m'],[[0,0,.035],[0,0,.075],[0,0,.118],[0,0,.149]],atol=1e-12,rtol=0.)
 query=check_lift_query(folder,root,result,scene,rows)
 return {**base,'early_lift_query':query,'early_contact_event_witness_samples':samples,'early_contact_events':events,'I2_additive_event_audit':'PASS','qualification':'Recorded witness geometry independently recomputed; full043 metrics unchanged.'}
if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1]),Path(sys.argv[2])),indent=2))
