"""I2 finite interior-pad event, metres/world XYZW/native link indices.
The .5mm penetration cap is a conservative candidate policy, not a rubric rule.
1um frame tolerance only accommodates Bullet link readback; finite x/z margins
must exceed it. No contact, collision, or hypothetical-state queries here.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pybullet as p
from .native import rotation
FRAME_EPS=1e-6
DEPTH_CAP=.0005

def finite_inner_pad(local,normal,half,opening_sign,distance,force):
    local,normal,half=map(lambda x:np.asarray(x,dtype=float),(local,normal,half))
    reasons=[]
    if (local.shape!=(3,) or normal.shape!=(3,) or half.shape!=(3,)
            or not np.isfinite(np.r_[local,normal,half,distance,force,opening_sign]).all()):
        return False,['nonfinite_or_shape']
    if np.any(half<=0) or opening_sign not in (-1.,1.):reasons.append('hardware')
    if not (-DEPTH_CAP<=distance<=0.):reasons.append('depth')
    if not force>0.:reasons.append('no_positive_cached_force')
    if half[0]-abs(local[0])<=FRAME_EPS:reasons.append('tip_or_root')
    if half[2]-abs(local[2])<=FRAME_EPS:reasons.append('lateral')
    if abs(local[1]+opening_sign*half[1])>FRAME_EPS:reasons.append('not_inner_face')
    if abs(np.linalg.norm(normal)-1)>FRAME_EPS or opening_sign*normal[1]<1-FRAME_EPS:reasons.append('normal')
    return not reasons,reasons

def admit_only_finger_contacts(violations,records):
    # Atomic: no joint/self/palm/arm/obstacle violation can be removed alongside
    # intended finger contact. Every removed original tuple has one record.
    if (records and len(violations)==len(records) and all(r['accepted'] for r in records)
            and all(v[0]=='robot_scene' for v in violations)):
        return []
    return violations

class FingerPads:
    def __init__(self,n):
        self.n=n;self.frames={}
        root=ET.parse(Path(__file__).resolve().parents[1]/'assets/fetch/fetch.urdf')
        for link in sorted(n.fingers):
            info=p.getJointInfo(n.robot,link);name=info[12].decode()
            joint=next(x for x in root.findall('joint') if x.find('child').get('link')==name)
            assert info[2]==p.JOINT_PRISMATIC and joint.get('type')=='prismatic'
            collision=root.find(f"./link[@name='{name}']/collision")
            box=collision.find('geometry/box');assert box is not None
            size=np.fromstring(box.get('size'),sep=' ');origin=collision.find('origin')
            xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');R=rotation(p.getQuaternionFromEuler(np.fromstring(origin.get('rpy','0 0 0'),sep=' ')))
            axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ');local_axis=R.T@axis
            assert np.allclose(abs(local_axis),[0,1,0],atol=1e-12,rtol=0.)
            shapes=p.getCollisionShapeData(n.robot,link);assert len(shapes)==1 and shapes[0][2]==p.GEOM_BOX
            np.testing.assert_allclose(shapes[0][3],size,atol=1e-12,rtol=0.)
            self.frames[link]=(size/2,float(np.sign(local_axis[1])),shapes[0],xyz,R)
    def classify(self,point,contacts):
        link=int(point[3]);half,sign,shape,xyz,R=self.frames[link]
        state=p.getLinkState(self.n.robot,link,computeForwardKinematics=True)
        # Bullet collision origin is relative to COM inertial frame. Use the
        # double precision COM pose, and crosscheck actual URDF collision frame.
        worldR=rotation(state[1])@rotation(shape[6]);origin=np.asarray(state[0])+rotation(state[1])@shape[5]
        np.testing.assert_allclose(origin,np.asarray(state[4])+rotation(state[5])@xyz,atol=FRAME_EPS,rtol=0.)
        np.testing.assert_allclose(worldR,rotation(state[5])@R,atol=FRAME_EPS,rtol=0.)
        local=worldR.T@(np.asarray(point[5])-origin);normal=worldR.T@point[7]
        force=sum(float(x[1]) for x in contacts[str(link)])
        accepted,reasons=finite_inner_pad(local,normal,half,sign,float(point[8]),force)
        if point[1]!=self.n.robot or point[2]!=self.n.object or point[4]!=-1:accepted=False;reasons.append('body_or_object_link')
        return {'accepted':accepted,'reasons':reasons,'link':link,'raw_point':list(point),
                'local_point_m':local.tolist(),'local_normalOnB':normal.tolist(),'half_m':half.tolist(),
                'opening_sign':sign,'cached_same_link_force_N':force,'distance_m':float(point[8])}
