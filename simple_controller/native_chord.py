"""Bounded native COM-chord placement; world SI and Bullet XYZW.
q slots13/14 map to native21/22. Collisionshape origins are COM-relative.
Only invoked in state-restoring planning; all physical dispatch remains legacy.
"""
from itertools import product
from time import monotonic
import numpy as np
import pybullet as p
from .native import rotation

IK_POSITION_TOLERANCE = .002  # Native.ik convergence bound, metres


class PlanningLimit(RuntimeError):
    pass


def first_touch(n, q, check=lambda: None):
    n.set(q)
    origin, quat = n.hand_pose()
    frame = rotation(quat)
    touched = {}
    for slot, link in zip((13, 14), sorted(n.fingers)):
        value = q[slot]
        for _ in range(60):
            check()
            p.resetJointState(n.robot, link, float(value))
            hits = p.getClosestPoints(n.robot, n.object, .12, linkIndexA=link)
            if not hits:
                break
            hit = min(hits, key=lambda x: x[8])
            if hit[8] <= 1e-6:
                touched[link] = {"finger_q_m": float(value), "distance_m": float(hit[8]),
                    "point_hand_m": ((np.asarray(hit[5])-origin)@frame).tolist(),
                    "point_object_world_m": list(hit[6]), "normal_world": list(hit[7])}
                break
            if value == n.lower[slot]:
                break
            value = max(n.lower[slot], value-hit[8])
        n.set(q)
    if len(touched) != 2:
        return touched, None
    records = list(touched.values())
    a, b = [np.asarray(x["point_object_world_m"]) for x in records]
    direction = (a-b)/np.linalg.norm(a-b)
    cos = [float(sign*direction@r["normal_world"]) for sign,r in zip((1.,-1.),records)]
    com = np.asarray(p.getBasePositionAndOrientation(n.object)[0])
    eccentricity = max(abs(r["point_hand_m"][0]) for r in records)
    return touched, {"normal_cosines": cos,
        "friction_tangents": [float(np.sqrt(max(0.,1-c*c))/c) if c>0 else None for c in cos],
        "lever_xy_m": float(np.linalg.norm((com-(a+b)/2)[:2])),
        "max_pad_abs_x_m": float(eccentricity)}


def inside_friction_cones(metric, mu):
    return metric is not None and all(v is not None and v<=mu for v in metric["friction_tangents"])


def contact_improves(old, new, mu):
    # Point-pair diagnostics are ambiguous for broad flat-face contact. Leave
    # those legacy plans untouched instead of diagnosing arbitrary corners.
    return (inside_friction_cones(old,mu) and inside_friction_cones(new,mu)
            and old["max_pad_abs_x_m"]-new["max_pad_abs_x_m"]>IK_POSITION_TOLERANCE
            and new["lever_xy_m"]<=old["lever_xy_m"]+IK_POSITION_TOLERANCE)


def pad_geometry(n, q):
    q=q.copy();q[-2:]=0.;n.set(q)
    hand,quat=n.hand_pose();frame=rotation(quat);bounds=[]
    for link in sorted(n.fingers):
        shapes=p.getCollisionShapeData(n.robot,link)
        if len(shapes)!=1 or shapes[0][2]!=p.GEOM_BOX:
            return None
        shape=shapes[0];state=p.getLinkState(n.robot,link,computeForwardKinematics=True)
        corners=np.asarray(list(product((-1.,1.),repeat=3)))*np.asarray(shape[3])/2
        world=(corners@rotation(shape[6]).T+shape[5])@rotation(state[1]).T+state[0]
        local=(world-hand)@frame
        bounds.append([local.min(axis=0),local.max(axis=0)])
    return np.asarray(bounds)


def improve_contact(n, observation, original, config, notes):
    """At most one COM chord/axial strip; original plan on any rejection."""
    started=monotonic();deadline=started+min(60.,config["wall_limit_seconds"]/3)
    counts={"queries":0,"ik":0,"edge":0};saved={}
    def check():
        counts["queries"]+=1
        if monotonic()>deadline or counts["queries"]>1200:
            raise PlanningLimit
    def wrap(name, maximum):
        fn=getattr(n,name);saved[name]=(name in n.__dict__,n.__dict__.get(name))
        def call(*a,**kw):
            check();counts[name]+=1
            if counts[name]>maximum:raise PlanningLimit
            return fn(*a,**kw)
        setattr(n,name,call)
    notes.update(status="review",eligibility="baseline_point_chord_inside_material_cones_then_improve_axial_pad_reserve",
                 limits={"seconds":min(60.,config["wall_limit_seconds"]/3),"queries":1200,"ik":80,"edge":256})
    wrap("ik",80);wrap("edge",256)
    try:
        start=n.q();com=np.asarray(p.getBasePositionAndOrientation(n.object)[0])
        mu=p.getDynamicsInfo(n.object,-1)[1]*min(p.getDynamicsInfo(n.robot,k)[1] for k in n.fingers)
        touches,baseline=first_touch(n,original.approach[-1],check);notes["baseline_first_touch"]=baseline;notes["mu_pair"]=mu
        if not inside_friction_cones(baseline,mu):
            notes["status"]="legacy_non_point_chord";return original
        n.set(original.approach[-1]);frame=rotation(n.hand_pose()[1]);axis=frame[:,1]
        n.set(start);box=np.asarray(p.getAABB(n.object));radius=float(np.linalg.norm(box[1]-box[0]));hits=[]
        for sign in (-1.,1.):
            check();hit=p.rayTest(com+sign*radius*axis,com)[0]
            hits.append(hit)
        if any(hit[0]!=n.object or hit[1]!=-1 for hit in hits):
            notes["status"]="object_ray_blocked";return original
        points=np.asarray([h[3] for h in hits]);middle=points.mean(axis=0);width=float(np.linalg.norm(points[1]-points[0]))
        bounds=pad_geometry(n,start)
        if bounds is None:notes["status"]="unsupported_pad_shape";return original
        center=np.mean(bounds[:,:,[0,2]],axis=(0,1));gap=bounds[0,0,1]-bounds[1,1,1]
        target=middle-frame[:,0]*center[0]-frame[:,2]*center[1]
        opening=max(0.,(width+2*IK_POSITION_TOLERANCE-gap)/2)
        notes.update(ray_points_world_m=points.tolist(),ray_normals_world=[list(h[4]) for h in hits],pad_bounds_hand_m=bounds.tolist(),opening_m=opening)
        if opening>min(n.upper[-2:]):notes["status"]="opening_limit";return original
        seed=original.pregrasp.copy();seed[-2:]=opening
        def clearance(offset):
            q=n.ik(seed,target-offset*frame[:,0],frame,config["ik_iterations"])
            if q is None:return None,None,[]
            witnesses=[]
            for a,b in product((0.,opening),repeat=2):
                actual=q.copy();actual[-2:]=[a,b];n.set(actual)
                for body in (*n.obstacles,n.object):
                    check()
                    for hit in p.getClosestPoints(n.robot,body,.1):
                        if hit[3] not in {n.hand,*n.fingers}:continue
                        if body==n.object and hit[3] in n.fingers and (a,b)!=(opening,opening):continue
                        witnesses.append((hit[8],np.asarray(hit[7])))
            return min((w[0] for w in witnesses),default=.1),q,witnesses
        d0,q0,witness=clearance(0.)
        if d0 is None:notes["status"]="center_ik";return original
        maximum=float(min(bounds[:,1,0]));up=-frame[:,0]
        projected=max([max(0.,-d)/(normal@up) for d,normal in witness if d<0. and normal@up>0.]+[0.])
        lo=0.;hi=min(projected,maximum);dh,qhi,_=clearance(hi)
        if dh is None or dh<0.:
            lo=hi;hi=maximum;dh,qhi,_=clearance(hi)
        if dh is None or dh<0.:notes["status"]="no_axial_strip";return original
        if d0>=0.:hi=0.;qhi=q0
        else:
            for _ in range(16):
                if hi-lo<=IK_POSITION_TOLERANCE:break
                mid=(lo+hi)/2;dm,qm,_=clearance(mid)
                if dm is not None and dm>=0.:hi=mid;qhi=qm
                else:lo=mid
        target=target-hi*frame[:,0];final=qhi.copy();final[-2:]=opening
        notes.update(axial_offset_m=hi,axial_bracket_m=[lo,hi],pad_tip_reserve_m=maximum-hi)
        staging=original.staging.copy();base=original.base.copy();staging[-2:]=base[-2:]=opening
        opened=start.copy();opened[-2:]=opening
        pre=n.ik(seed,target-.083*frame[:,0],frame,config["ik_iterations"])
        if pre is None:notes["status"]="pregrasp_ik";return original
        for a,b in ((start,opened),(opened,staging),(staging,base),(base,pre)):
            if not n.edge(a,b):notes["status"]="access_edge";return original
        previous=pre;approach=[]
        for distance in (.055,.028,0.):
            q=final.copy() if distance==0. else n.ik(previous,target-distance*frame[:,0],frame,45)
            if q is None or not n.edge(previous,q):notes["status"]="approach_edge";return original
            approach.append(q);previous=q
        touched,metric=first_touch(n,previous,check);notes["candidate_first_touch"]=metric
        if not contact_improves(baseline,metric,mu):notes["status"]="no_contact_improvement";return original
        closed=previous.copy();closed[-2:]=0.
        if not n.edge(previous,closed,fingers=True):notes["status"]="closure_edge";return original
        measured=previous.copy();measured[-2:]=[touched[k]["finger_q_m"] for k in sorted(n.fingers)]
        n.set(measured);relative=n.payload_transform();lift=[]
        for rise in (.035,.075,.118,.149):
            q=n.ik(measured,target+[0.,0.,rise],frame,50)
            if q is None or not n.edge(measured,q,fingers=True,payload=relative,held_palm=True):notes["status"]="lift_edge";return original
            lift.append(q);measured=q
        n.set(measured);n.position_payload(relative)
        # Reuse the original full route/terminal admission, no duplicated planner.
        from .controller import Controller
        ctrl=object.__new__(Controller);ctrl.native=n;ctrl.env=n.env;ctrl.config=config;ctrl.extra={}
        selected=ctrl.plan_carry(measured,relative)
        if selected is None:notes["status"]="no_carry";return original
        check();notes.update(status="selected",prospective_carry_kind=selected[3],prospective_carry_prefix_edges=len(selected[0]),prospective_route_waypoints=len(selected[1]),first_touch={str(k):v for k,v in touched.items()})
        return type(original)(staging,base,pre,approach,target,frame,lift,original.family)
    except PlanningLimit:
        notes["status"]="planning_limit";return original
    finally:
        notes.update(counts=counts,elapsed_seconds=monotonic()-started)
        for name,(existed,value) in saved.items():
            if existed:setattr(n,name,value)
            else:delattr(n,name)
