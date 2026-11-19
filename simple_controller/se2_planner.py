"""Bounded private SE2 lattice. WorldSI/radians; q15 fixed slots3:;
baseXY=q[:2]+[-2,0], continuous yawslot2. No IK, physics or object IDs.
"""
import heapq,math,time
import numpy as np
from simple_controller.native import EDGE_RESOLUTION
class Bound(RuntimeError):pass
def R2(yaw):
 c,s=np.cos(yaw),np.sin(yaw);return np.array([[c,-s],[s,c]])
def delta(a,b,wrap):
 d=b-a;d[wrap]=(d[wrap]+np.pi)%(2*np.pi)-np.pi;return d
def samples(a,b,wrap):
 d=delta(a,b,wrap);count=max(1,int(np.ceil(np.max(abs(d)/EDGE_RESOLUTION))))
 return [a+fraction*d for fraction in np.linspace(0.,1.,count+1)]
def goal_for_heading(start,object_pos,goal,yaw):
 base=start[:2]+[-2.,0.];local=R2(-start[2])@(np.asarray(object_pos)[:2]-base);q=start.copy();q[2]=yaw;q[:2]=np.asarray(goal)[:2]-R2(yaw)@local-[-2.,0.];return q
class Planner:
 spacing=.25;headings=16;node_limit=16000;edge_limit=80000;closest_limit=400000
 def __init__(self,n,start,relative,goal):
  self.n=n;self.start=start.copy();self.relative=relative;self.goal=np.asarray(goal);self.wrap=n.wrap;self.began=time.monotonic();self.nodes=0;self.edge_count=0;self.closest_count=0;self.pose_count=0;self.cache={};self.edge_cache={};self.last={};self.rejects={};self.completed=None
  n.set(start);n.position_payload(relative);import pybullet as p
  self.p=p;self.object0=np.asarray(p.getBasePositionAndOrientation(n.object)[0]);self.yaws=start[2]+2*np.pi*np.arange(self.headings)/self.headings;self.goals=[goal_for_heading(start,self.object0,goal,yaw) for yaw in self.yaws];self.lo=np.array([-18,-23]);self.hi=np.array([72,23]);self.counts={}
 def check(self):
  if time.monotonic()-self.began>90:raise Bound('90ssearchwall')
  if self.nodes>=self.node_limit:raise Bound('16000expandednodes')
  if self.edge_count>=self.edge_limit:raise Bound('80000searchededges')
  if self.closest_count>=self.closest_limit:raise Bound('400000sceneclosestqueries')
 def place(self,q):
  assert np.array_equal(q[3:],self.start[3:])
  for index,value in zip(self.n.env.joint_indices[:3],q[:3]):self.p.resetJointState(self.n.robot,index,float(value))
  self.n.position_payload(self.relative)
 def scene(self,q,usecache=True):
  self.check();key=tuple(q[:3]);self.pose_count+=1
  if usecache and key in self.cache:return self.cache[key]
  if not np.isfinite(q).all() or np.any(q<self.n.lower-.001) or np.any(q>self.n.upper+.001):self.cache[key]=False;return False
  self.place(q);bad=None
  for bodyA,bodyB in ((self.n.robot,self.n.env.table_id),(self.n.robot,self.n.env.maze_id),(self.n.object,self.n.env.table_id),(self.n.object,self.n.env.maze_id)):
   self.check();self.closest_count+=1;hits=self.p.getClosestPoints(bodyA,bodyB,0.)
   if hits:
    h=min(hits,key=lambda x:x[8]);bad={'bodyA':bodyA,'bodyB':bodyB,'linkA':h[3],'linkB':h[4],'distance_m':h[8],'pointA_world_m':h[5],'pointB_world_m':h[6],'normalOnB_world':h[7]};break
  good=bad is None;self.cache[key]=good
  if bad:
   label=str((bad['bodyA'],bad['bodyB'],bad['linkA'],bad['linkB']));self.rejects[label]=self.rejects.get(label,0)+1
   if len(self.last)<12:self.last[label]={'q':q.copy(),'witness':bad}
  return good
 def edge(self,a,b):
  self.check();self.edge_count+=1;key=(tuple(a[:3]),tuple(b[:3]))
  if key in self.edge_cache:return self.edge_cache[key]
  if not self.scene(b):self.edge_cache[key]=False;return False
  for q in samples(a,b,self.wrap):
   if not self.scene(q):self.edge_cache[key]=False;return False
  self.edge_cache[key]=True;return True
 def qnode(self,node):
  q=self.start.copy();q[:2]=self.spacing*np.asarray(node[:2])-[-2.,0.];q[2]=self.yaws[node[2]];return q
 def heuristic_grid(self,valid_goals):
  # Point-robot grid ONLY orders search; never rejects an SE2 node/edge.
  points={};walls=self.n.env.cube_positions[:,:2];front=[];cost={}
  for i in range(self.lo[0],self.hi[0]+1):
   for j in range(self.lo[1],self.hi[1]+1):
    x=np.array([i,j])*self.spacing;points[(i,j)]=not np.any(np.all(abs(walls-x)<=.500001,axis=1))
  for k in valid_goals:
   node=tuple(np.rint((self.goals[k][:2]+[-2.,0.])/self.spacing).astype(int))
   if points.get(node):cost[node]=0.;heapq.heappush(front,(0.,node))
  while front:
   d,node=heapq.heappop(front)
   if d!=cost[node]:continue
   for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
    nxt=(node[0]+di,node[1]+dj);nd=d+self.spacing
    if points.get(nxt,False) and nd<cost.get(nxt,float('inf')):cost[nxt]=nd;heapq.heappush(front,(nd,nxt))
  return cost
 def search(self):
  valid=[k for k,g in enumerate(self.goals) if self.scene(g)];self.valid_goals=valid
  if not valid:return None,'no_scene_clear_lattice_goal_heading'
  hgrid=self.heuristic_grid(valid);heap=[];cost={};parent={};serial=0;lead=tuple(np.rint((self.start[:2]+[-2.,0.])/self.spacing).astype(int));offsets=[(i,j) for i in (-1,0,1) for j in (-1,0,1)]
  def heuristic(node):
   xy=self.spacing*np.array(node[:2]);straight=min(np.linalg.norm(xy-(self.goals[k][:2]+[-2.,0.])) for k in valid);return max(straight,hgrid.get(node[:2],straight))
  for di,dj in offsets:
   node=(lead[0]+di,lead[1]+dj,0);q=self.qnode(node)
   if self.edge(self.start,q):g=float(np.linalg.norm(q[:2]-self.start[:2]));cost[node]=g;parent[node]=None;serial+=1;heapq.heappush(heap,(g+heuristic(node),serial,g,node))
  while heap:
   self.check();_,_,g,node=heapq.heappop(heap)
   if g!=cost.get(node):continue
   self.nodes+=1;q=self.qnode(node);k=node[2]
   if k in valid and np.linalg.norm(q[:2]-self.goals[k][:2])<=self.spacing*np.sqrt(2)+1e-12 and self.edge(q,self.goals[k]):
    path=[self.goals[k].copy()];cur=node
    while cur is not None:path.append(self.qnode(cur));cur=parent[cur]
    path.append(self.start.copy());self.completed=path[::-1];return self.completed,'found'
   successors=[(node[0]+di,node[1]+dj,k) for di,dj in offsets if di or dj]+[(node[0],node[1],(k+d)%self.headings) for d in (-1,1)]
   for nxt in successors:
    if not(self.lo[0]<=nxt[0]<=self.hi[0] and self.lo[1]<=nxt[1]<=self.hi[1]):continue
    turn=nxt[2]!=k;ng=g+(.20 if turn else self.spacing*np.linalg.norm(np.asarray(nxt[:2])-node[:2]))
    if ng>=cost.get(nxt,float('inf')):continue
    qnext=self.qnode(nxt)
    if not self.edge(q,qnext):continue
    cost[nxt]=ng;parent[nxt]=node;serial+=1;heapq.heappush(heap,(ng+heuristic(nxt),serial,ng,nxt))
  return None,'lattice_exhausted_UNKNOWN_continuous_feasibility'
 def record(self):return {'nodes_expanded':self.nodes,'edges_considered':self.edge_count,'scene_closest_queries':self.closest_count,'pose_requests':self.pose_count,'cached_poses':len(self.cache),'wall_seconds':time.monotonic()-self.began,'scene_rejection_counts':self.rejects,'scene_rejection_examples':self.last}
