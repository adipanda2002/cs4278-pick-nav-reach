"""Execute only a newly certifiedSE2fallback; never flatten yaw intoXYroute.
Originalfollow/step closecommand/contact/geometry/clock/arrival guards apply.
"""
def execute(controller,path):
 from .controller import StopTrial
 if len(path)<2:raise ValueError('SE2fullchainrequired')
 previous=path[0]
 for target in path[1:]:
  yaw_delta=(float(target[2])-float(previous[2])+3.141592653589793)%(2*3.141592653589793)-3.141592653589793
  posture_changed=any(abs(float(a)-float(b))>1e-9 for a,b in zip(target[3:],previous[3:]))
  controller.follow(target,'navigate',continuous=abs(yaw_delta)<=1e-9 and not posture_changed)
  previous=target
  if controller.arrival:return
 target=path[-1]
 for _ in range(controller.config['waypoint_wait_calls']):
  controller.step(target)
  if controller.arrival:return
 controller.finish_goal()
 if not controller.arrival:raise StopTrial('goal_not_held_within_tolerance')
