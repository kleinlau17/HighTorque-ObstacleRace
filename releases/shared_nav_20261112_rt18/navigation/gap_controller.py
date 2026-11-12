"""Follow the observed open side, brake into the gap and cross on one controller."""
import numpy as np


def rotation(yaw):
    return np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])


class GapController:
    # Body margins in the lane frame. The archived Pro left-wrist contact
    # extended about 0.15 m forward of the base at the obstacle's height.
    front_reach = .155
    rear_reach = .135
    side_reach = .285

    def __init__(self):
        self.phase = 'idle'
        self.origin = self.target = self.identity = self.entry = None
        self.heading = self.started = self.half_window = 0.
        self.completed = []
        self.error = np.zeros(2)
        self.stalled_since = None
        self.recoveries = 0
        self.drive_until = self.hold_reason = None
        self.side_clearance = True

    def translate_map(self, displacement):
        for name in ('origin', 'target', 'identity', 'entry'):
            point = getattr(self, name)
            if point is not None:
                setattr(self, name, point+displacement)
        self.completed = [point+displacement for point in self.completed]

    def _select(self, perception, observed_pose, heading):
        basis = rotation(heading)
        transform = rotation(observed_pose[2]).T@basis
        boxes = np.asarray(perception.obstacles).reshape(-1, 4)
        xy = boxes[:, :2]@transform
        sizes = boxes[:, 2:]@abs(transform)
        lane_y = np.polyval(perception.lane, boxes[:, 0])
        lane_xy = np.column_stack([boxes[:,0],lane_y])@transform
        offsets = xy[:,1]-lane_xy[:,1]
        yaw_error = observed_pose[2]-heading
        c, s = abs(np.cos(yaw_error)), abs(np.sin(yaw_error))
        front_reach = c*self.front_reach+s*self.side_reach
        rear_reach = c*self.rear_reach+s*self.side_reach
        side_reach = c*self.side_reach+s*max(self.front_reach,self.rear_reach)
        ordered = np.argsort(xy[:, 0])
        for i, j in zip(ordered, ordered[1:]):
            if offsets[i]*offsets[j]>=0 or xy[j,0]<=0 or xy[j,0]>1.4:
                continue
            lower = xy[i,0]+sizes[i,0]/2+rear_reach
            upper = xy[j,0]-sizes[j,0]/2-front_reach
            if upper<-.06 or lower>.90:
                continue
            identity = observed_pose[:2]+basis@((xy[i]+xy[j])/2)
            if any(np.linalg.norm(identity-p)<.20 for p in self.completed):
                continue
            # An active pair must not switch to the next gap while crossing.
            if self.phase!='idle' and np.linalg.norm(identity-self.identity)>.20:
                continue
            gap = (lower+upper)/2
            entry_side = xy[i,1]-np.sign(offsets[i])*(sizes[i,1]/2+side_reach+.01)
            exit_side = xy[j,1]-np.sign(offsets[j])*(sizes[j,1]/2+side_reach+.01)
            self.side_clearance = max(abs(entry_side-lane_xy[i,1]),abs(exit_side-lane_xy[j,1]))<=.27
            self.entry_direction = -np.sign(offsets[i])
            self.exit_direction = -np.sign(offsets[j])
            self.identity = identity
            self.origin = observed_pose[:2]+basis@np.array([gap, 0.])
            self.entry = observed_pose[:2]+basis@np.array([gap, entry_side])
            self.target = observed_pose[:2]+basis@np.array([gap, exit_side])
            self.half_window = (upper-lower)/2
            self.first_face = xy[i,0]-sizes[i,0]/2-front_reach-gap
            self.heading = heading
            return True
        return False

    def update(self, time, perception, observed_pose, pose, world_velocity,
               corridor_heading=None, velocity_valid=True):
        self.drive_until = self.hold_reason = None
        if self.phase=='idle':
            lane_tangent = np.arctan(np.polyval(np.polyder(perception.lane), .35))
            heading = observed_pose[2]+lane_tangent if corridor_heading is None else corridor_heading
            if not self._select(perception, observed_pose, heading):
                return None
            self.phase = 'approach'
            self.started = time
            self.stalled_since = None
            self.recoveries = 0
        elif perception.lane is not None:
            # Stored targets bridge occlusion. Fresh faces refine the same pair.
            self._select(perception, observed_pose, self.heading)
        stopped = np.zeros(2)
        if self.half_window<=0 or not self.side_clearance:
            self.hold_reason='insufficient_body_clearance'
        elif not velocity_valid:
            self.hold_reason='unknown_body_velocity'
        elif self.phase=='failed':
            self.hold_reason='crossing_unreachable'
        if self.hold_reason is not None:
            self.stalled_since=None
            return stopped
        basis = rotation(self.heading)
        velocity = world_velocity@basis
        error = (self.target-pose[:2])@basis
        self.error = error
        entry_error = ((self.entry-pose[:2])@basis)[1]
        stamp = getattr(perception, 'time', time)
        age = max(0., time-stamp)
        # The adapter raises positive straight-only requests to at least 0.20.
        # Reserve the distance covered during frame age and AMP deceleration.
        closing = max(.20, velocity[0])
        stop_distance = .5*closing
        if self.phase=='approach':
            lateral = np.clip(2.5*entry_error-.20*velocity[1], -.20, .20)
            if self.entry_direction*entry_error>.01:
                lateral = self.entry_direction*max(.10,self.entry_direction*lateral)
            advance = error[0]-stop_distance
            if advance<=closing*age:
                self.phase='brake'
                self.started=time
            else:
                if self.entry_direction*entry_error>.01:
                    advance=min(advance, error[0]+self.first_face-stop_distance)
                vx=.20 if advance>closing*age else 0.
                if vx:
                    self.drive_until=stamp+advance/closing
                return np.array([vx,lateral])@basis.T@rotation(pose[2])
        # Extrapolate measured drift through the age of this image. A stale
        # stationary estimate cannot grant a lateral crossing outside the gap.
        projected_error = error[0]-velocity[0]*age
        in_window = max(abs(error[0]),abs(projected_error))<=self.half_window
        if self.phase=='brake':
            if abs(velocity[0])>.025 or not in_window:
                # A stop short of the window may resume only with room for
                # another minimum-speed stopping distance. Never pulse blindly.
                if abs(velocity[0])<.02 and time-self.started>.6:
                    if error[0]>stop_distance+.015:
                        self.phase='approach'
                    elif not in_window:
                        self.phase='failed'
                return stopped
            self.phase='shift'
            self.started=time
        if self.phase=='settle':
            if time-self.started<.5:
                return stopped
            self.phase='shift'
            self.started=time
        if not in_window:
            self.phase='brake'
            return stopped
        if self.exit_direction*error[1]<=.01 and abs(velocity[1])<.035:
            self.completed.append(self.identity.copy())
            self.phase='idle'
            return stopped
        if abs(error[1])>.025 and abs(velocity[1])<.015:
            if self.stalled_since is None:
                self.stalled_since=time
            elif time-self.stalled_since>.6:
                self.recoveries+=1
                self.phase='settle' if self.recoveries<=2 else 'failed'
                self.started=time
                self.stalled_since=None
                return stopped
        else:
            self.stalled_since=None
        vy=np.clip(3*(error[1]-.2*velocity[1]),-.20,.20)
        if self.exit_direction*error[1]>.01 and abs(velocity[1])<.035:
            vy=self.exit_direction*max(.10,self.exit_direction*vy)
        vx=np.clip(6*projected_error-.04*abs(vy)/.12-2*velocity[0],-.12,.12)
        if abs(vy)>.02:
            self.drive_until = stamp+abs(error[1])/max(.20,abs(velocity[1]))
        return np.array([vx,vy])@basis.T@rotation(pose[2])
