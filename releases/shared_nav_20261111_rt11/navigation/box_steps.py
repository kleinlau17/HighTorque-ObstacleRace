"""A short motion, an actual zero publication, then a newer camera capture."""
import numpy as np

MIN_STEP_PROGRESS=.003


class BoxSteps:
    def __init__(self):
        self.active=None
        self.stop_requested=None
        self.zero_stamp=None
        self.previous=None
        self.count=0
        self.reason=None
        self.last_publication_nonzero=False

    def stop(self,stamp):
        if self.active is not None and self.active['sent'] and self.active['axis']<2:
            self.previous=self.active
        self.active=None
        if self.stop_requested is None:
            self.stop_requested=stamp
            self.zero_stamp=None

    def published(self,stamp,vector):
        self.last_publication_nonzero=bool(np.max(np.abs(vector))>=1.e-8)
        if not self.last_publication_nonzero:
            if self.stop_requested is not None and self.zero_stamp is None and stamp>=self.stop_requested:
                self.zero_stamp=stamp
        elif self.active is not None and not self.active['sent']:
            # Give a weak-response repeat its full gait time from the first publication.
            self.active['sent']=True
            if self.active['retry'] or self.active.get('measured_look'):
                until=stamp+self.active['duration']
                self.active['until']=min(until,self.active.get('look_deadline',until))

    def completing_retry(self,t,frame):
        move=self.active
        return (move is not None and move['retry'] and move['frame_stamp']==frame.stamp
                and frame.stamp<=t<move['until'])

    def check(self,t,frame,position,measurements=None,yaw=None):
        move=self.active
        if move is not None:
            measured=None if measurements is None or move['axis']>=2 else measurements[move['axis']]
            observed=(measured is not None and move['sign']*(measured-move['visual'])>=move['distance'])
            angle_limit=.16 if move.get('measured_look') else .22
            turned=(move['axis']==2 and yaw is not None and move.get('yaw') is not None and
                    abs(np.arctan2(np.sin(yaw-move['yaw']),np.cos(yaw-move['yaw'])))>=angle_limit)
            axis=move['axis']
            delta=position-move['origin']
            displacement=abs(delta[axis]) if move['retry'] else np.linalg.norm(delta)
            displaced=not move.get('measured_look') and displacement>=move['distance']
            if observed or turned or t>=move['until'] or displaced:
                self.stop(t)
        if self.stop_requested is not None:
            if self.zero_stamp is None:
                return 'await_zero_publication'
            if frame.stamp<=self.zero_stamp:
                return 'wait_post_step_capture'
            self.stop_requested=None
        return None

    def little_response(self,axis,sign,measured):
        previous=self.previous
        return (previous is not None and previous['axis']==axis and previous['sign']==sign
                and measured is not None and sign*(measured-previous['visual'])<MIN_STEP_PROGRESS)

    def apply(self,t,frame,position,vector,axis,sign,distance=None,measurements=None,yaw=None,soft_range=False):
        if not np.any(np.abs(vector)>1.e-5):
            if self.active is not None:self.stop(t)
            return vector
        if self.active is not None:
            if self.active['axis']!=axis or self.active['sign']!=sign:
                self.stop(t)
                self.reason='direction_change'
                return np.zeros(3)
            depth_restored=self.active.get('soft_forward_retry',False) and not soft_range
            if distance is not None and axis==0 and (distance<self.active['distance'] or depth_restored):
                # A new depth capture can shorten a running grant immediately.
                distance=min(distance,self.active['distance'])
                if np.linalg.norm(position-self.active['origin'])>=distance:
                    self.stop(t)
                    return np.zeros(3)
                self.active['distance']=distance
                speed=float(np.linalg.norm(vector[:2]))
                if not (soft_range and self.active['retry']):
                    start=self.active['until']-self.active['duration'] if depth_restored else t
                    self.active['duration']=min(self.active['duration'],distance/max(.1,speed))
                    self.active['until']=min(self.active['until'],start+self.active['duration'])
                    self.active['soft_forward_retry']=False
                    if t>=self.active['until']:
                        self.stop(t)
                        self.reason='await_zero_publication'
                        return np.zeros(3)
            return vector
        xy=frame.visual_xy if measurements is None else measurements
        measured=xy[axis] if axis<2 else None
        if axis<2 and measured is None:
            self.reason='missing_step_measurement'
            return np.zeros(3)
        if distance is None:
            distance=.046 if axis==1 else .063
        # Replan every step from the new observation. Poor response can justify
        # a longer pulse in the same direction, never a veto on the next step.
        retry=axis<2 and self.little_response(axis,sign,measured)
        if axis<2:
            self.previous=None
        # A geometric stop is evaluated on every policy publication as well
        # as at each plan. The independent time cap also bounds every grant
        # when the encoders are missing or the robot remains stationary.
        duration=.85 if axis==2 else .36 if axis==1 else .40
        if retry:
            duration+=.16
        # A short blind-patch step is a displacement target. If the first
        # pulse couldn't start the gait, keep its measured distance limit and
        # allow time for the repeat. An observed obstacle retains its time cap.
        soft_forward_retry=axis==0 and soft_range and retry
        if axis==0 and distance<.05 and not soft_forward_retry:
            duration=min(duration,distance/max(.1,np.linalg.norm(vector[:2])))
        self.active=dict(axis=axis,sign=sign,visual=measured,origin=position.copy(),
                         until=t+duration,duration=duration,distance=distance,sent=False,yaw=yaw,retry=retry,
                         frame_stamp=frame.stamp,soft_forward_retry=soft_forward_retry)
        self.count+=1
        self.reason='short_side_step' if axis==1 else 'short_forward_step' if axis==0 else 'short_turn'
        return vector
