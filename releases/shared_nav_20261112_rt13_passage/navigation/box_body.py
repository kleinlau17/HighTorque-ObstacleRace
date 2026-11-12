"""Extremities of the calibrated Pro, in AHRS axes relative to the trunk."""
import numpy as np

from navigation.leg_odometry import _feet, _rotation, LEG_JOINTS


def limb_points(joints,attitude):
    """Sphere centers and radii for toes, heels, arm and wrist capsules."""
    try:
        feet,_=_feet(np.array([joints[name] for name in LEG_JOINTS]),attitude)
        samples=[feet.reshape(-1,3)]
        radii=[np.full(feet.size//3,.010)]
        for side,sign in [('l',1.),('r',-1.)]:
            origin=np.zeros(3)
            orientation=attitude
            for name,axis,offset in [('shoulder_pitch',1,[0.,sign*.1084,.117]),
                                     ('shoulder_roll',0,[0.,sign*.038,0.]),
                                     ('upper_arm',2,[0.,0.,-.0724]),
                                     ('elbow',1,[0.,0.,-.038])]:
                origin=origin+orientation@offset
                orientation=orientation@_rotation(axis,joints[f'{side}_{name}_joint'])
            # Elbow sphere, forearm capsule ends and the distal wrist sphere.
            samples.append(np.array([origin-distance*orientation[:,2]
                                     for distance in (.050,.054,.104,.126)]))
            radii.append(np.array([.0298,.020,.020,.0298]))
    except (KeyError,ValueError):
        return None
    return np.column_stack([np.vstack(samples),np.concatenate(radii)])


def station_window(body,basis,position,target_y,back,front,trailing_foot_sweep=False):
    """Limit a cross-lane step using the foot and arm pose read this tick."""
    points=body[:,:2]@basis
    margin=.009
    radii=body[:,3]+margin
    # Allow for up to one short lateral step and residual gait travel.
    lookahead=np.clip(target_y-position[1],-.055,.055)
    low,high=-np.inf,np.inf
    for obstacle,behind in [(back,True),(front,False)]:
        travel=np.full(len(points),lookahead)
        if behind and trailing_foot_sweep:
            travel[:16]=target_y-position[1]
        low_y=position[1]+points[:,1]+np.minimum(0.,travel)
        high_y=position[1]+points[:,1]+np.maximum(0.,travel)
        dy=np.maximum(0.,np.maximum(obstacle[1]-.10-high_y,low_y-obstacle[1]-.10))
        near=dy<radii
        if not np.any(near):
            continue
        extension=np.sqrt(radii[near]**2-dy[near]**2)
        if behind:
            low=np.max(obstacle[0]+.10-points[near,0]+extension)
        else:
            high=np.min(obstacle[0]-.10-points[near,0]-extension)
    return low,high
