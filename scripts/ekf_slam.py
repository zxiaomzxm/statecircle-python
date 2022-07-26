#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed May 27 14:56:34 2020

@author: zhaoxm
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from pytest import approx
from tqdm import tqdm

#from statecircle.lib.harem.debugger import Debugger
#dprint = Debugger.dprint

class GaussianState:
    def __init__(self, mean, cov, meas_id=[]):
        self.mean = mean
        self.cov = cov
        self.meas_id = np.array(meas_id)
        
    @property
    def ndim(self):
        return len(self.mean)
    
    @property
    def num_marks(self):
        return len(self.meas_id)

class Measurement:
    def __init__(self, dist, bearing, mid):
        self.dist = dist 
        self.bearing = bearing
        self.mid = mid

def rem2pi(x):
    return x - 2 * np.pi * np.round(x / 2 / np.pi)

def draw_2d_robot(ax, pose):
    x, y, theta = pose
    x_ = x + 0.3 * np.cos(theta)
    y_ = y + 0.3 * np.sin(theta)
    ax.plot(x, y, 'bo')
    ax.plot([x, x_], [y, y_], 'b')


def transition_model(pose, vel, yaw_rate, dt):
    x, y, theta = pose
    x += vel * dt * np.cos(theta)
    y += vel * dt * np.sin(theta)
    theta += yaw_rate * dt
    # rem2pi
    theta = rem2pi(theta)
    return np.array([x, y, theta])


def measurement_model(pose, mea):
    dist, bearing, _ = mea
    rx, ry, theta = pose
    mx = rx + dist * np.cos(bearing + theta)
    my = ry + dist * np.sin(bearing + theta)
    return np.array([mx, my])


def belief_init(initial_pose):
    ego_ndim = 3
    mean = np.nan * np.zeros((ego_ndim))
    mean[:3] = initial_pose
    cov = np.zeros((ego_ndim, ego_ndim))
    return GaussianState(mean, cov)


def predict(belief, vel, yaw_rate, dt, var):
    # var: [var_x, var_y, var_theta]
    
    belief = deepcopy(belief)
    rx, ry, theta = belief.mean[:3]
    belief.mean[:3] = transition_model([rx, ry, theta], vel, yaw_rate, dt)
    
    # Jacobian
    F = np.array([[1, 0, -vel * dt * np.sin(theta)],
                  [0, 1,  vel * dt * np.cos(theta)],
                  [0, 0,  1]
                  ])
    
    # noise
    Q = np.diag(var)
    
    cov_xx = belief.cov[:3, :3]
    cov_xm = belief.cov[:3, 3:]

    cov_pred = belief.cov
    cov_pred[:3, :3] = F.dot(cov_xx).dot(F.T) + Q
    cov_pred[:3, 3:] = F.dot(cov_xm)
    cov_pred[3:, :3] = cov_xm.T.dot(F.T)
    
    belief.cov = cov_pred
    
    return belief

def associate(meas_ids, mid_state):
    # mid_state: current meas id in state
    # meas_ids: current measurement id
    # return assignament vector, 
    # assign[i] = j stands for the `i`th landmark
    # is associate with `j`th mid_state.
    # assign[i] = -1 means new landmark
    assign = -1 * np.ones_like(meas_ids)
    for i, mid in enumerate(meas_ids):
        idx = np.argwhere(mid == mid_state)
        if len(idx) > 0:
            assign[i] = idx[0,0]
    return assign
    
def update(belief, meas, var):
    belief = deepcopy(belief)
    # meas: [n, 3], one meas: [dist, bearing, mid]
    rx, ry, theta = belief.mean[:3]
    
    num_meas = len(meas)
    ndim = len(belief.mean)
    
    if num_meas == 0:
        return belief
    
    # noise cov
    R = np.kron(np.eye(num_meas), np.diag(var))
    
    zs = []
    zs_hat = []
    
    # data association
    meas_ids = [int(mea[2]) for mea in meas]
    assign = associate(meas_ids, belief.meas_id)
    
    ndim += (assign == -1).sum() * 2
    
    # Jacobian
    H = np.zeros((2 * num_meas, ndim))
    
    for i, mea in enumerate(meas):
        dist, bearing, mid = mea
        mid = int(mid)
        
        idx = assign[i]
        if idx == -1:
            current_state_dim = len(belief.mean)
            # initialize the new landmark state
            belief.meas_id = np.append(belief.meas_id, mid)
            mx, my = measurement_model([rx, ry, theta], mea)
            belief.mean = np.hstack([belief.mean, [mx, my]])
            
            __naive__ = False
            if __naive__:
                # naive version: augment cov
                aug_cov = 1000 * np.eye(current_state_dim + 2)
                aug_cov[:current_state_dim, :current_state_dim] = belief.cov
                belief.cov = aug_cov
            else:
                # accurate version
                # Jacobian: \partial meas_x, meas_y / x, y, theta 
                Jxr = np.array([[1, 0, -dist * np.sin(theta + bearing)],
                                [0, 1,  dist * np.cos(theta + bearing)]])
                # Jacobian: \partial meas_x, meas_y / dist, bearing
                Jz = np.array([[np.cos(theta + bearing), -dist * np.sin(theta + bearing)],
                               [np.sin(theta + bearing),  dist * np.cos(theta + bearing)]])
                cov_xx = belief.cov[:3, :3]
                cov_mm = Jxr.dot(cov_xx).dot(Jxr.T) + Jz.dot(R[2*i:2*i+2, 2*i:2*i+2]).dot(Jz.T)
                cov_xm = Jxr.dot(belief.cov[:3, :])
                
                aug_cov = np.eye(current_state_dim + 2)
                aug_cov[:current_state_dim, :current_state_dim] = belief.cov
                aug_cov[-2:, -2:] = cov_mm
                aug_cov[-2:, :current_state_dim] = cov_xm
                aug_cov[:current_state_dim, -2:] = cov_xm.T
                belief.cov = aug_cov
                
                
            idx = belief.num_marks - 1
            
        zs.append([dist, bearing])
        mx, my = belief.mean[3+2*idx:3+2*idx+2]
        delta = [mx - rx, my - ry]
        dist_pred = np.linalg.norm(delta)
        theta_pred = np.arctan2(delta[1], delta[0]) - theta
        zs_hat.append([dist_pred, theta_pred])
        
        # Jacobian for one landmark
        Hi = np.zeros((2, ndim))
        delta_x, delta_y = delta
        Hi[:2, :3] = np.array([[-dist_pred * delta_x, -dist_pred * delta_y,   0],
                               [ delta_y,            -delta_x,              -dist_pred**2]]) / dist_pred**2
        Hi[:2, 3+2*idx:3+2*idx+2] = np.array([[dist_pred * delta_x, dist_pred * delta_y],
                                              [-delta_y,            delta_x]]) / dist_pred**2
        H[2*i:2*i+2, :] = Hi
        
    zs = np.array(zs).ravel()
    zs_hat = np.array(zs_hat).ravel()
    delta_z = zs - zs_hat
    # rem2pi
    delta_z[1::2] = rem2pi(delta_z[1::2])
    
    K = belief.cov.dot(H.T).dot(np.linalg.inv(H.dot(belief.cov).dot(H.T) + R))
    
    belief.mean += K.dot(delta_z)
    belief.cov = (np.eye(ndim) - K.dot(H)).dot(belief.cov)
    # rem2pi
    belief.mean[2] = rem2pi(belief.mean[2])
    
    return belief

def reduct(state, max_num_marks):
    # pruning
    nmarks = state.num_marks
    ndim = 2 * max_num_marks
    if nmarks > max_num_marks:
        state.mean = np.append(state.mean[:3], state.mean[-ndim:])
        reduce_cov = np.eye(3 + ndim)
        reduce_cov[:3, :3] = state.cov[:3, :3]
        reduce_cov[3:, 3:] = state.cov[-ndim:, -ndim:]
        state.cov = reduce_cov
        state.meas_id = state.meas_id[-max_num_marks:]

def precise_transition_model(pose, vel, yaw_rate, dt):
    rx, ry, theta = pose
    if yaw_rate == approx(0):
        rx += vel * dt * np.cos(theta)
        ry += vel * dt * np.sin(theta)
    else:
        rx += vel / yaw_rate * (np.sin(theta + yaw_rate * dt) - np.sin(theta)) 
        ry += vel / yaw_rate * (np.cos(theta) - np.cos(theta + yaw_rate * dt))
    theta += yaw_rate * dt
    return [rx, ry, theta]
    
def generate_dataset(landmarks, std, initial_pose=[6, 0, 0], dt=0.01, T=100, scope=3, noise=True):
    # cos: [2, 2]
    # landmarks: [n, 3] for [mx, my, id]
    timestamps = np.arange(0, T, dt)
    vels = 0.5 * np.ones_like(timestamps)
    yaw_rates = 0.1 * np.ones_like(timestamps)
    pose = initial_pose
    meas = []
    poses = []
    for step in range(len(timestamps)):
        poses.append(pose)
        rx, ry, theta = pose
        # find the landmarks inside the agent's scope
        dists = np.linalg.norm(landmarks[:,:2] - [rx, ry], axis=1)
        inside_idx = dists < scope
        mark_inside = landmarks[inside_idx]
        dists = dists[inside_idx]
        meas_frame = np.zeros((len(dists), 3))
        for i, (mark, dist) in enumerate(zip(mark_inside, dists)):
            mx, my, mid = mark
            dist += std[0] * np.random.randn()
            bearing = np.arctan2(my - ry, mx - rx) - theta + std[1] * np.random.randn()
            meas_frame[i] = [dist, bearing, mid]
        meas.append(meas_frame)
            
        pose = precise_transition_model(pose, vels[step], yaw_rates[step], dt)
        
    if noise:
        # TODO: view as parameters
        std_vel = 0.3
        std_yaw_rate = 0.3
        return vels + std_vel * np.random.randn(*vels.shape), \
                yaw_rates + std_yaw_rate * np.random.randn(*vels.shape), \
                meas, \
                poses
    else:
        return vels, yaw_rates, meas, poses
        
#%%
#np.random.seed(4)

initial_pose = [6, 0, 0]
scope = 12
vision_scope = 6
num_landmarks = 9
max_num_marks = 7

transition_var = [0.1, 0.1, 0.01] # var_x, var_y, var_theta
measure_var = [0.1, 0.1] # var_x, var_y

dt = 0.1
T = 50
landmarks = scope * np.random.rand(num_landmarks, 2)
landmarks = np.hstack((landmarks, np.arange(num_landmarks)[:, None]))

fix, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(landmarks[:, 0], landmarks[:, 1], 'r*')
draw_2d_robot(ax, [6, 0, 0])
plt.axis('equal')

#%%
belief = belief_init(initial_pose)

# generate odometry & measurements
vels, yaw_rates, meas, poses = generate_dataset(initial_pose=initial_pose, dt=dt, T=T, std=[0.02, 0.02], landmarks=landmarks, scope=vision_scope)

#%%
believes = []
timestamps = np.arange(0, T, dt)
for step in tqdm(range(len(timestamps))):
    belief = update(belief, meas[step], measure_var)
    reduct(belief, max_num_marks)
    believes.append(belief)
    belief = predict(belief, vels[step], yaw_rates[step], dt, transition_var)

    
#%% plot
gt_track = np.array(poses)
ego_track = np.array([belief.mean[:3] for belief in believes])

ax.plot(ego_track[:, 0], ego_track[:, 1])
ax.plot(gt_track[:, 0], gt_track[:, 1], '--')
plt.axis('square')
plt.axis([-5, 16, -5, 16])

#%%
#for i, belief in enumerate(believes):
#    cov = belief.cov
##    cov[cov==1000]=0
#    cov_x = cov[::2,::2]
#    cov_y = cov[1::2,1::2]
#    
#    fig, ax = plt.subplots(1, 1)
#    ax.imshow(cov, vmin=cov.min(), vmax=cov.max())
#    ax.set_title('step: {}, covariace matrix'.format(i))
#    plt.show()
#    print('')
#    plt.close('all')
#%%
plt.figure()
num_marks = [ele.num_marks for ele in believes]
plt.plot(num_marks)
