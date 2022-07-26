#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu May 28 16:52:01 2020

@author: zhaoxm
"""
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from scipy.stats import chi2

from ..base import SingleObjectTracker
from statecircle.utils.common import rem2pi
from statecircle.types.state import GaussianLabeledState
from statecircle.datasets.base import SENSOR_TYPE

class EgoSLAMTracker:
    r""" Ego car EKF-SLAM tracker """
    def __init__(self, initial_pose, transition_model, measurement_model, estimator,
                 max_num_marks, gating_percentile=0.999, *args, **kwargs):
        self.state = None
        self.timestamp = None
        self.initial_pose = initial_pose
        self.birth_initialized = False
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.estimator = estimator
        self.max_num_marks = max_num_marks
        self.gating_thresh = chi2.ppf(gating_percentile, self.measurement_model.ndim_meas)
        
        self.stable_landmarks = []
    
    def birth(self):
        if not self.birth_initialized:
            ego_ndim = self.transition_model.ndim
            mean = np.array(self.initial_pose)
            cov = np.zeros((ego_ndim, ego_ndim))
            label = []
            self.state = GaussianLabeledState(label, mean, cov)
            self.birth_initialized = True
    
    def predict(self, meas_data):
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        
        self.timestamp, time_step = timestamp, timestamp - self.timestamp
        
        vel, yaw_rate = meas_data.ego_meas[:, 0]
        rx, ry, theta = self.state.mean[:3]
        self.state.mean[:3] = self.transition_model.forward(self.state.mean[:3], vel, yaw_rate, time_step)
        
        # Jacobian
        F = self.transition_model.transition_matrix(self.state.mean[:3], vel, yaw_rate, time_step)
        
        # TODO: debug the noise covariance matrix
        Q = self.transition_model.noise_covar(self.state.mean[:3, 0], time_step)
        
        cov_xx = self.state.cov[:3, :3]
        cov_xm = self.state.cov[:3, 3:]
    
        cov_pred = self.state.cov
        cov_pred[:3, :3] = F.dot(cov_xx).dot(F.T) + Q
        cov_pred[:3, 3:] = F.dot(cov_xm)
        cov_pred[3:, :3] = cov_xm.T.dot(F.T)
        
        self.state.cov = cov_pred
    
    def associate(self, meas_ids, mid_state):
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
        
    def update(self, meas_data):
        if meas_data.sensor_type in SENSOR_TYPE.CAMERA_TYPES:
            return
        
        # meas: [3, n], one meas: [dist, bearing, mid]
        meas = meas_data.range_bearing_id_meas
        # filter out motion object measurement
        # the object which velocity < 1.5 m/s are considered as static object
        vel = meas_data.ego_meas[0, 0]
        static_velocity_res = 1.0
        static_idx = np.abs(meas_data.meas[3] + vel) < static_velocity_res
        meas = meas[:, static_idx]
        
        rx, ry, theta = self.state.mean[:3, 0]
            
        num_meas = meas.shape[1]
        ndim = len(self.state.mean)
            
        if num_meas == 0:
            return
        
        # noise cov
        covar = self.measurement_model.noise_covar()
        R = np.kron(np.eye(num_meas), covar)

        zs = []
        zs_hat = []
        
        # data association
        meas_ids = meas[2].astype(np.int)
        assign = self.associate(meas_ids, self.state.label)
        
        ndim += (assign == -1).sum() * 2
        
        # Jacobian
        H = np.zeros((2 * num_meas, ndim))
        
        for i, mea in enumerate(meas.T):
            dist, bearing, mid = mea
            mid = int(mid)
            idx = assign[i]
            
            if idx == -1:
                current_state_dim = len(self.state.mean)
                # initialize the new landmark state
                self.state.append(mid)
                mx, my = self.measurement_model.reverse(mea[:2], self.state.mean[:3, 0])
                self.state.mean = np.vstack([self.state.mean, np.array([mx, my])[:, None]])
                
                # TODO: clean up
                __naive__ = False
                if __naive__:
                    # naive version: augment cov
                    aug_cov = 1000 * np.eye(current_state_dim + 2)
                    aug_cov[:current_state_dim, :current_state_dim] = self.state.cov
                    self.state.cov = aug_cov
                else:
                    # accurate version
                    # inverse measurement model
                    # Jacobian: partial meas_x, meas_y / x, y, theta 
                    Jxr = np.array([[1, 0, -dist * np.sin(theta + bearing)],
                                    [0, 1,  dist * np.cos(theta + bearing)]])
                    # Jacobian: partial meas_x, meas_y / dist, bearing
                    Jz = np.array([[np.cos(theta + bearing), -dist * np.sin(theta + bearing)],
                                   [np.sin(theta + bearing),  dist * np.cos(theta + bearing)]])
                    cov_xx = self.state.cov[:3, :3]
                    cov_mm = Jxr.dot(cov_xx).dot(Jxr.T) + Jz.dot(R[2*i:2*i+2, 2*i:2*i+2]).dot(Jz.T)
                    cov_xm = Jxr.dot(self.state.cov[:3, :])
                    
                    aug_cov = np.eye(current_state_dim + 2)
                    aug_cov[:current_state_dim, :current_state_dim] = self.state.cov
                    aug_cov[-2:, -2:] = cov_mm
                    aug_cov[-2:, :current_state_dim] = cov_xm
                    aug_cov[:current_state_dim, -2:] = cov_xm.T
                    self.state.cov = aug_cov
                    
                idx = self.state.num_label - 1
                
            zs.append(np.array([dist, bearing]))
            mx, my = self.state.mean[3+2*idx:3+2*idx+2, 0]
            delta = np.array([mx - rx, my - ry])
            dist_pred = np.linalg.norm(delta)
            bearing_pred = np.arctan2(delta[1], delta[0]) - theta
            zs_hat.append(np.array([dist_pred, bearing_pred]))
            
            # Jacobian for one landmark
            Hi = np.zeros((2, ndim))
            delta_x, delta_y = delta
            Hi[:2, :3] = np.array([[-dist_pred * delta_x, -dist_pred * delta_y,   0],
                                   [ delta_y,            -delta_x,              -dist_pred**2]]) / dist_pred**2
            Hi[:2, 3+2*idx:3+2*idx+2] = np.array([[dist_pred * delta_x, dist_pred * delta_y],
                                                  [-delta_y,            delta_x]]) / dist_pred**2
            Hi_cat = np.hstack([Hi[:2, :3], Hi[:2, 3+2*idx:3+2*idx+2]])
            cov_cat = np.vstack([
                                np.hstack([self.state.cov[:3, :3], 
                                           self.state.cov[:3, 3+2*idx:3+2*idx+2]
                                           ]),
                                np.hstack([self.state.cov[3+2*idx:3+2*idx+2, :3],
                                           self.state.cov[3+2*idx:3+2*idx+2, 3+2*idx:3+2*idx+2]])
                                ])
                    
            # gating
            Si = Hi_cat.dot(cov_cat).dot(Hi_cat.T) + R[2*i:2*i+2, 2*i:2*i+2]
            delta_zi = zs[-1] - zs_hat[-1]
            dist = delta_zi.dot(np.linalg.inv(Si)).dot(delta_zi)
            if dist < self.gating_thresh:
                H[2*i:2*i+2, :] = Hi
                
        zs = np.array(zs).ravel()
        zs_hat = np.array(zs_hat).ravel()
        delta_z = zs - zs_hat
        # rem2pi
        delta_z[1::2] = rem2pi(delta_z[1::2])
        
        S = H.dot(self.state.cov).dot(H.T) + R
        
        K = self.state.cov.dot(H.T).dot(np.linalg.inv(S))
        
        self.state.mean += K.dot(delta_z)[:, None]
        self.state.cov = (np.eye(ndim) - K.dot(H)).dot(self.state.cov)
        # rem2pi
        self.state.mean[2] = rem2pi(self.state.mean[2])
    
    def reduction(self):
        def trans_to_ego_coordiantes(pose, landmarks):
            # y -> front , x -> right, azimuth = theta - np.pi / 2
            theta = pose[2] - np.pi / 2
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            P = pose[:2]
            return R.T.dot(landmarks - P[:, None])
        
        
        # pruning the landmarks behind ego car
        pose = self.state.mean[:3, 0]
        landmarks = self.state.mean[3:, 0].reshape(-1, 2).T
        
        if landmarks.shape[1] == 0:
            return
        
        landmarks_ego = trans_to_ego_coordiantes(pose, landmarks)
        landmark_keep_idx = landmarks_ego[1] > 0
        keep_idx = np.hstack([[True] * 3, np.kron(landmark_keep_idx, np.array([True, True]))])
        
        self.stable_landmarks.append(self.state.mean[~keep_idx].reshape(-1, 2).T)
        
        self.state.mean = self.state.mean[keep_idx]
        self.state.cov = self.state.cov[np.ix_(keep_idx, keep_idx)]
        self.state.label = self.state.label[landmark_keep_idx]
        
        # capping
        nmarks = self.state.num_label
        ndim = 2 * self.max_num_marks
        if nmarks > self.max_num_marks:
            self.state.mean = np.vstack((self.state.mean[:3], self.state.mean[-ndim:]))
            reduce_cov = np.eye(3 + ndim)
            reduce_cov[:3, :3] = self.state.cov[:3, :3]
            reduce_cov[3:, 3:] = self.state.cov[-ndim:, -ndim:]
            self.state.cov = reduce_cov
            self.state.label = self.state.label[-self.max_num_marks:]
            
    def estimate(self, full_state=True):
        if full_state:
            return deepcopy(self.state)
        else:
            return deepcopy(self.estimator(self.state))
        
    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # covert km/h and degree/s to m/s and rad/s
            vel = meas_data.ego_info['Velocity'] / 3.6
            yaw_rate = meas_data.ego_info['YawRate'] * -np.pi / 180
            meas_data.ego_meas = np.array([vel, yaw_rate])[:, None]
            self.step = t
            # predict
            self.predict(meas_data)
            
            # birth
            self.birth()
                
            # update
            self.update(meas_data)
            
            # reduct
            self.reduction()

            # estimate
            estimates.append(self.estimate(True))

        return estimates
