#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sat May 30 13:53:12 2020

@author: zhaoxm
"""
import numpy as np
from tqdm import tqdm

from .md_tracker import MDTracker

class ProtoTypeTracker(MDTracker):
    r""" EKF SLAM ego tracker + PMBM tracker """
    def __init__(self, ego_tracker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ego_tracker = ego_tracker
    
    def ego_motion_update(self, meas_data):
        # covert km/h and degree/s to m/s and rad/s
        vel = meas_data.ego_info['Velocity'] / 3.6
        yaw_rate = meas_data.ego_info['YawRate'] * -np.pi / 180
        meas_data.ego_meas = np.array([vel, yaw_rate])[:, None]
        
        # ego predict
        self.ego_tracker.predict(meas_data)
        
        # ego birth
        self.ego_tracker.birth()
            
        # ego update
        self.ego_tracker.update(meas_data)
        
        # ego reduct
        self.ego_tracker.reduction()
        
        ego_pose = self.ego_tracker.state.mean[:3, 0] # [ego_x, ego_y, theta]
        landmarks = self.ego_tracker.state.mean[3:, 0].reshape(-1, 2).T # landmarks [x1, y1, ..., xn, yn]
        state_cov = self.ego_tracker.state.cov
        return ego_pose, landmarks, state_cov
        
    def filtering(self, data_reader, cut_latency_time=10):
        # tracks multiple objects using Poisson multi-Bernoulli mixture filter
        estimates = []
        ego_trace = []
        landmark_trace = []
        cov_trace = []
        for self.step, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # ego car EKF-SLAM
            ego_pose, landmarks, state_cov = self.ego_motion_update(meas_data)
            ego_trace.append(ego_pose)
            landmark_trace.append(landmarks)
            cov_trace.append(state_cov)
            
            # ego car motion compensation
            self.ego_motion_compensation(meas_data)
            
            # PMBM prediction
            self.predict(meas_data)
            
            # PMBM birth
            self.birth(meas_data)

            # PMBM update
            self.update(meas_data)
            
            # extract state estimates from PMBM
            estimates.append(self.estimate())

            # Bern recycling & reduction and PPP reduction
            self.reduction(cut_latency_time)
            
        return estimates, np.array(ego_trace), landmark_trace, cov_trace