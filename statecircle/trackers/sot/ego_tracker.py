#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun May 24 18:31:19 2020

@author: zhaoxm
"""
from tqdm import tqdm
import numpy as np

from ..base import SingleObjectTracker
from statecircle.types.state import GaussianState


class EgoTracker(SingleObjectTracker):
    r""" ego tracker
    
    The assumed density is Gaussian state
    """
    def __init__(self, meas_models_dict, clutter_models_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cyclic state in sot
        self.state = None
        self.birth_initialized = False
        self.meas_models_dict = meas_models_dict
        self.clutter_models_dict = clutter_models_dict

    def update(self, meas_data):
        # TODO: reformat the update input parameters, hard coding for now...
        meas = meas_data.ego_meas
        
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        meas_ingate, meas_index = self.gate.gating(self.state, meas, self.measurement_model)
        pred_loglik = self.density_model.predicted_log_likelihood(self.state, meas_ingate, self.measurement_model)

        num_meas_ingate = meas_ingate.shape[1]
        log_weights_unnorm = np.empty(num_meas_ingate + 1)
        log_weights_unnorm[0] = np.log(1 - detection_rate)
        log_weights_unnorm[1:] = np.log(detection_rate) + pred_loglik - np.log(intensity_clutter)

        theta_max = np.argmax(log_weights_unnorm)
        if theta_max == 0:
            # missed, not update state
            pass
        else:
            # detected
            self.state = self.density_model.update(self.state, meas_ingate[:, theta_max - 1], self.measurement_model)

    def estimate(self):
        return self.estimator(self.state)

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp
        
        self.state = self.density_model.predict(self.state, time_step, self.transition_model)
    
    def birth(self, meas_data):
        if not self.birth_initialized:
            # initialize birth state
            vel, yaw_rate = meas_data.ego_meas[0, 0], meas_data.ego_meas[1, 0]
            self.state = GaussianState(mean=np.array([[0, 0, vel, 0, yaw_rate]]).T, cov=np.zeros((5,5)))
            self.birth_initialized = True
        # TODO: add radar meas
        # use UNKNOWN measurement model first
        self.measurement_model = self.meas_models_dict[-1]
        self.clutter_model = self.clutter_models_dict[-1]

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # covert km/h and degree/s to m/s and rad/s
            vel = meas_data.ego_info['Velocity'] / 3.6
            yaw_rate = meas_data.ego_info['YawRate'] * -np.pi / 180
            meas_data.ego_meas = np.array([vel, yaw_rate])[:, None]
            
            # predict
            self.predict(meas_data)
            
            # birth
            self.birth(meas_data)
                
            # gating & update
            self.update(meas_data)

            # estimate
            estimates.append(self.estimate())

        return estimates
