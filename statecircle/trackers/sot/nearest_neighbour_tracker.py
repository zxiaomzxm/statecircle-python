#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 15:07:05 2019

@author: zhxm
"""
from tqdm import tqdm
import numpy as np

from ..base import SingleObjectTracker


class NearestNeighbourTracker(SingleObjectTracker):
    r""" Nearest neighbour tracker 
    
    The assumed density is Gaussian state
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cyclic state in sot
        self.state = None
        self.birth_initialized = False

    def update(self, meas_data):
        meas = meas_data.meas
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
            self.state = self.birth_model.birth()
            self.birth_initialized = True

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # predict
            self.predict(meas_data)
            
            # birth
            self.birth(meas_data)
                
            # gating & update
            self.update(meas_data)

            # estimate
            estimates.append(self.estimate())

        return estimates
