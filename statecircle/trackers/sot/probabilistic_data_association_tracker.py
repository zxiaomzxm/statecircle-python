#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 18 11:01:58 2019

@author: zhxm
"""
import numpy as np
from tqdm import tqdm

from ..base import SingleObjectTracker
from statecircle.utils.common import normalize_log_weights


class ProbabilisticDataAssociationTracker(SingleObjectTracker):
    r""" Single object tracker using probabilistic data association 
    
    The assumed density is Gaussian state
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cyclic state in sot
        self.state = None
        self.birth_initialized = False

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
            self.state = self.birth_model.birth()
            self.birth_initialized = True
            
    def update(self, meas_data):
        r"""
        Parameters
        ----------
        state : `State`
        meas : Array, [meas_dim, 1]
        """
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        meas_ingate, meas_index = self.gate.gating(self.state, meas, self.measurement_model)
        pred_loglik = self.density_model.predicted_log_likelihood(self.state, meas_ingate, self.measurement_model)

        num_meas_ingate = meas_ingate.shape[1]
        log_weights_unnorm = np.empty(num_meas_ingate + 1)

        # generate hypothesis tree
        hypo_tree = []

        # missed detection hypothesis
        log_weights_unnorm[0] = np.log(1 - detection_rate)
        hypo_tree.append(self.state)

        # object detection hypothesis
        log_weights_unnorm[1:] = np.log(detection_rate) + pred_loglik - np.log(intensity_clutter)
        for i in range(1, num_meas_ingate + 1):
            hypo_tree.append(self.density_model.update(self.state, meas_ingate[:, i - 1], self.measurement_model))

        # normalize hypothesis weights
        log_weights, log_sum_weights = normalize_log_weights(log_weights_unnorm)

        # prune hypothesis
        log_weights, hypo_tree = self.reductor.prune(log_weights, hypo_tree)

        # re-normalize
        log_weights, log_sum_w = normalize_log_weights(log_weights)

        # moment matching
        self.state = self.reductor.moment_matching(log_weights, hypo_tree)

    def estimate(self):
        return self.estimator(self.state)

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # predict
            self.predict(meas_data)
                
            # birth
            self.birth(meas_data)
            
            # update
            self.update(meas_data)

            # estimate
            estimates.append(self.estimate())

            # reduction
            self.reduction()

        return estimates
