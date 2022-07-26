#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 18 13:58:56 2019

@author: zhxm
"""
import numpy as np
from tqdm import tqdm

from ..base import SingleObjectTracker
from statecircle.utils.common import normalize_log_weights


class GaussianSumTracker(SingleObjectTracker):
    r"""Gaussian sum filter for single object
    
    In fact, the assumed density in `GaussianSumTracker` is a Gaussian mixture state,
    not a Gaussian sum intensity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: use specified hypothesis type
        self.hypo_tree = []
        self.log_weights = np.empty([0])
        self.birth_initialized = False

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp
        
        # for each hypothesis, perform prediction
        self.hypo_tree = [self.density_model.predict(hypo, time_step, self.transition_model)
                          for hypo in self.hypo_tree]

    def birth(self, meas_data):
        if not self.birth_initialized:
            self.hypo_tree = [self.birth_model.birth()]
            self.log_weights = np.zeros(1)
            self.birth_initialized = True
        
    def update(self, meas_data):
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter
        log_weights, hypo_tree = [], []
        for i, hypo in enumerate(self.hypo_tree):
            meas_ingate, meas_index = self.gate.gating(hypo, meas, self.measurement_model)
            num_meas_ingate = meas_ingate.shape[1]

            # create missed detection hypothesis for each hypothesis
            hypo_tree.append(hypo)
            log_weights.append(self.log_weights[i] + np.log(1 - detection_rate))
            
            if num_meas_ingate > 0:
                # create object detection hypothesis for each detection inside the gate
                pred_loglik = self.density_model.predicted_log_likelihood(hypo, meas_ingate, self.measurement_model)
                log_weights.append(self.log_weights[i] + np.log(detection_rate) + pred_loglik - np.log(intensity_clutter))
                for k in range(num_meas_ingate):
                    hypo_tree.append(self.density_model.update(hypo, meas_ingate[:, k], self.measurement_model))
        log_weights = np.hstack(log_weights)
        
        # normalize hypothesis weights
        log_weights, log_suw_weihts = normalize_log_weights(log_weights)

        # return
        self.log_weights = log_weights
        self.hypo_tree = hypo_tree

    def estimate(self):
        # extract object state estimate using the most probably hypothesis estimation
        max_idx = np.argmax(self.log_weights)
        return self.estimator(self.hypo_tree[max_idx])

    def reduction(self):
        # prune hypothesis with samll weights, and then re-normalize the weights
        log_weights, hypo_tree = self.reductor.prune(self.log_weights, self.hypo_tree)
        log_weights, log_sum_w = normalize_log_weights(log_weights)

        # hypothesis merging
        log_weights, hypo_tree = self.reductor.merge(log_weights, hypo_tree)

        # cap the number of the hypothesis, and then re-normalize the weights
        log_weights, hypo_tree = self.reductor.cap(log_weights, hypo_tree)
        log_weights, log_suw_weihts = normalize_log_weights(log_weights)

        # return
        self.log_weights = log_weights
        self.hypo_tree = hypo_tree

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            #predict
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
