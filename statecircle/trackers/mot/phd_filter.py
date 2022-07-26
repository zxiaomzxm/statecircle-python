#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec 19 18:17:12 2019

@author: zhaoxm
"""
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from statecircle.trackers.base import MultiObjectFilter
from statecircle.types.state import GaussianSumState, PoissonState


class PHDFilter(MultiObjectFilter):
    r"""Gaussian mixture intensity probability hypothesis density filter
    
    The Gaussian mixture is a intensity (NOT density) function in PHD, and the assumed
    density in PHD is Poisson random finite sets(PRFS) or Poisson point process(PPP)
    """

    def __init__(self, surviving_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.surviving_rate = surviving_rate
        self.poisson_state = PoissonState(GaussianSumState())

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # predict performs PPP predicion step
        # predict each Gaussian component in the intensity for pre-existing objects
        log_weights = np.log(self.surviving_rate) + self.poisson_state.intensity.log_weights
        gaussian_states = [self.density_model.predict(state, time_step, self.transition_model)
                           for state in self.poisson_state.intensity.gaussian_states]
        self.poisson_state.intensity = GaussianSumState(log_weights, gaussian_states)
        
    def birth(self, meas_data):
        # add (Gaussian mixture) Poisson birth intensity to (Gaussian mixture)
        # Poisson intensity for pre-existing objects
        birth_state = self.birth_model.birth().intensity
        self.poisson_state.intensity += deepcopy(birth_state)

    def update(self, meas_data):
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter
        num_comps = self.poisson_state.intensity.num_comps
        num_meas = meas.shape[1]

        # 1) construct update components resulted from missed detection
        log_weights_missed = np.log(1 - detection_rate) + self.poisson_state.intensity.log_weights
        intensity_missed = GaussianSumState(log_weights_missed, self.poisson_state.intensity.gaussian_states)

        # 2) perform ellipsoidal gating for each Gaussian component in
        #    the Poisson intensity
        gating_mat = np.zeros((num_comps, num_meas), dtype=np.bool)
        for i, state in enumerate(self.poisson_state.intensity.gaussian_states):
            _, meas_index = self.gate.gating(state, meas, self.measurement_model)
            gating_mat[i, :] = meas_index

        meas_in_all_gates = np.any(gating_mat != 0, axis=0)
        meas = meas[:, meas_in_all_gates]
        gating_mat = gating_mat[:, meas_in_all_gates]
        num_meas = meas.shape[1]

        # 3) construct update components resulted from object
        #    detections that are inside the gates
        states_update = []
        log_weights_update = []
        for i in range(num_meas):
            log_weights_unnorm = []
            for h, state in enumerate(self.poisson_state.intensity.gaussian_states):
                if gating_mat[h, i]:
                    # 3) construct update components resulted from object
                    #    detections that are inside the gates
                    pred_loglik = self.density_model.predicted_log_likelihood(state,
                                                                              meas[:, i],
                                                                              self.measurement_model)
                    
                    # TODO: optimization for the update parameters without measurement input
                    # TODO: (mean, covariance for measurement, Kalman gain and updated covariance)
                    states_update.append(self.density_model.update(state, meas[:, i], self.measurement_model))
                    log_weights_unnorm.append(np.log(detection_rate) + self.poisson_state.intensity.log_weights[h] + pred_loglik)

            log_weights_update.append(log_weights_unnorm - \
                                      np.log(intensity_clutter + np.sum(np.exp(log_weights_unnorm))))
        if len(log_weights_update) > 0:
            intensity_update = GaussianSumState(np.hstack(log_weights_update), states_update)
        else: # there is no meansurement in all gates
            intensity_update = GaussianSumState()

        self.poisson_state.intensity = intensity_missed + intensity_update

    def reduction(self):
        # component reduction approximates the PPP by representing its intensity
        # with fewer parameters
        reduction_function_handles = (self.reductor.prune, self.reductor.merge, self.reductor.cap)
        for func in reduction_function_handles:
            self.poisson_state.intensity = GaussianSumState(*func(self.poisson_state.intensity.log_weights,
                                                             self.poisson_state.intensity.gaussian_states))

    def estimate(self):
        # PHD estimator performs object state estimation in the GMPHD filter

        # 1) get a mean estimate of the cadinality of objects by
        # taking the summation of the weights of the Gaussian
        # components (rounded to the nearest integer), denoted as n
        card_mean = np.int(np.minimum(self.poisson_state.intensity.num_comps,
                                      np.round(self.poisson_state.mean)
                                      )
                           )

        # 2) extract n object states from the means of the n Gaussian components
        # with the hightest weights
        keep_idx = np.argsort(-self.poisson_state.intensity.log_weights)[:card_mean]
        est = [self.poisson_state.intensity.gaussian_states[idx].mean for idx in keep_idx]
        
        if len(est) == 0:
            return np.empty((self.transition_model.ndim, 0))
                            
        return np.hstack(est)

    def estimate_duplicated(self, *args, **kwargs):
        raise NotImplementedError

    def filtering(self, data_reader):
        estimates = []
        for t, meas_data in tqdm(enumerate(data_reader), total=len(data_reader)):
            # PPP prediction
            self.predict(meas_data)
            
            # PPP birth
            self.birth(meas_data)

            # PPP update
            self.update(meas_data)

            # extract state estimates from PPP
            estimates.append(self.estimate())
            
            # PPP approximation
            self.reduction()

        return estimates
