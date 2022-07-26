#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 18 16:01:24 2019

@author: zhxm
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm import tqdm

from statecircle.trackers.base import MultiObjectTracker
from statecircle.utils.data_association import data_association


class GlobalNearestNeighbourTracker(MultiObjectTracker):
    """
    Global Nearest Neighbour tracker with known objects number
    
    The assumed density is Gaussian state
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Every local hypothesis tree contains only one local hypothesis for GNN tracker.
        self.hypo_forest = []
        # Number of object
        self.num_trees = 0
        
        self.birth_initialized = False

    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp
        
        for i, hypo in enumerate(self.hypo_forest):
            # NOTE: shallow copy
            # predict each local hypothesis
            hypo[0] = self.density_model.predict(hypo[0], time_step, self.transition_model)

    def birth(self, meas_data):
        if not self.birth_initialized:
            # initialize birth state
            self.hypo_forest = [[state] for state in self.birth_model.birth()]
            self.num_trees = self.num_objs = len(self.hypo_forest)
            self.birth_initialized = True
            
    def update(self, meas_data):
        meas = meas_data.meas
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter
        num_meas = meas.shape[1]
        num_objs = self.num_trees

        # create the cost matrix
        cost_mat = np.zeros([self.num_trees, num_meas + self.num_trees])
        for i in range(self.num_trees):
            hypo = self.hypo_forest[i][0]
            # ellipsoidal gating for each predicted local hypotheses separately
            meas_ingate, meas_index = self.gate.gating(hypo, meas, self.measurement_model)
            # construct 2D cost matrix of size (num_objs x (num_meas in gate + num_objs))
            pred_loglik = self.density_model.predicted_log_likelihood(hypo, meas_ingate, self.measurement_model)
            li0 = -np.inf * np.ones(num_objs)
            lij = -np.inf * np.ones(num_meas)
            li0[i] = np.log(1 - detection_rate)
            lij[meas_index] = np.log(detection_rate) - np.log(intensity_clutter) + pred_loglik
            cost_mat[i, :] = np.hstack((-lij, -li0))

        # TODO: simplify the cost matrix
        # delete the missed detected cols from cost matrix
        #            detected_cols = np.any(cost_mat != inf, 0)
        #            associated_num_meas = cost_mat.shape[1] - num_objs
        #            cost_mat = cost_mat[:, detected_cols]

        # find the best assignment matrix using a 2D assignment solver
        __use_hungarian_algorithm = False
        if __use_hungarian_algorithm:
            row_idx, col_idx = linear_sum_assignment(cost_mat)
        else:
            col_idx, cost = data_association(cost_mat, topN=1)
        theta_t = col_idx
        # theta == -1 means miss detection
        theta_t[theta_t > num_meas - 1] = -1

        # create new local hypothesis according to the best assignmet matrix obtained
        for i in range(num_objs):
            if theta_t[i] >= 0:
                self.hypo_forest[i][0] = self.density_model.update(self.hypo_forest[i][0], meas[:, theta_t[i]], self.measurement_model)

    def estimate(self):
        # extract object state estimates
        est_t = [self.estimator(hypo[0]) for hypo in self.hypo_forest]
        return np.hstack(est_t)

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
        
