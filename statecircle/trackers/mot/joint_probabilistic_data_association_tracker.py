#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec 19 10:39:45 2019

@author: zhaoxm
"""
import numpy as np
from scipy.misc import logsumexp
from tqdm import tqdm

from statecircle.trackers.base import MultiObjectTracker
from statecircle.utils.data_association import data_association
from statecircle.utils.common import normalize_log_weights

class JointProbabilisticDataAssociationTracker(MultiObjectTracker):
    """
    Joint Probability Data Association Filter with known objects number
    
    The assumed density is Gaussian state
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hypotheses forest, every term in hypotheses forest is a local hypotheses
        # tree, each tree has some local hypotheses (leaves)
        # each tree only contains one leaf in JPDA tracker
        self.hypo_forest = []
        # Number of local hypotheses trees / object
        self.num_trees = self.num_objs = 0
        
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
        
        cost_mat = np.zeros((self.num_objs, num_meas + self.num_objs))
        for i in range(self.num_objs):
            hypo = self.hypo_forest[i][0]
            # 1.ellipsoidal gating for each predict local hypothesis seperately
            [meas_ingate, meas_index] = self.gate.gating(hypo, meas, self.measurement_model)
            
            # 2.construct 2D cost matrix of size (num_meas in gate + self.num_objs)
            pred_loglik = self.density_model.predicted_log_likelihood(hypo, meas_ingate, self.measurement_model)
            li0 = -np.inf * np.ones(self.num_objs)
            lij = -np.inf * np.ones(num_meas)
            li0[i] = np.log(1 - detection_rate)
            lij[meas_index] = np.log(detection_rate) - np.log(intensity_clutter) + pred_loglik
            cost_mat[i,:] = np.hstack((-lij, -li0))
        
        # TODO: simplify the cost matrix
        # 3.find the M best assignment matrices using a M-best 2D assignment solver
        col, cost = data_association(cost_mat, self.reductor.capping_num)
        theta_t = col
        log_weights = -cost
        
        # 4.normalize the weights of different data association hypotheses
        log_weights, log_sum_weights = normalize_log_weights(log_weights)
        
        # 5.prune assignment matrices that correspond to data association hypotheses 
        # with low weights and renormalize the weights
        log_weights, keep_idx = self.reductor.prune(log_weights, np.arange(theta_t.shape[1]))
        log_weights, log_sum_weights = normalize_log_weights(log_weights)
        theta_t = theta_t[:,keep_idx]
        # '-1' means missed detection hypothesis
        theta_t[theta_t > num_meas - 1] = -1
        for i in range(self.num_objs):
            hypo = self.hypo_forest[i][0]
            # 6.create new local hypotheses for each of the data asspcoatopm results
            log_marginal_weights = []
            multi_hypo = []
            for it, j in enumerate(np.unique(theta_t[i, :])):
                meas_idx_i = theta_t[i,:] == j
                if np.any(meas_idx_i):
                    multi_hypo.append(hypo if j == -1 \
                                      else self.density_model.update(hypo, meas[:,j], self.measurement_model))
                    log_marginal_weights.append(logsumexp(log_weights[meas_idx_i]))
                    
            # 7.merge local hypotheses theta correspond to the same object by moment matching
            self.hypo_forest[i][0] = self.reductor.moment_matching(log_marginal_weights, multi_hypo)

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