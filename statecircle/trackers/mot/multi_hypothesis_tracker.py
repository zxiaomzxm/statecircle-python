#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec 19 13:43:26 2019

@author: zhaoxm
"""
import numpy as np
from tqdm import tqdm

from statecircle.trackers.base import MultiObjectTracker
from statecircle.utils.data_association import data_association
from statecircle.utils.common import normalize_log_weights


class TrackOrientedMultiHypothesisTracker(MultiObjectTracker):
    """
    Track Oriented Multi Hypothesis Tracker(TO-MHT) with known number of objects
    """
    # TODO: MHT have unnormal trajectory switch occasionally
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hypotheses forest, every term in hypotheses forest is a local
        # hypotheses tree, each tree has some local hypotheses (leaves)
        self.hypo_forest = []
        
        # Number of local hypotheses trees / object
        self.num_trees = self.num_objs = 0
        
        # global hypotheses look-up table, 
        # has size: [#global hypothesis, #local hypotheses tree]
        self.hypo_table = np.empty((0, self.num_trees))
        
        # global hypothesis weights
        self.global_log_weights = np.empty(0)
        
        self.birth_initialized = False
        
    def predict(self, meas_data):
        # TODO: implenment time_step using datetime
        timestamp = meas_data.timestamp
        if self.timestamp is None:
            self.timestamp = timestamp
            return
        self.timestamp, time_step = timestamp, timestamp - self.timestamp

        # predict each local hypothesis in each hypothesis tree
        self.hypo_forest = [[self.density_model.predict(hypo, time_step, self.transition_model)
                             for hypo in hypo_tree]
                            for hypo_tree in self.hypo_forest]
    
    def birth(self, meas_data):
        if not self.birth_initialized:
            # initialize birth state
            self.hypo_forest = [[state] for state in self.birth_model.birth()]
            self.num_trees = self.num_objs = len(self.hypo_forest)
            num_global_hypo = 1
            self.hypo_table = np.zeros((num_global_hypo, self.num_trees), dtype=np.int)
            self.global_log_weights = np.zeros(num_global_hypo)
            self.birth_initialized = True

    def update(self, meas_data):
        """local hypothese (self.hypo_forest) update
        for each local hypothesis in each hypothesis tree
           1) ellipsoidal gating
           2) calculate missed detection and predicted likelihood for each
              measurement inside the gate
           3) create updated local hypothesis
        """
        meas = meas_data.meas
        num_meas = meas.shape[1]
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter

        # input: self.hypo_forest: [#object/tree][#local hypothesis] `State`
        # output: hypo_forest: [#object/tree][#local hypothesis * (num_meas_ingate + 1)] `State`
        #         assoc_loglik: [#object/tree][#local hypothesis, num_meas_ingate + 1] float
        loglik_mats = [None for _ in range(self.num_objs)]
        hypo_forest = [None for _ in range(self.num_objs)]
        for i, hypo_tree in enumerate(self.hypo_forest):
            hypo_tree_update = [] # initialize the hypo_forest
            # NOTE: we make the assoc_loglik has shape [#local hypothesis, num_objs + num_meas],
            # thus we can create cost matrix conveniently later.
            loglik_mats[i] = -np.inf * np.ones((len(hypo_tree), num_meas + self.num_objs))
            for j, hypo in enumerate(hypo_tree):
                # 1) implement ellipsoidal gating
                [meas_ingate, meas_index] = self.gate.gating(hypo, meas, self.measurement_model)

                # 2) calculate missed detection and predicted likelihood for each measurement inside the gate
                pred_loglik = self.density_model.predicted_log_likelihood(hypo, meas_ingate, self.measurement_model)
                missed_loglik = -np.inf * np.ones(self.num_objs)
                missed_loglik[i] = np.log(1 - detection_rate)
                detected_loglik = -np.inf * np.ones(num_meas)
                detected_loglik[meas_index] = np.log(detection_rate) - np.log(intensity_clutter) + pred_loglik
                # NOTE: we make the assoc_loglik has shape [#local hypothesis, num_objs + num_meas],
                # thus we can create cost matrix conveniently later.
                loglik_mats[i][j, :] = np.hstack((detected_loglik, missed_loglik)) # shape: [num_objs + num_meas]
                for k in range(-1, num_meas):
                    if k == -1:
                        # missed detection hypothesis
                        hypo_tree_update.append(hypo)
                    elif meas_index[k]:
                        # meas[:, k] in gate, update detection hypothesis state with meas_j
                        hypo_tree_update.append(self.density_model.update(hypo, meas[:, k], self.measurement_model))
                    else:
                        # meas[:, k] outside the gate, need to append 'None' to form the correct indices
                        hypo_tree_update.append(None)
            hypo_forest[i] = hypo_tree_update

        """global weights/look-up table/hypothesis forest update
        for each predicted global hypothesis
          1) create 2D cost matrix
          2) obtain M best assignments using Murty algorithm
          3) update global hypotheses look-up table  according to the M
             best assignment matrices obatained and use your new local hypotheses indexing
        """
        hypo_table_update = []
        log_weights_unnorm = []
        num_global_hypo = len(self.global_log_weights)
        for h in range(num_global_hypo):
            # 1) create 2D cost matrix of size [num_trees, num_meas + num_trees]
            # NOTE: `num_trees` is a constant in this MHT with known number of objects,
            # but it's not in general case.
            num_trees = len(self.hypo_forest)
            cost_mat = np.zeros((num_trees, num_meas + num_trees))
            for i in range(num_trees):
                # NOTE: cost matrix contains negative log likelihoods
                cost_mat[i,:] = -loglik_mats[i][self.hypo_table[h,i], :]

            # 2) obtain M best assigments using Murty algorithm
            assoc_num = np.ceil(np.exp(self.global_log_weights[h]) * self.reductor.capping_num)
            
            # Murty data association
            col_idx, cost = data_association(cost_mat, assoc_num)
            
            theta_t = col_idx
            theta_t[theta_t > num_meas - 1] = -1 # i.e. missed detection
            assoc_num = theta_t.shape[1]
            log_weights_unnorm.append(self.global_log_weights[h] - cost)
            for m in range(assoc_num):
                # 3) update global hypotheses look-up table according to the `assoc_num`
                #    best assignment matrices obtained and use new local
                #    hypotheses indexing
                table_row = np.zeros(num_trees, dtype=np.int)
                for i in range(num_trees):
                    # the first hypo DA in one leaf means missed local hypo,
                    # therefore 1+theta = 0 when theta == -1
                    table_row[i] = self.hypo_table[h, i] * (num_meas + 1) + 1 + theta_t[i, m] 
                hypo_table_update.append(table_row)

        # return
        self.global_log_weights, _ = normalize_log_weights(np.hstack(log_weights_unnorm))
        self.hypo_table = np.stack(hypo_table_update, 0)
        self.hypo_forest = hypo_forest

    def estimate(self):
        # extract object state estimates from the global hypothesis with
        # the highest weight
        max_idx = np.argmax(self.global_log_weights)
        est_t = [self.estimator(hypo_tree[self.hypo_table[max_idx, i]]) \
                 for i, hypo_tree in enumerate(self.hypo_forest)]
        return np.hstack(est_t)
    
    def reduction(self):
        # normalise global hypothesis weights and implement hypothesis reduction technique: prunning and capping
        num_global_hypo = self.hypo_table.shape[0]
        log_weights, keep_idx_prune = self.reductor.prune(self.global_log_weights, np.arange(num_global_hypo))
        log_weights, keep_idx_cap = self.reductor.cap(log_weights, np.arange(len(log_weights)))
        self.global_log_weights, log_weights_sum = normalize_log_weights(log_weights)
        self.hypo_table = self.hypo_table[keep_idx_prune, :]
        self.hypo_table = self.hypo_table[keep_idx_cap, :]
        
        # prune local hypotheses that are not include in any of the global hypotheses
        # local hypotheses_update: [i Hk_local * (1 + num_meas_ingate)]
        reindex_table = np.zeros_like(self.hypo_table)
        num_objs = len(self.hypo_forest)
        for i in range(num_objs):
            keep_local_idx, ic = np.unique(self.hypo_table[:,i], return_inverse=True)
            self.hypo_forest[i] = [self.hypo_forest[i][idx] for idx in keep_local_idx]
            
            # Re-index global hypotheses look-up table
            # H_table: [#global hypo, num_objs]
            reindex_table[:, i] = ic
        self.hypo_table = reindex_table
        
        assert self.hypo_table.shape == (len(self.global_log_weights), len(self.hypo_forest))
        
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