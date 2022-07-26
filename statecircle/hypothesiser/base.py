#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 24 11:16:36 2019

@author: zhaoxm
"""
import numpy as np
from abc import abstractmethod

from statecircle.utils.common import normalize_log_weights
from statecircle.utils.data_association import data_association
from ..base import Base

class Hypothesis(Base):
    r"""Base hypothesis class"""
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def predict(self):
        r"""virtual predict method"""
        
    @abstractmethod
    def update(self):
        r"""virtual update method"""

    @abs
    def reduction(self):
        r"""virtual reduction method"""
    

class MultiHypothesis(Hypothesis):
    r"""Multi-hypothesis class
    Multi-hypothesis has two equivalent representations:
    1) local hypotheses (hypotheses forest) representation
    2) global hypotheses representation
    this class use the local hypothesis representation, which has three parts
    1) global weights
    2) hypotheses look-up table
    3) local hypotheses forest
    """
    def __init__(self):
        # global log weights
        self.log_weights = np.array([])
        # hypothesis look-up table
        self.hypo_table = np.empty((0, 0))
        # hypothesis forest
        self.hypo_forest = []
        
    def predict(self, time_step, density_model, transition_model):
        # predict each local hypothesis in each hypothesis tree
        self.sanity_check()
        self.hypo_forest = [[density_model.predict(hypo, time_step, transition_model)
                             for hypo in hypo_tree]
                            for hypo_tree in self.hypo_forest]
    
    def update(self, meas, density_model, measurement_model, clutter_model):
        """local hypothese (self.hypo_forest) update
        for each local hypothesis in each hypothesis tree
           1) ellipsoidal gating
           2) calculate missed detection and predicted likelihood for each
              measurement inside the gate
           3) create updated local hypothesis
        """
        self.sanity_check()
        num_meas = meas.shape[1]
        detection_rate = self.clutter_model.detection_rate
        intensity_clutter = self.clutter_model.intensity_clutter
        # intialize updated parameters
        assoc_loglik = [None for _ in range(self.num_objs)]
        hypo_forest = [None for _ in range(self.num_objs)]
        for i, hypo_tree in enumerate(self.hypo_forest):
            hypo_tree_update = [] # initialize the hypo_forest
            # NOTE: we make the asssoc_loglik has shape [#local hypothesis, num_objs + num_meas],
            # thus we can create cost matrix conveniently later.
            assoc_loglik[i] = -np.inf * np.ones((len(hypo_tree), num_meas + self.num_objs))
            for j, hypo in enumerate(hypo_tree):
                # 1) implement ellipsoidal gating
                [meas_ingate, meas_index] = self.gate.gating(hypo, meas, self.measurement_model)
                
                # 2) calculate missed detection and predicted likelihood for each measurement inside the gate
                pred_loglik = density_model.predicted_log_likelihood(hypo, meas_ingate, self.measurement_model)
                missed_loglik = -np.inf * np.ones(self.num_objs)
                missed_loglik[i] = np.log(1 - detection_rate)
                detected_loglik = -np.inf * np.ones(num_meas)
                detected_loglik[meas_index] = np.log(detection_rate) - np.log(intensity_clutter) + pred_loglik
                # NOTE: we make the asssoc_loglik has shape [#local hypothesis, num_objs + num_meas],
                # thus we can create cost matrix conveniently later.
                assoc_loglik[i][j, :] = np.hstack((detected_loglik, missed_loglik)) # shape: [num_objs + num_meas]
                for k in range(-1, num_meas):
                    if k == -1:
                        # missed detection hypothesis
                        hypo_tree_update.append(hypo)
                    elif meas_index[k]:
                        # meas[:, k] in gate, update detection hypothesis state with meas_j
                        hypo_tree_update.append(density_model.update(hypo, meas[:, k], self.measurement_model))
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
            num_trees = len(self.hypo_forest)
            cost_mat = np.zeros((num_trees, num_meas + num_trees))
            for i in range(num_trees):
                # NOTE: cost matrix contains negative log likelihoods
                cost_mat[i,:] = -assoc_loglik[i][self.hypo_table[h,i], :]
                
            # 2) obtain M best assignments using Murty algorithm
            assoc_num = np.ceil(np.exp(self.global_log_weights[h]) * self.reductor.capping_num)
            # Murty data association
            col_idx, cost = data_association(cost_mat, assoc_num)
            
            theta_t = col_idx
            theta_t[theta_t > num_meas - 1] = -1 # i.e. missed detection
            assoc_num = theta_t.shape[1]
            log_weights_unnorm.append(self.global_log_weights[h] - cost)
            for m in range(assoc_num):
                # 3) update global hypotheses look-up table according to the `assoc_num`
                #    best assigment matrices obtained and use new local 
                #    hypotheses indexing
                table_row = np.zeros(num_trees, dtype=np.int)
                for i in range(num_trees):
                    # the first hypo DA in one leaf means missed local hypo,
                    # therefore 1+theta = 0 when theta == -1
                    table_row[i] = self.hypo_table[h, i] * (num_meas + 1) + 1 + theta_t[i, m] 
                hypo_table_update.append(table_row)
            
        # return updated parameters
        self.log_weights, _ = normalize_log_weights(np.hstack(log_weights_unnorm))
        self.hypo_table = np.stack(hypo_table_update, 0)
        self.hypo_forest = hypo_forest

    def reduction(self, reductor):
        # normalise global hypothesis weights and implement hypothesis reduction technique: pruning and capping
        num_global_hypo = self.hypo_table.shape[0]
        log_weights, keep_idx_prune = reductor.prune(self.log_weights, np.arange(num_global_hypo))
        log_weights, keep_idx_cap = reductor.cap(log_weights, np.arange(len(log_weights)))
        self.log_weights, log_weights_sum = normalize_log_weights(log_weights)
        self.hypo_table = self.hypo_table[keep_idx_prune, :]
        self.hypo_table = self.hypo_table[keep_idx_cap, :]

        # prune local hypotheses that are not include in any of the global hypotheses
        # local hypotheses_update: [i Hk_local * (1 + num_meas_ingate)]
        reindex_table = np.zeros_like(self.hypo_table)
        num_objs = len(self.hypo_forest)
        for i in range(num_objs):
            keep_local_idx, ic = np.unique(self.hypo_table[:, i], return_inverse=True)
            self.hypo_forest[i] = [self.hypo_forest[i][idx] for idx in keep_local_idx]

            # Re-index global hypotheses look-up table
            # H_table: [#global hypo, num_objs]
            reindex_table[:, i] = ic
        self.hypo_table = reindex_table

    def sanity_check(self):
        assert self.hypo_tabe.shape == (len(self.log_weights), len(self.hypo_forest))
        for i, hypo_tree in enumerate(self.hypo_forest):
            assert len(np.unique(self.hypo_table[:, i])) == len(hypo_tree)
        
        
        
    