#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 18 11:29:15 2019

@author: zhxm
"""
import numpy as np

from statecircle.types.state import GaussianState
from statecircle.utils.common import list_logical_index, list_index, normalize_log_weights
from .base import Reductor


class HypothesisReductor(Reductor):
    r""" Hypothesis trees reductor """

    def __init__(self, weight_min, merging_threshold, capping_num, 
                 prob_min=None, prob_recycle=None):
        r"""

        Parameters
        ----------
        weight_min : used for prune method
        merging_threshold : used for merge method
        capping_num : used for cap method
        prob_min, prob_recycle: used for recycle method
        """
        self.weight_min = weight_min
        self.log_weight_min = np.log(weight_min)
        self.merging_threshold = merging_threshold
        self.capping_num = capping_num
        self.prob_min = prob_min
        self.prob_recycle = prob_recycle

    def prune(self, log_weights, hypo_tree):
        r"""Prune hypothesis with weights lower than threshold

        Parameters
        ----------
        hypo_tree
        threshold

        Returns
        -------
        log_weights, hypo_tree
        """
        if len(log_weights) == 0:
            return log_weights, hypo_tree
        
        idx = log_weights > self.log_weight_min
        log_weights = log_weights[idx]
        hypo_tree = list_logical_index(hypo_tree, idx)
        return log_weights, hypo_tree

    def cap(self, log_weights, hypo_tree):
        r"""Capping hypotheses let the number of hypotheses <= capping_num

        Parameters
        ----------
        hypo_tree

        Returns
        -------
        log_weights, hypo_tree
        """
        if len(log_weights) == 0:
            return log_weights, hypo_tree
        
        if len(log_weights) <= self.capping_num:
            return log_weights, hypo_tree

        idx = np.argsort(-log_weights)[:self.capping_num]
        return log_weights[idx], list_index(hypo_tree, idx)

    def moment_matching(self, log_weights, states):
        r"""Moment matching for 1st/2nd moments

        Parameters
        ----------
        log_weights
        states

        Returns
        -------
        state
        """
        if len(log_weights) == 0:
            return states
        
        if len(log_weights) == 1:
            return states[0]

        w = np.exp(log_weights)

        num_states = len(states)
        # TODO: support different states
        if num_states == 0:
            return None
        
        state = type(states[0])(0., 0.)
        for i in range(num_states):
            state.mean = state.mean + w[i] * states[i].mean

        for i in range(num_states):
            state.cov = state.cov + w[i] * (states[i].cov + np.outer((state.mean - states[i].mean), (state.mean - states[i].mean)))

        if hasattr(states[0], 'label'):
            max_idx = log_weights.argmax()
            state.label = states[max_idx].label

        return state

    def merge(self, log_weights, states):
        r"""Merge state components

        Parameters
        ----------
        log_weights
        states

        Returns
        -------
        log_weights, states

        """
        if len(log_weights) <= 1:
            return log_weights, states

        # index set of components
        I = list(range(len(states)))
        el = 0
        log_w_hat = []
        states_hat = []
        while len(I) != 0:
            Ij = []
            # find the component with the highest weight
            j = np.argmax(log_weights)

            for i in I:
                tmp = (states[i].mean - states[j].mean)[:,0]
                val = tmp.dot(np.linalg.inv(states[j].cov)).dot(tmp)

                # find other similar components in the sense of small Mahalnobis distance
                # TODO: use a percentile to calulate the merging threshold like the gating size
                if val < self.merging_threshold:
                    Ij.append(i)

            # merge components by moment matching
            tmp, log_w_upd = normalize_log_weights(np.array([log_weights[idx] for idx in Ij]))
            states_upd = self.moment_matching(tmp, [states[idx] for idx in Ij])

            log_w_hat.append(log_w_upd)
            states_hat.append(states_upd)

            # remove indices of merged components from index set
            I = set(I) - set(Ij)
            # set a negative to make sure this component won't be selected again
            for idx in Ij:
                log_weights[idx] = np.log(1e-100)
            el = el + 1

        return np.array(log_w_hat), states_hat
