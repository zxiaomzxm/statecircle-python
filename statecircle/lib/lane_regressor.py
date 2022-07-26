#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon May 25 22:38:11 2020

@author: zhaoxm
"""
import numpy as np

from sklearn.linear_model import RANSACRegressor

class LaneRegressor:
    def __init__(self, min_shift=0.0, min_samples=5, sample_range=[0, 20], sample_num=10, parallel=True):
        self.estimator_pos = RANSACRegressor(loss='squared_loss', stop_n_inliers=6)
        self.estimator_neg = RANSACRegressor(loss='squared_loss', stop_n_inliers=6)
        self.min_samples = min_samples
        self.min_shift = min_shift
        self.sample_range = sample_range
        self.sample_num = sample_num
        self.parallel = parallel
        self.pos_bond = None
        self.neg_bond = None
        
    def fit(self, coords):
        # coords: [2, #pts]
        # filter with lane poitns
        coords = coords[:, np.abs(coords[0]) >= self.min_shift]
        pos_idx = coords[0] >= 0
        self.pos_bond = coords[:, pos_idx][:, :, None]
        self.neg_bond = coords[:, ~pos_idx][:, :, None]
        
        if self.pos_bond.shape[1] > self.min_samples:
            self.reg_pos = self.estimator_pos.fit(self.pos_bond[1], self.pos_bond[0])
        else:
            self.reg_pos = None
            
        if self.neg_bond.shape[1] > self.min_samples:
            self.reg_neg = self.estimator_neg.fit(self.neg_bond[1], self.neg_bond[0])
        else:
            self.reg_neg = None
        
        if self.parallel:
            # parallize boundaries
            if self.reg_pos is not None and self.reg_neg is not None:
                self.reg_pos.estimator_.coef_ = self.reg_neg.estimator_.coef_ = \
                                (self.reg_pos.estimator_.coef_ + self.reg_pos.estimator_.coef_) / 2
        return self
        
    def sample(self):
        samples = []
        if self.reg_pos is not None:
            y_pos = np.linspace(*self.sample_range, self.sample_num)[:, None]
            x_pos = self.reg_pos.predict(y_pos)
            samples.append(np.hstack((x_pos, y_pos)).T)
            
        if self.reg_neg is not None:
            y_neg = np.linspace(*self.sample_range, self.sample_num)[:, None]
            x_neg = self.reg_neg.predict(y_neg)
            samples.append(np.hstack((x_neg, y_neg)).T)
            
        return samples
                        