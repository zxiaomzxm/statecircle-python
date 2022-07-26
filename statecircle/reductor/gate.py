#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 15:28:34 2019

@author: zhxm
"""

import numpy as np
from scipy.stats import chi2

from statecircle.models.measurement.linear import LinearMeasurementModel
from statecircle.models.measurement.nonlinear import NonlinearMeasurementModel
from .base import Reductor


class Gate(Reductor):
    r""" Gating base class """

    def gating(self):
        """ virtual gating operation function """
        raise NotImplementedError


class RectangularGate(Gate):
    r""" Rectangular gate """
    def gating(self):
        raise NotImplementedError



class EllipsoidalGate(Gate):
    r"""Ellipsoidal gating mechanism """

    def __init__(self, percentile=None, thresh=None):
        r"""

        Parameters
        ----------
        percentile
        thresh
        """
        self.percentile = percentile
        self.thresh = thresh

    def gating(self, state, meas, measurement_model):
        r"""Ellipsoidal gating operation function

        Parameters
        ----------
        state
        meas
        measurement_model

        Returns
        -------
        meas_ingate : `Matrix` (meas_dim, num_meas_in_gate)
            measurements in current state gate scope
        meas_index : `BoolVector` (num_meas)
            bool vector represents the measurements index which inside the gate
        """
        # used for `KalmanAccumulatedDensityModel`
        ndim = state.mean.shape[0]
        ndim_meas = measurement_model.ndim_meas
            
        if len(meas) == 0:
            return np.empty((ndim_meas, 0)), np.empty(0)
        
        if self.percentile is not None:
            self.thresh = chi2.ppf(self.percentile, ndim_meas)
        elif self.thresh is not None:
            self.percentile = chi2.cdf(self.thresh, ndim_meas)
        else:
            raise ValueError
            
        # measurement matrix
        H = measurement_model.measurement_matrix(state.mean[:, -1:])

        # innovation covariance
        meas_pred = measurement_model.forward(state.mean[:, -1:])
        S = H.dot(state.cov[-ndim:, -ndim:]).dot(H.T) + measurement_model.noise_covar(meas_pred)

        # make sure matrix S is positive definite
        S = (S + S.T) / 2
            
        num_meas = meas.shape[1]
        distances = np.zeros(num_meas)
        for i in range(num_meas):
            distances[i] = (meas[:, i] - meas_pred[:, 0]).dot(np.linalg.inv(S)).dot(meas[:, i] - meas_pred[:, 0])
                
        meas_index = distances <= self.thresh
        meas_ingate = meas[:, meas_index]

        return meas_ingate, meas_index
