#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 25 14:33:07 2019

@author: zhaoxm
"""
import numpy as np
from copy import deepcopy

from scipy.stats import multivariate_normal

from statecircle.types.state import GaussianState, GaussianAccumulatedState
from .base import ConjugateDensityModel


# TODO: support unscented state
class KalmanAccumulatedDensityModel(ConjugateDensityModel):
    r"""[Extended] Kalman accumulated state model

    linear gaussian transition/measurement model
    """
    def __init__(self, traceback_range=100):
        r"""

        Parameters
        ----------
        traceback_range : int
            accumulated density range
        """
        assert traceback_range > 0, "`traceback_range` muse be positive."
        self.traceback_range = traceback_range

    def predict_state(self, state, time_step, transition_model):
        assert isinstance(state, GaussianState)

        if state.mean.ndim == 1:
            state.mean = state.mean[:, None]
        d = transition_model.state_dim

        mean_pred = transition_model.forward(state.mean[:, -1], time_step=time_step)
        F = transition_model.transition_matrix(state.mean[:, -1], time_step)
        cov_pred = F.dot(state.cov[-d:, -d:]).dot(F.T) + transition_model.noise_covar(time_step=time_step)

        return GaussianState(mean_pred, cov_pred)

    def predict(self, state, time_step, transition_model):
        state.mean = state.mean[:, None] if state.mean.ndim == 1 else state.mean
        mean_pred = np.hstack((state.mean, transition_model.forward(state.mean[:, -1:], time_step)))
        F = transition_model.transition_matrix(state.mean[:, -1], time_step)
        d = transition_model.state_dim
        if self.traceback_range > 1:
            step_d = (self.traceback_range - 1) * d
            new_cov1 = state.cov[-step_d:, -d:].dot(F.T)
            new_cov2 = F.dot(state.cov[-d:, -step_d:])
            new_cov3 = F.dot(state.cov[-d:, -d:]).dot(F.T) + transition_model.noise_covar(time_step)
            cov_pred = np.vstack((np.hstack((state.cov[-step_d:, -step_d:], new_cov1)),
                                  np.hstack((new_cov2, new_cov3))
                                  ))
            return GaussianAccumulatedState(mean_pred, cov_pred)
        else:
            cov_pred = F.dot(state.cov[-d:, -d:]).dot(F.T) + transition_model.noise_covar(time_step)
            return GaussianAccumulatedState(mean_pred, cov_pred)

    def update(self, state_pred, meas, measurement_model):
        meas = meas[:, None] if meas.ndim == 1 else meas
        assert meas.shape[1] == 1, 'expected one measurement for update step, but got {} measurements.' \
            .format(meas.shape[1])

        state_pred.mean = state_pred.mean[:, None] if state_pred.mean.ndim == 1 else state_pred.mean

        nx = state_pred.mean.shape[0]

        # measurement model Jacobian
        H = measurement_model.measurement_matrix(state_pred.mean[:, -1])

        # innovation covariance
        meas_pred = measurement_model.forward(state_pred.mean[:, -1])
        S = H.dot(state_pred.cov[-nx:, -nx:]).dot(H.T) + measurement_model.noise_covar(meas_pred)

        # make sure matrix S is positive definite
        S = (S + S.T) / 2

        step = self.traceback_range
        K = state_pred.cov[:, -nx:].dot(H.T).dot(np.linalg.inv(S))

        # state update
        state_upd = deepcopy(state_pred)
        state_upd.mean[:, -step:] = state_pred.mean[:, -step:] + (
                                                        K.dot(meas - meas_pred)).reshape(-1, nx).T

        # covariance update
        # bug fix, use Kalman update equation `P = P - KSK` to avoid numerical problem
        # state_upd.cov = state_pred.cov- K.dot(H).dot(state_pred.cov[-nx:, :])
        state_upd.cov = state_pred.cov - K.dot(S).dot(K.T)
        
        return state_upd


    def predicted_log_likelihood(self, state_pred, meas, measurement_model):
        r"""

        Parameters
        ----------
        state_pred : `GaussianState`
        meas : array [`ndim_meas`, num_meas]
        measurement_model :

        Returns
        -------
        log_likelihood : array [num_meas]
        """
        state_pred.mean = state_pred.mean[:, None] if state_pred.mean.ndim == 1 else state_pred.mean
        
        if 0 in meas.shape:
            return []

        nx = state_pred.mean.shape[0]

        # measurement model Jocobian
        H = measurement_model.measurement_matrix(state_pred.mean[:, -1])

        # innovation convariance
        meas_pred = measurement_model.forward(state_pred.mean[:, -1])
        S = H.dot(state_pred.cov[-nx:, -nx:]).dot(H.T) + measurement_model.noise_covar(meas_pred)

        # make sure matrix S is positive definite
        S = (S + S.T) / 2

        pred_loglik = multivariate_normal.logpdf(meas.T, meas_pred[:, 0], S, allow_singular=True)

        return pred_loglik
    
