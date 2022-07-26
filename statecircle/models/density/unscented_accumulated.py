#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Mar  3 17:16:59 2020

@author: zhaoxm
"""

import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal

from statecircle.models.density.unscented import UnscentedDensityModel
from statecircle.types.base import SigmaPoints
from statecircle.types.state import GaussianState, GaussianAccumulatedState, UnscentedState
from .base import ConjugateDensityModel

from deprecated import deprecated

@deprecated(reason="uncompleted.")
class UnscentedAccumulatedDensityModel(UnscentedDensityModel):
    r"""[Extended] Kalman accumulated state model

    linear gaussian transition/measurement model
    """

    def __init__(self, state_dim, alpha=1.0, beta=2.0, kappa=None, traceback_range=100):
        r"""

        Parameters
        ----------
        state_dim: int
            state dimension
        alpha: float
            Determine the spread of the sigma points
        beta: float
            Incorpate prior knowledge of the distribution
        kappa: float
            Secondary scaling parameter which is usually set to 3 - `state_dim`
        traceback_range : int
            accumulated density range
        """
        super().__init__(state_dim, alpha, beta, kappa)
        self.traceback_range = traceback_range

    def update_sigma_points(self, state):
        # inplace op
        # TODO: membrane unscented parameters
        L = np.linalg.cholesky(state.cov)
        points = np.tile(state.mean.astype('float'), (1, 2 * self.state_dim + 1))
        points[:, 1:(self.state_dim + 1)] += np.sqrt(self.state_dim + self.lambda_) * L
        points[:, (self.state_dim + 1):] -= np.sqrt(self.state_dim + self.lambda_) * L

        sigma_pts = SigmaPoints(points, self.mean_weights, self.cov_weights)
        state.sigma = sigma_pts

    def predict_state(self, state, time_step, transition_model):
        if state.mean.ndim == 1:
            state.mean = state.mean[:, None]
        d = transition_model.state_dim

        assert isinstance(state, UnscentedState)
        # generate sigma points
        self.update_sigma_points(UnscentedState(state.mean[:, -1], state.cov[-d:, -d:]))

        points_pred = transition_model.forward(state.sigma.points, time_step=time_step)
        mean_pred = points_pred.dot(state.sigma.mean_weights)[:, None]
        res_pred = points_pred - mean_pred
        cov_pred = res_pred.dot(np.diag(state.sigma.cov_weights).dot(res_pred.T)) + \
                   transition_model.noise_covar(time_step)

        return GaussianState(mean_pred, cov_pred)

    def predict(self, state, time_step, transition_model):
        state.mean = state.mean[:, None] if state.mean.ndim == 1 else state.mean
        # sigma points predict
        points_pred = transition_model.forward(state.sigma.points, time_step=time_step)
        mean_pred = points_pred.dot(state.sigma.mean_weights)[:, None]
        mean_pred_accu = np.hstack((state.mean, mean_pred))

        F = transition_model.transition_matrix(state.mean[:, -1], time_step)
        d = transition_model.state_dim
        step_d = self.traceback_range * d
        new_cov1 = state.cov[-step_d:, -d:].dot(F.T)
        new_cov2 = F.dot(state.cov[-d:, -step_d:])
        new_cov3 = F.dot(state.cov[-d:, -d:]).dot(F.T) + transition_model.noise_covar(time_step)
        cov_pred = np.vstack((np.hstack((state.cov[-step_d:, -step_d:], new_cov1)),
                              np.hstack((new_cov2, new_cov3))
                              ))
        return GaussianAccumulatedState(mean_pred_accu, cov_pred)

    def update(self, state_pred, meas, measurement_model):
        meas = meas[:, None] if meas.ndim == 1 else meas
        assert meas.shape[1] == 1, 'expected one measurement for update step, but got {} measurements.' \
            .format(meas.shape[1])

        state_pred.mean = state_pred.mean[:, None] if state_pred.mean.ndim == 1 else state_pred.mean

        nx = state_pred.mean.shape[0]

        # measurement model Jacobian
        H = measurement_model.measurement_matrix(state_pred.mean[:, -1])

        # innovation covariance
        S = H.dot(state_pred.cov[-nx:, -nx:]).dot(H.T) + measurement_model.noise_covar()

        # make sure matrix S is positive definite
        S = (S + S.T) / 2

        step = self.traceback_range
        step_d = self.traceback_range * nx
        K = state_pred.cov[-step_d:, -nx:].dot(H.T).dot(np.linalg.inv(S))

        # state update
        state_upd = deepcopy(state_pred)
        state_upd.mean[:, -step:] = state_pred.mean[:, -step:] + (
            K.dot(meas - measurement_model.forward(state_pred.mean[:, -1]))).reshape(-1, nx).T

        # covariance update
        state_upd.cov = state_pred.cov[-step_d:, -step_d:] - K.dot(H).dot(state_pred.cov[-nx:, -step_d:])

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
        S = H.dot(state_pred.cov[-nx:, -nx:]).dot(H.T) + measurement_model.noise_covar()

        # make sure matrix S is positive definite
        S = (S + S.T) / 2

        mean = measurement_model.forward(state_pred.mean[:, -1])
        pred_loglik = multivariate_normal.logpdf(meas.T, mean[:, 0], S, allow_singular=True)

        return pred_loglik
