#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Feb 27 09:49:13 2020

@author: zhaoxm
"""

import numpy as np
from scipy.stats import multivariate_normal

from statecircle.types.base import SigmaPoints
from .base import ConjugateDensityModel
from statecircle.types.state import UnscentedState, GaussianState

# TODO: add `UnscentedAccumulatedDensityModel`
class UnscentedDensityModel(ConjugateDensityModel):
    """
    # UnscentedState defination:
    # state.mean [state_dim] mean
    # state.cov [state_dim ,state_dim] covariance
    # state.sigma.points [state_dim/meas_dim, 2*state_dim + 1] sigma points position
    # state.sigma.mean_weights [2*state_dim + 1] mean sigma points weights
    # state.sigma.cov_weights [2*state_dim + 1] covariance sigma points weights

    Ref:
    [1] Wan E A, Van Der Merwe R. The unscented Kalman filter for nonlinear estimation. //
    Proceedings of the IEEE 2000 Adaptive System for Signal Processing, Communications and
    Control Symposium.Ieee, 2000: 153-158
    [2] S.J. Julier and J.K. Uhlmann. A new extension of the Kalman gilter to nonlinear system. //
    In proc. of AeroSense: The 11th Int.Symp.on Aerospace/Defence Sensing, Simulation and Controls,
    1997.
    """

    def __init__(self, state_dim, alpha=1.0, beta=2.0, kappa=None):
        """

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

        """
        super().__init__()
        self.state_dim = state_dim

        # hyper-pamateters
        self.alpha = alpha  # range in (0, 1]
        self.beta = beta  # optimal choice for Gaussians is 2
        self.kappa = kappa or 3. - state_dim  # 'state_dim + kappa range' in [0, +inf)
        self.lambda_ = alpha ** 2 * (state_dim + self.kappa) - state_dim

        # constant sigma weights
        self.mean_weights = np.ones([2 * state_dim + 1]) / 2 / (state_dim + self.lambda_)
        self.mean_weights[0] = self.lambda_ / (state_dim + self.lambda_)

        self.cov_weights = self.mean_weights.copy()
        self.cov_weights[0] += 1 - alpha ** 2 + beta

    def update_sigma_points(self, state):
        # inplace op
        # TODO: membrane unscented parameters
        L = np.linalg.cholesky(state.cov)
        points = np.tile(state.mean.astype('float'), (1, 2 * self.state_dim + 1))
        points[:, 1:(self.state_dim + 1)] += np.sqrt(self.state_dim + self.lambda_) * L
        points[:, (self.state_dim + 1):] -= np.sqrt(self.state_dim + self.lambda_) * L

        sigma_pts = SigmaPoints(points, self.mean_weights, self.cov_weights)
        state.sigma = sigma_pts

    def compute_innovation_cov(self, measurement_model, sigma):
        sigma_pts_meas = measurement_model.forward(sigma.points)

        meas_hat = sigma_pts_meas.dot(sigma.mean_weights)[:, None]
        # innovation covariance
        res_meas = sigma_pts_meas - meas_hat
        S = res_meas.dot(np.diag(sigma.cov_weights).dot(res_meas.T)) + measurement_model.noise_covar(meas_hat)

        return S

    def predict(self, state, time_step, transition_model):
        assert isinstance(state, UnscentedState)
        # generate sigma points
        self.update_sigma_points(state)

        # TODO: consider the nonlinear noise factor, i.e. augmented sigma points
        # sigma points predict
        points_pred = transition_model.forward(state.sigma.points, time_step=time_step)

        mean_pred = points_pred.dot(state.sigma.mean_weights)[:, None]
        res = points_pred - mean_pred
        cov_pred = res.dot(np.diag(state.sigma.cov_weights).dot(res.T)) + transition_model.noise_covar(time_step)

        # TODO: add labeled state
        # if hasattr(state, 'label'):
        #     state_pred.label = state.label

        return UnscentedState(mean_pred, cov_pred, SigmaPoints(points_pred, self.mean_weights, self.cov_weights))

    def update(self, state_pred, meas, measurement_model):
        meas = meas[:, None] if meas.ndim == 1 else meas
        assert meas.shape[1] == 1, 'expected one measurement for update step, but got {} measurements.' \
            .format(meas.shape[1])

        sigma_pts_meas = measurement_model.forward(state_pred.sigma.points)
        meas_hat = sigma_pts_meas.dot(state_pred.sigma.mean_weights)[:, None]

        # innovation covariance
        res_meas = sigma_pts_meas - meas_hat
        # NOTE: pre-calculate the innovation covariance matrix in `predicted_log_likelihood` function
        # and save the property in `UnscentedState`, because this function always run before `update`
        # function
        # inno_cov = res_meas.dot(np.diag(state_pred.sigma.cov_weights).dot(res_meas.T)) + measurement_model.noise_covar()

        # cross-covariance
        res_state = state_pred.sigma.points - state_pred.sigma.points[:, 0:1]
        # almost equal to the below equation
        # res_state = state_pred.sigma.points - state_pred.sigma.points.dot(state_pred.sigma.mean_weights)[:, None]
        
        T = res_state.dot(np.diag(state_pred.sigma.cov_weights).dot(res_meas.T))
        # kalman gain
        K = T.dot(np.linalg.inv(state_pred.inno_cov))

        # state update
        mean_upd = state_pred.mean + K.dot(meas - meas_hat)
        # covariance update
        cov_upd = state_pred.cov - K.dot(state_pred.inno_cov).dot(K.T)

        # TODO: add labeled state
        # if hasattr(state_pred, 'label'):
        #     state_upd.label = state_pred.label
        
        return UnscentedState(mean_upd, cov_upd, SigmaPoints)

    def predicted_log_likelihood(self, state_pred, meas, measurement_model):
        r"""

        Parameters
        ----------
        state_pred : `GaussianState` -> `UnscentedState`
        meas : array [`ndim_meas`, num_meas]
        measurement_model :

        Returns
        -------
        log_likelihood : array [num_meas]
        """
        if 0 in meas.shape:
            return []

        if isinstance(state_pred, GaussianState) or state_pred.sigma is None:
            # initialize sigma points
            self.update_sigma_points(state_pred)

        sigma_pts_meas = measurement_model.forward(state_pred.sigma.points)

        if isinstance(state_pred, GaussianState) or state_pred.inno_cov is None:
            # update innovation covariance
            meas_hat = sigma_pts_meas.dot(state_pred.sigma.mean_weights)[:, None]
            res_meas = sigma_pts_meas - meas_hat
            inno_cov = res_meas.dot(np.diag(state_pred.sigma.cov_weights).dot(res_meas.T)) + \
                                                            measurement_model.noise_covar(meas_hat)
        else:
            inno_cov = state_pred.inno_cov

        mean_meas = sigma_pts_meas.dot(state_pred.sigma.mean_weights)
        pred_loglik = multivariate_normal.logpdf(meas.T, mean_meas, inno_cov, allow_singular=True)

        state_pred.inno_cov = inno_cov

        return pred_loglik

    @property
    def ndim(self):
        return self.state_dim
