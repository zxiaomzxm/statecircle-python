#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
from scipy.stats import multivariate_normal

from .base import ConjugateDensityModel
from statecircle.types.state import LabeledState, GaussianLabeledState, GaussianState


class KalmanDensityModel(ConjugateDensityModel):
    r"""[Extended] Kalman density model

    linear gaussian transition/measurement model
    """

    def predict(self, state, time_step, transition_model):
        assert isinstance(state, GaussianState)
        mean_pred = transition_model.forward(state.mean, time_step=time_step)
        F = transition_model.transition_matrix(state.mean, time_step)
        cov_pred = F.dot(state.cov).dot(F.T) + transition_model.noise_covar(time_step=time_step)

        # TODO: add labeled state
        # if isinstance(state, LabeledState):
        #     return GaussianLabeledState(state.label, mean_pred, cov_pred)

        return GaussianState(mean_pred, cov_pred)

    def update(self, state_pred, meas, measurement_model):
        meas = meas[:, None] if meas.ndim == 1 else meas
        assert meas.shape[1] == 1, 'expected one measurement for update step, but got {} measurements.' \
            .format(meas.shape[1])

        # measurement matrix
        H = measurement_model.measurement_matrix(state_pred.mean)

        # innovation covariance
        meas_pred = measurement_model.forward(state_pred.mean)
        S = H.dot(state_pred.cov).dot(H.T) + measurement_model.noise_covar(meas_pred)

        # make sure matrix S is positive definite
        S = (S + S.T) / 2

        K = state_pred.cov.dot(H.T).dot(np.linalg.inv(S))

        # density update
        mean_upd = state_pred.mean + K.dot(meas - meas_pred)
        # covariance update
        # alternative formula:
        # cov_upd = (np.eye(measurement_model.ndim) - K.dot(H)).dot(state_pred.cov)
        cov_upd = state_pred.cov - K.dot(S).dot(K.T)

        # TODO: add labeled state
        # if isinstance(state_pred, LabeledState):
        #     return GaussianLabeledState(mean_upd, cov_upd, state_pred.label)

        return GaussianState(mean_upd, cov_upd)

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
        if 0 in meas.shape:
            return []

        # measurement matrix
        H = measurement_model.measurement_matrix(state_pred.mean)

        # innovation convariance
        meas_pred = measurement_model.forward(state_pred.mean)
        S = H.dot(state_pred.cov).dot(H.T) + measurement_model.noise_covar(meas_pred)

        # make sure matrix S is positive definite
        S = (S + S.T) / 2

        pred_loglik = multivariate_normal.logpdf(meas.T, meas_pred[:, 0], S, allow_singular=True)

        return pred_loglik
