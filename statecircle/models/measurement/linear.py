#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""

import numpy as np
from scipy.stats import multivariate_normal

from .base import MeasurementModel


class LinearMeasurementModel(MeasurementModel):
    r""" Linear measurement models

    Some dimensions of the state can be measured.
    Linear measuremodel can be described as follows:

    .. math::

        y_t = H_t*x_t + v_t, \ \ \ \ v_t \sim \mathcal{N}(0, R)

    Attributes
    ----------
    mapping : boolean mapping index
        Represent which dimension of the state can be measured
    """

    def __init__(self, mapping, sigma=None, cov=None):
        if sigma is not None:
            assert sigma > 0
        self.sigma = sigma
        self.mapping = np.array(mapping, dtype=np.bool_)

        # measurement funciton, in linear case, observation function is a matrix
        self.meas_mat = np.zeros((self.ndim_meas, self.ndim), dtype=np.float_)
        row = 0
        for idx, ele in enumerate(self.mapping):
            if ele:
                self.meas_mat[row, idx] = 1
                row += 1

        # noise convariance
        if cov is not None:
            self.noise_cov = cov
        elif sigma is not None:
            self.noise_cov = self.sigma ** 2 * np.eye(self.ndim_meas, dtype=np.float_)
        else:
            raise ValueError

    def measurement_matrix(self, x):
        return self.meas_mat

    def noise_covar(self, *args, **kwargs):
        return self.noise_cov

    def rvs(self, mean, cov, num_samples):
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T

    @property
    def ndim(self):
        return len(self.mapping)

    @property
    def ndim_meas(self):
        return self.mapping.sum()

    def forward(self, x, noisy=False):
        r"""calcute measurements with/without noise"""
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        num_states = x.shape[1]
        if noisy:
            noise = self.rvs(np.zeros(self.ndim_meas, dtype=np.float_),
                             self.noise_cov,
                             num_states)
            return self.meas_mat.dot(x) + noise
        else:
            return self.meas_mat.dot(x)

    def reverse(self, z):
        r"""calcute state from measurement"""
        assert z.shape[0] == self.ndim_meas
        return np.linalg.pinv(self.meas_mat).dot(z)
