#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""

import numpy as np
from scipy.stats import multivariate_normal

from .base import TransitionModel


class LinearTransitionModel(TransitionModel):
    r""" Linear measurement models """


class ConstantVelocityModel(LinearTransitionModel):
    r"""Constant velocity linear transition model

    .. math::

        x_{t+1} = F_t*x_t + u, \ \ \ \ u \sim \mathcal{N}(0, Q_t)
    """

    state_dim = 4
    __exact_cov__ = True
    
    def __init__(self, sigma):
        r"""

        Parameters
        ----------
        sigma: noise variance scale
        """
        self.sigma = sigma

    def transition_matrix(self, x, time_step):
        r"""Linear forward matrix

        Parameters
        ----------
        time_step: scalar

        Returns
        -------
        forward matrix
        """
        # transition function
        T = time_step
        return np.array([[1, 0, T, 0],
                         [0, 1, 0, T],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def noise_covar(self, time_step):
        r"""

        Parameters
        ----------
        time_step: scalar

        Returns
        noise covariance matrix
        -------

        """
        T = time_step
        if not self.__exact_cov__:
            covar = self.sigma ** 2 * np.array([[T ** 4 / 4, 0, T ** 3 / 2, 0],
                                                [0, T ** 4 / 4, 0, T ** 3 / 2],
                                                [T ** 3 / 2, 0, T ** 2, 0],
                                                [0, T ** 3 / 2, 0, T ** 2]])    
        else:
            covar = self.sigma ** 2 * np.kron(np.array([[T**3/3, T**2/2], 
                                                        [T**2/2, T]])
                                                ,np.eye(2))
            
        return covar

    def rvs(self, mean, cov, num_samples):
        r"""

        Parameters
        ----------
        mean: mean vector
        cov: covariance matrix
        num_samples: scalar

        Returns
        -------
        random sample
        """
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T

    @property
    def ndim(self):
        return self.state_dim

    def forward(self, x, time_step, noisy=False):
        r"""

        Parameters
        ----------
        x : :attr: `State`
        time_step: scalar
        noisy: boolean

        Returns
        -------
        `State`(next step)
        """
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        T = time_step
        if noisy:
            num_states = x.shape[1]
            noise = self.rvs(np.zeros(self.ndim, dtype=np.float_),
                             self.noise_covar(T), num_states)
            return self.transition_matrix(x, T).dot(x) + noise  # additive Gaussian noise
        else:
            return self.transition_matrix(x, T).dot(x)
        
    def transform_state(self, state_mean, rot_mat, translation, vel, theta):
        # TODO: we need the exact extrinsic parameter of radar, not the
        # hand-craft number here...
        # radar coordinate system covert to ego car coordinate system
        state_mean[1,:] += 3.3
        
        # TODO: compensate ego car's velocity or not?
#        state_mean[3,:] += vel
        state_mean[:2,:] = rot_mat.T.dot(state_mean[:2,:] - translation)
        state_mean[2:,:] = rot_mat.T.dot(state_mean[2:,:])
#        state_mean[3,:] -= vel
    
        # radar coordinate system covert to ego car coordinate system
        state_mean[1,:] -= 3.3
        
        return state_mean
    
    
class ConstantAccelerationModel(LinearTransitionModel):
    r"""Constant Acceleration linear transition model
    
        .. math::
    
            x_{t+1} = F_t*x_t + u, \ \ \ \ u \sim \mathcal{N}(0, Q_t)
        """

    state_dim = 6
    __exact_cov__ = False
    
    def __init__(self, sigma):
        r"""

        Parameters
        ----------
        sigma: noise variance scale
        """
        self.sigma = sigma

    def transition_matrix(self, x, time_step):
        r"""Linear forward matrix

        Parameters
        ----------
        time_step: scalar

        Returns
        -------
        forward matrix
        """
        # transition function
        T = time_step
        return np.kron(np.array([[1, T, 0.5*T*T],
                         [0, 1, T],
                         [0, 0, 1]]), np.eye(2))

    def noise_covar(self, time_step):
        r"""

        Parameters
        ----------
        time_step: scalar

        Returns
        noise covariance matrix
        -------

        """
        T = time_step
        if not self.__exact_cov__:
            covar = self.sigma ** 2 * np.kron(np.array([[T**6 / 36, T**5/12, T**4 / 6],
                                                        [T**5/12, T**4 / 4, T**3 / 2],
                                                [T**4 / 6, T**3 / 2, T**2]]), np.eye(2))
        else:
            covar = self.sigma ** 2 * np.kron(np.array([T**5/20, T**4/8, T**3/6],
                                                       [T**4/8,  T**3/3, T**2/2],
                                                       [T**3/6,  T**2/2, T]),
                                              np.eye(2))

        return covar

    def rvs(self, mean, cov, num_samples):
        r"""

        Parameters
        ----------
        mean: mean vector
        cov: covariance matrix
        num_samples: scalar

        Returns
        -------
        random sample
        """
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T

    @property
    def ndim(self):
        return self.state_dim

    def forward(self, x, time_step, noisy=False):
        r"""

        Parameters
        ----------
        x : :attr: `State`
        time_step: scalar
        noisy: boolean

        Returns
        -------
        `State`(next step)
        """
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        T = time_step
        if noisy:
            num_states = x.shape[1]
            noise = self.rvs(np.zeros(self.ndim, dtype=np.float_),
                             self.noise_covar(T), num_states)
            return self.transition_matrix(x, T).dot(x) + noise  # additive Gaussian noise
        else:
            return self.transition_matrix(x, T).dot(x)
        
    def transform_state(self, state_mean, R, Translation, vel, theta):
        # TODO: we need the exact extrinsic parameter of radar, not the
        # hand-craft number here...
        # radar coordinate system covert to ego car coordinate system
        state_mean[1,:] += 3.3
        
        # TODO: compensate ego car's velocity or not?
#        state_mean[3,:] += vel
        state_mean[:2,:] = R.T.dot(state_mean[:2,:]) - R.T.dot(Translation)
        R_kron = np.kron(np.eye(2), R)
        state_mean[2:,:] = R_kron.T.dot(state_mean[2:,:])
#        state_mean[3,:] -= vel
    
        # radar coordinate system covert to ego car coordinate system
        state_mean[1,:] -= 3.3
        return state_mean
