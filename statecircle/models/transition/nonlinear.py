#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from abc import abstractmethod
import numpy as np
from scipy.stats import multivariate_normal
from pytest import approx

from .base import TransitionModel
from statecircle.utils.common import rem2pi

class NonlinearTransitionModel(TransitionModel):
    r"""Nonlinear transition model base class"""

    def transition_matrix(self, *args, **kwargs):
        return self.jacobian(*args, **kwargs)

    @abstractmethod
    def jacobian(self, *args, **kwargs):
        r"""virtual jacobian matrix method"""


class MonoCycleModel(NonlinearTransitionModel):
    r""" Moni cycle model, simplest rear axis center model for car
    state: [x, y, theta]
    """
    state_dim = 3
    
    def __init__(self, sigma_v, sigma_w):
        self.sigma_v = sigma_v
        self.sigma_w = sigma_w
        
    def rvs(self, mean, cov, num_samples):
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T
        
    def noise_covar(self, x, time_step):
        # additive noise form
        theta = x[2]
        G = np.array([[time_step * np.cos(theta), 0],
                      [time_step * np.sin(theta), 0],
                      [0, time_step]])
        covar = G.dot(np.diag(np.array([self.sigma_v ** 2,
                                        self.sigma_w ** 2]))).dot(G.T)
        return covar
    
    def forward(self, state, vel, yaw_rate, time_step, noisy=False):
        r"""

        Parameters
        ----------
        state: State, shape [state_dim. num_states]
        vel: input velocity
        yaw_rate: input yaw rate
        time_step: scalar, timestamp
        noisy: bool

        Returns
        -------
        State
        """
        x, y, theta = state
        x += vel * time_step * np.cos(theta)
        y += vel * time_step * np.sin(theta)
        theta += yaw_rate * time_step
        # rem2pi
        theta = rem2pi(theta)
        state_next = np.vstack([x, y, theta])
    
        if noisy:
            noise = self.rvs(np.zeros(self.ndim, dtype=np.float_),
                             self.noise_covar(state, time_step),
                             x.shape[1])
            return state_next + noise
        else:
            return state_next
        
    def jacobian(self, x, vel, yaw_rate, time_step):
        r"""

        Parameters
        ----------
        x: State
        time_step: scalar

        Returns
        -------
        Jacobian matrix
        """
        if x.ndim == 2:
            assert x.shape[1] == 1
        elif x.ndim > 2:
            raise ValueError
        T = time_step
        return np.array([[1, 0, -T * vel * np.sin(x[2])],
                         [0, 1, T * vel * np.cos(x[2])],
                         [0, 0, 1]])
        
    @property
    def ndim(self):
        return self.state_dim
        

class SimpleCTRVModel(NonlinearTransitionModel):
    r"""constant turn rate and velocity transition model
    
    state = [x, y, v, \theta, \omega]
    x, y prediction is simplified with assumption yaw rate \omega = 0
    """
    state_dim = 5

    def __init__(self, sigma_vel, sigma_omega):
        self.sigma_vel = sigma_vel
        self.sigma_omega = sigma_omega

    def rvs(self, mean, cov, num_samples):
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T

    def noise_covar(self, time_step):
        # additive noise form
        G = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 0],
                      [0, 1]])
        covar = G.dot(np.diag(np.array([self.sigma_vel ** 2,
                                        self.sigma_omega ** 2]))).dot(G.T)
        return covar

    def forward(self, x, time_step, noisy=False):
        r"""

        Parameters
        ----------
        x: State, shape [state_dim. num_states]
        time_step: scalar
        noisy: bool

        Returns
        -------
        State
        """
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        T = time_step
        num_states = x.shape[1]
        x_next = x + np.stack([T * x[2] * np.cos(x[3]),
                               T * x[2] * np.sin(x[3]),
                               np.zeros(num_states, dtype=x.dtype),
                               T * x[4],
                               np.zeros(num_states, dtype=x.dtype)], axis=0)
        x_next[3] = rem2pi(x_next[3])
        
        if noisy:
            noise = self.rvs(np.zeros(self.ndim, dtype=np.float_),
                             self.noise_covar(T),
                             x.shape[1])
            return x_next + noise
        else:
            return x_next

    def jacobian(self, x, time_step):
        r"""

        Parameters
        ----------
        x: State
        time_step: scalar

        Returns
        -------
        Jacobian matrix
        """
        if x.ndim == 2:
            assert x.shape[1] == 1
        elif x.ndim > 2:
            raise ValueError
        T = time_step
        return np.array([[1, 0, T * np.cos(x[3]), -T * x[2] * np.sin(x[3]), 0],
                         [0, 1, T * np.sin(x[3]), T * x[2] * np.cos(x[3]), 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, T],
                         [0, 0, 0, 0, 1]])

    @property
    def ndim(self):
        return self.state_dim


class ConstantTurnRateVelocityModel(NonlinearTransitionModel):
    r"""constant turn rate transition model
    
    state = [x, y, v, \theta, \omega]
    """
    state_dim = 5

    def __init__(self, sigma_acc, sigma_omega):
        self.sigma_acc = sigma_acc
        self.sigma_omega = sigma_omega

    def rvs(self, mean, cov, num_samples):
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T

    def noise_covar(self, time_step, state=None):
        # additive noise form
        T = time_step
        theta = state[3]
        G = np.array([[1/2*T**2*np.cos(theta), 0],
                      [1/2*T**2*np.sin(theta), 0],
                      [T, 0],
                      [0, 1/2*T**2],
                      [0, T]])
        covar = G.dot(np.diag([self.sigma_acc ** 2,
                               self.sigma_omega ** 2])).dot(G.T)
        return covar

    def forward(self, x, time_step, noisy=False):
        r"""

        Parameters
        ----------
        x: State of shape [state_dim. num_states]
        time_step: scalar
        noisy: bool

        Returns
        -------
        State
        """
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        T = time_step
        num_states = x.shape[1]
        # find the state with yaw rate \approx 0
        idx0 = x[-1, :] == approx(0)
        x_next_0 = x[:,idx0] + np.stack([T * x[2,idx0] * np.cos(x[3,idx0]),
                               T * x[2,idx0] * np.sin(x[3,idx0]),
                               np.zeros(num_states, dtype=x.dtype),
                               T * x[4,idx0],
                               np.zeros(num_states, dtype=x.dtype)], axis=0)
    
        # find the state with yaw rate \ne 0
        idx1 = ~idx0
        # TODO: W.I.P....
        x_next_1 = x[:,idx1] + 0 #W.I.P....
        x_next = np.zeros_like(x)
        x[:,idx0] = x_next_0
        x[:,idx1] = x_next_1
        if noisy:
            noise = self.rvs(np.zeros(self.ndim, dtype=np.float_),
                             self.noise_covar(T, x),
                             x.shape[1])
            return x_next + noise
        else:
            return x_next

    def jacobian(self, x, time_step):
        r"""

        Parameters
        ----------
        x: State
        time_step: scalar

        Returns
        -------
        Jacobian matrix
        """
        if x.ndim == 2:
            assert x.shape[1] == 1
        elif x.ndim > 2:
            raise ValueError
        T = time_step
        return np.array([[1, 0, T * np.cos(x[3]), -T * x[2] * np.sin(x[3]), 0],
                         [0, 1, T * np.sin(x[3]), T * x[2] * np.cos(x[3]), 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, T],
                         [0, 0, 0, 0, 1]])

    @property
    def ndim(self):
        return self.state_dim
