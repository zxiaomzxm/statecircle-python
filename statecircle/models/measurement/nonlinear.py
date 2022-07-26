#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
from scipy.stats import multivariate_normal
from abc import abstractmethod

from .base import MeasurementModel

class NonlinearMeasurementModel(MeasurementModel):
    r"""Nonlinear meansuremnent models base"""

    def measurement_matrix(self, *args, **kwargs):
        return self.jacobian(*args, **kwargs)

    @abstractmethod
    def jacobian(self, *args, **kwargs):
        r"""virtual jacobian matrix method"""


class RangeBearningMeasurementModel(NonlinearMeasurementModel):
    r"""Radar range&bearing mearsurement models"""

    state_dim = 5  # state_dim for range barning measurement model is constant 5
    meas_dim = 2  # meas_dim for range barning measurement model is constant 2

    def __init__(self, sigma_range, sigma_bearing, origin=[0,0]):
        r"""
        :param sigma_range:
        :param sigma_bearing:
        :param origin:
        """
        assert sigma_range > 0
        assert sigma_bearing > 0
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        self.origin = np.atleast_2d(origin).T

        # measurement noise covariance matrix
        self.noise_cov = np.array([[self.sigma_range ** 2, 0],
                                   [0, self.sigma_bearing ** 2]], dtype=np.float_)

    def noise_covar(self, *args, **kwargs):
        return self.noise_cov

    def rvs(self, mean, cov, num_samples):
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T

    def _range(self, x):
        return np.linalg.norm(x[:2] - self.origin, axis=0)

    def _bearing(self, x):
        return np.arctan2(x[1] - self.origin[1], x[0] - self.origin[0])

    def jacobian(self, x):
        rng = self._range
        s = self.origin
        if x.ndim == 2:
            assert x.shape[1] == 1, "jacobian method only support one state vector"
        elif x.ndim > 2:
            raise ValueError
            
        j11 = ((x[0] - s[0]) / rng(x))[0]
        j12 = ((x[1] - s[1]) / rng(x))[0]
        j21 = (-(x[1] - s[1]) / (rng(x) ** 2))[0]
        j22 = ((x[0] - s[0]) / (rng(x) ** 2))[0]
        return np.array([[j11, j12, 0, 0, 0],
                         [j21, j22, 0, 0, 0]])

    def forward(self, x, noisy=False):
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        num_states = x.shape[1]
        if noisy:
            noisy = self.rvs(np.zeros(self.ndim_meas, dtype=np.float_),
                                 self.noise_cov,
                                 num_states)
            return np.stack((self._range(x), self._bearing(x)), axis=0) + noisy
        else:
            return np.stack((self._range(x), self._bearing(x)), axis=0)

    def reverse(self, z):
        rng, ber = z[0], z[1]
        return np.vstack((np.atleast_2d(rng * np.cos(ber) + self.origin[0]),
                          np.atleast_2d(rng * np.sin(ber) + self.origin[1]),
                          np.zeros((self.state_dim - self.meas_dim, z.shape[1]), dtype=np.float_)
                          ))

    @property
    def ndim(self):
        return self.state_dim

    @property
    def ndim_meas(self):
        return self.meas_dim
    

class SimpleRangeBearingMeasurementModel(RangeBearningMeasurementModel):
    r""" state: [x, y, theta]
         meas:  [range, bearing]
    """
    state_dim = 3  # state_dim for range barning measurement model is constant 3
    meas_dim = 2  # meas_dim for range barning measurement model is constant 2
    
    def forward(self, x, noisy=False):
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        num_states = x.shape[1]
        if noisy:
            noisy = self.rvs(np.zeros(self.ndim_meas, dtype=np.float_),
                                 self.noise_cov,
                                 num_states)
            return np.stack((self._range(x), self._bearing(x)), axis=0) + noisy
        else:
            return np.stack((self._range(x), self._bearing(x)), axis=0)
        
    def jacobian(self, x):
        rng = self._range
        s = self.origin
        if x.ndim == 2:
            assert x.shape[1] == 1, "jacobian method only support one state vector"
        elif x.ndim > 2:
            raise ValueError
            
        j11 = ((x[0] - s[0]) / rng(x))[0]
        j12 = ((x[1] - s[1]) / rng(x))[0]
        j21 = (-(x[1] - s[1]) / (rng(x) ** 2))[0]
        j22 = ((x[0] - s[0]) / (rng(x) ** 2))[0]
        return np.array([[j11, j12,  0],
                         [j21, j22, -1]])
    
    def reverse(self, z, pose):
        dist, bearing = z
        rx, ry, theta = pose
        mx = rx + dist * np.cos(bearing + theta)
        my = ry + dist * np.sin(bearing + theta)
        return np.array([mx, my])
    
class RangeBearing2DMeasurementModel(NonlinearMeasurementModel):
    r""" state: [px, py, vx, vy]
         meas:  [range, bearing]
    """
    state_dim = 4
    meas_dim = 2
    
    def __init__(self, sigma_range, sigma_bearing):
        r"""
        :param sigma_range:
        :param sigma_bearing:
        :param origin:
        """
        # measurement noise covariance matrix
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing

    def noise_covar(self, mea, *args, **kwargs):
        assert mea.ndim == 2 and mea.shape[1] == 1
        rng, bearing = mea[:, 0]
        sigma_range = np.maximum(self.sigma_range / 10, rng * self.sigma_range)
        sigma_bearing = self.sigma_bearing
        noise_cov = np.diag([sigma_range**2, sigma_bearing**2])
        return noise_cov
    
    def rvs(self, mean, cov, num_samples):
        noise = multivariate_normal.rvs(mean, cov, num_samples)
        return np.atleast_2d(noise).T
    
    def _range(self, x):
        return np.linalg.norm(x[:2], axis=0)
    
    def _bearing(self, x):
        return np.arctan2(x[1], x[0])
    
    def forward(self, x, noisy=False):
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        num_states = x.shape[1]
        if noisy:
            noisy = self.rvs(np.zeros(self.ndim_meas, dtype=np.float_),
                                 self.noise_cov,
                                 num_states)
            return np.stack((self._range(x), self._bearing(x)), axis=0) + noisy
        else:
            return np.stack((self._range(x), self._bearing(x)), axis=0)
        
    def jacobian(self, x):
        rng = self._range
        if x.ndim == 2:
            assert x.shape[1] == 1, "jacobian method only support one state vector"
            x = x[:, 0]
        elif x.ndim > 2:
            raise ValueError

        return np.array([
                        [ x[0] / rng(x),    x[1] / rng(x),    0, 0],
                        [-x[1] / rng(x)**2, x[0] / rng(x)**2, 0, 0]
                        ])
    
    def reverse(self, z):
        assert z.shape[0] == self.meas_dim
        dist, bearing = z
        px = dist * np.cos(bearing)
        py = dist * np.sin(bearing)
        return np.vstack((np.array([px, py]), np.zeros((2, z.shape[1]))))
       
    @property
    def ndim(self):
        return self.state_dim

    @property
    def ndim_meas(self):
        return self.meas_dim 
    
    
class RangeBearing4DMeasurementModel(RangeBearing2DMeasurementModel):
    r""" state: [px, py, vx, vy]
         meas:  [range, bearing, vx, vy]
    """
    state_dim = 4
    meas_dim = 4
    
    def __init__(self, sigma_range, sigma_bearing, sigma_vx, sigma_vy):
        r"""
        :param sigma_range:
        :param sigma_bearing:
        :param origin:
        """
        # measurement noise covariance matrix
        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing
        self.sigma_vx = sigma_vx
        self.sigma_vy = sigma_vy

    def noise_covar(self, mea, *args, **kwargs):
        assert mea.ndim == 2 and mea.shape[1] == 1
        rng, bearing, vx, vy = mea[:, 0]
        if rng > 170:
            sigma_range = 2 * self.sigma_range
        else:
            sigma_range = self.sigma_range
        sigma_bearing = self.sigma_bearing
        sigma_vx = self.sigma_vx
        sigma_vy = self.sigma_vy
        noise_cov = np.diag([sigma_range**2, sigma_bearing**2, sigma_vx**2, sigma_vy**2])
        return noise_cov
    
    def forward(self, x, noisy=False):
        x = x[:, None] if x.ndim == 1 else x
        assert x.shape[0] == self.ndim
        num_states = x.shape[1]
        if noisy:
            noisy = self.rvs(np.zeros(self.ndim_meas, dtype=np.float_),
                                 self.noise_cov,
                                 num_states)
            return np.stack((self._range(x), self._bearing(x), x[2], x[3]), axis=0) + noisy
        else:
            return np.stack((self._range(x), self._bearing(x), x[2], x[3]), axis=0)
        
    def jacobian(self, x):
        rng = self._range
        if x.ndim == 2:
            assert x.shape[1] == 1, "jacobian method only support one state vector"
            x = x[:, 0]
        elif x.ndim > 2:
            raise ValueError
        
        return np.array([
                        [ x[0] / rng(x),    x[1] / rng(x),    0, 0],
                        [-x[1] / rng(x)**2, x[0] / rng(x)**2, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        ])
    
    def reverse(self, z):
        assert z.shape[0] == self.meas_dim
        dist, bearing, vx, vy = z
        px = dist * np.cos(bearing)
        py = dist * np.sin(bearing)
        return np.array([px, py, vx, vy]) 
