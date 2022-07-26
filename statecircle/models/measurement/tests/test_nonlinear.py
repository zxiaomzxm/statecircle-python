#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""

import numpy as np
from operator import __ne__
from statecircle.models.measurement.nonlinear import *


def test_range_bearing_measurement_model_1state():
    sigma_range = 1.
    sigma_bearing = 2.
    origin = np.array([10, 20])
    meas_model = RangeBearningMeasurementModel(sigma_range, sigma_bearing, origin)

    x = np.random.rand(5, 1)
    z = meas_model.forward(x, noisy=False)
    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)

    x_delta = 0.01 * x
    z_delta = meas_model.forward(x + x_delta) - meas_model.forward(x)
    jacobian_mat = meas_model.jacobian(x)
    z_delta_approx = jacobian_mat.dot(x_delta)
    x_inv = meas_model.reverse(z)

    assert x.shape[0] == meas_model.state_dim == meas_model.ndim
    assert z.shape[0] == meas_model.meas_dim == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)
    np.testing.assert_almost_equal(z_delta, z_delta_approx, decimal=3)
    assert jacobian_mat.shape == (meas_model.meas_dim, meas_model.state_dim)
    x[2:] = 0
    np.testing.assert_almost_equal(x, x_inv)


def test_range_bearing_measurement_model_2state():
    sigma_range = 1.
    sigma_bearing = 2.
    origin = np.array([10, 20])
    meas_model = RangeBearningMeasurementModel(sigma_range, sigma_bearing, origin)

    x = np.random.rand(5, 2)
    z = meas_model.forward(x, noisy=False)
    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)
    x_inv = meas_model.reverse(z)
    assert x.shape[0] == meas_model.state_dim == meas_model.ndim
    assert z.shape[0] == meas_model.meas_dim == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)
    x[2:] = 0
    np.testing.assert_almost_equal(x, x_inv)
    
    
def test_range_bearing_2d_model_1state():
    noise_cov = np.diag([1, 1])
    meas_model = RangeBearing2DMeasurementModel(noise_cov)

    x = np.random.rand(4, 1)
    z = meas_model.forward(x, noisy=False)
    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)

    x_delta = 0.01 * x
    z_delta = meas_model.forward(x + x_delta) - meas_model.forward(x)
    jacobian_mat = meas_model.jacobian(x)
    z_delta_approx = jacobian_mat.dot(x_delta)
    x_inv = meas_model.reverse(z)

    assert x.shape[0] == meas_model.state_dim == meas_model.ndim
    assert z.shape[0] == meas_model.meas_dim == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)
    np.testing.assert_almost_equal(z_delta, z_delta_approx, decimal=3)
    assert jacobian_mat.shape == (meas_model.meas_dim, meas_model.state_dim)
    x[2:] = 0
    np.testing.assert_almost_equal(x, x_inv)


def test_range_bearing_2d_model_3state():
    noise_cov = np.diag([1, 1])
    meas_model = RangeBearing2DMeasurementModel(noise_cov)

    x = np.random.rand(4, 3)
    z = meas_model.forward(x, noisy=False)
    x_inv = meas_model.reverse(z)

    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)
    
    assert x.shape[0] == meas_model.state_dim == meas_model.ndim
    assert z.shape[0] == meas_model.meas_dim == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)
    x[2:] = 0
    np.testing.assert_almost_equal(x, x_inv)
    
def test_range_bearing_4d_model_1state():
    noise_cov = np.diag([1, 1, 1, 1])
    meas_model = RangeBearing4DMeasurementModel(noise_cov)

    x = np.random.rand(4, 1)
    z = meas_model.forward(x, noisy=False)
    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)

    x_delta = 0.01 * x
    z_delta = meas_model.forward(x + x_delta) - meas_model.forward(x)
    jacobian_mat = meas_model.jacobian(x)
    z_delta_approx = jacobian_mat.dot(x_delta)
    x_inv = meas_model.reverse(z)

    assert x.shape[0] == meas_model.state_dim == meas_model.ndim
    assert z.shape[0] == meas_model.meas_dim == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)
    np.testing.assert_almost_equal(z_delta, z_delta_approx, decimal=3)
    assert jacobian_mat.shape == (meas_model.meas_dim, meas_model.state_dim)
    np.testing.assert_almost_equal(x, x_inv)
    

def test_range_bearing_4d_model_3state():
    noise_cov = np.diag([1, 1, 1, 1])
    meas_model = RangeBearing4DMeasurementModel(noise_cov)

    x = np.random.rand(4, 3)
    z = meas_model.forward(x, noisy=False)
    x_inv = meas_model.reverse(z)

    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)
    
    assert x.shape[0] == meas_model.state_dim == meas_model.ndim
    assert z.shape[0] == meas_model.meas_dim == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)
    np.testing.assert_almost_equal(x, x_inv)
    
    
if __name__ == "__main__":
    test_range_bearing_measurement_model_1state()
    test_range_bearing_measurement_model_2state()
    
    test_range_bearing_2d_model_1state()
    test_range_bearing_2d_model_3state()
    
    test_range_bearing_4d_model_1state()
    test_range_bearing_4d_model_3state()
