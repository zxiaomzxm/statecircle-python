#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
from operator import __ne__
from ..linear import LinearMeasurementModel
import pytest

@pytest.fixture(scope='module')
def state_measurement_model_testing(x, observered_index):
    meas_model = LinearMeasurementModel(mapping=observered_index,
                                       sigma=1)
    z = meas_model.forward(x, noisy=False)
    x_inv = meas_model.reverse(z)
    z_noise1 = meas_model.forward(x, noisy=True)
    z_noise2 = meas_model.forward(x, noisy=True)
    np.testing.assert_almost_equal(meas_model.jacobian(x), meas_model.meas_mat)
    x[~observered_index] = 0
    assert x.shape[0] == meas_model.ndim
    assert z.shape[0] == meas_model.ndim_meas
    assert x.shape[1] == z.shape[1]
    np.testing.assert_almost_equal(x, x_inv)
    np.testing.assert_array_compare(__ne__, z_noise1, z_noise2)


def test_state_measurement_model_case1():
    x = np.random.rand(4, 1)
    observered_index = np.array([1, 1, 0, 0], dtype=np.bool)
    state_measurement_model_testing(x, observered_index)


def test_state_measurement_model_case2():
    x = np.random.rand(4, 2)
    observered_index = np.array([1, 0, 1, 0], dtype=np.bool)
    state_measurement_model_testing(x, observered_index)


def test_state_measurement_model_case3():
    x = np.random.rand(5, 1)
    observered_index = np.array([1, 1, 0, 0, 0], dtype=np.bool)
    state_measurement_model_testing(x, observered_index)


def test_state_measurement_model_case4():
    x = np.random.rand(5, 3)
    observered_index = np.array([1, 0, 0, 1, 1], dtype=np.bool)
    state_measurement_model_testing(x, observered_index)
