#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
import pytest
from operator import __ne__
import datetime

from ..linear import ConstantVelocityModel


def constant_velocity_model_testing(x, step):
    trans_model = ConstantVelocityModel(sigma=1)
    x_next = trans_model.forward(x, step, noisy=False)
    x_next_noise = trans_model.forward(x, step, noisy=True)
    np.testing.assert_almost_equal(trans_model.jacobian(x, step),
                                   trans_model.transition_matrix(step))
    assert x.shape[0] == trans_model.state_dim
    assert x_next.shape[0] == trans_model.ndim
    np.testing.assert_array_compare(__ne__, x_next, x_next_noise)


def test_constant_velocity_model_case1():
    state = np.random.rand(4, 1)
    old_timestamp = datetime.datetime.now()
    timediff = 1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    step = (new_timestamp - old_timestamp).total_seconds()
    constant_velocity_model_testing(state, step)


def test_constant_velocity_model_case2():
    state = np.random.rand(4, 2)
    old_timestamp = datetime.datetime.now()
    timediff = 10  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    step = (new_timestamp - old_timestamp).total_seconds()
    constant_velocity_model_testing(state, step)
