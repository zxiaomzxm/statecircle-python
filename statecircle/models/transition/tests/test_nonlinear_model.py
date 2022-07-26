#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
import pytest
from operator import __ne__
import datetime

from ..nonlinear import SimpleCTRVModel


def constant_turnrate_model_testing(x, step):
    trans_model = SimpleCTRVModel(sigma_vel=1, sigma_omega=2)
    x_next = trans_model.forward(x, step, noisy=False)
    x_next_noise = trans_model.forward(x, step, noisy=True)

    assert x.shape[0] == trans_model.state_dim
    assert x_next.shape[0] == trans_model.ndim
    # assert (x_next - x_next_noise).sum() > 1e-10

    if x.shape[1] == 1:
        x_delta = 0.01 * x
        x_next_delta = trans_model.forward(x + x_delta, step) - \
                       trans_model.forward(x, step)
        x_next_delta_approx = trans_model.jacobian(x, step).dot(x_delta)
        np.testing.assert_almost_equal(x_next_delta, x_next_delta_approx, decimal=3)


def test_constant_turnrate_model_case1():
    state = np.random.rand(5, 1)
    old_timestamp = datetime.datetime.now()
    timediff = 0.1  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    step = (new_timestamp - old_timestamp).total_seconds()
    constant_turnrate_model_testing(state, step)


def test_constant_turnrate_model_case2():
    state = np.random.rand(5, 2)
    old_timestamp = datetime.datetime.now()
    timediff = 10  # 1sec
    new_timestamp = old_timestamp + datetime.timedelta(seconds=timediff)
    step = (new_timestamp - old_timestamp).total_seconds()
    constant_turnrate_model_testing(state, step)
