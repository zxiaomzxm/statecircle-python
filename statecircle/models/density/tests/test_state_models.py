#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
import pytest

from ..kalman import KalmanDensityModel
from statecircle.types.state import GaussianState, GaussianLabeledState
from statecircle.models.transition.linear import ConstantVelocityModel
from statecircle.models.measurement.linear import LinearMeasurementModel


@pytest.fixture()
def input_gaussian_state():
    ndim = 4
    mean = np.array([1, 2, 3, 4])[:, None]
    cov = np.eye(ndim)
    state = GaussianState(mean, cov)
    time_step = 0.1
    return state, time_step

@pytest.fixture()
def input_gaussian_labeled_state():
    ndim = 4
    mean = np.array([1, 2, 3, 4])[:, None]
    cov = np.eye(ndim)
    label = 10
    state = GaussianLabeledState(label, mean, cov)
    time_step = 0.1
    return state, time_step


def test_kalman_state_model_gaussian_state(input_gaussian_state):
    state, time_step = input_gaussian_state

    transition_model = ConstantVelocityModel(sigma=10)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0], sigma=5)
    state_model = KalmanDensityModel()
    single_meas = measurement_model.forward(state.mean, noisy=True)
    some_meas = measurement_model.forward(np.repeat(state.mean, 10, axis=1), noisy=True)

    state_pred = state_model.predict(state, time_step, transition_model)
    state_upd = state_model.update(state, single_meas, measurement_model)
    loglik = state_model.predicted_log_likelihood(state, some_meas, measurement_model)
    # print(state_pred)
    # print(state_upd)
    # print(loglik)


def test_kalman_state_model_labeled_gaussian_state(input_gaussian_labeled_state):
    state, time_step = input_gaussian_labeled_state

    transition_model = ConstantVelocityModel(sigma=10)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0], sigma=5)
    state_model = KalmanDensityModel()
    single_meas = measurement_model.forward(state.mean, noisy=True)
    some_meas = measurement_model.forward(np.repeat(state.mean, 10, axis=1), noisy=True)

    state_pred = state_model.predict(state, time_step, transition_model)
    state_upd = state_model.update(state, single_meas, measurement_model)
    loglik = state_model.predicted_log_likelihood(state, some_meas, measurement_model)
    # print(state_pred)
    # print(state_upd)
    # print(loglik)
