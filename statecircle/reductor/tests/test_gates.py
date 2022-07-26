#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 16:00:54 2019

@author: zhxm
"""
import numpy as np
from scipy.stats import multivariate_normal
from pytest import approx

from ..gate import EllipsoidalGate, RectangularGate
from statecircle.types.state import GaussianState
from statecircle.models.measurement.linear import LinearMeasurementModel


def test_retangular_gate():
    pass


def test_ellipsoidal_gate():
    ndim = 4
    ndim_meas = 2
    num_samples = 10000
    percentile = 0.3
    gate = EllipsoidalGate(percentile=percentile)
    mean = np.random.rand(ndim, 1)
    cov = np.eye(4)
    state = GaussianState(mean, cov)
    meas_model = LinearMeasurementModel(mapping=[1, 1, 0, 0], sigma=2)
    input_states = multivariate_normal.rvs(mean[:, 0], cov, num_samples).T
    meas = meas_model.forward(input_states, noisy=True)
    meas_ingate, meas_index = gate.gating(state, meas, meas_model)
    assert meas_ingate.shape[1] == meas_index.sum()
    assert meas_ingate.shape[1] / num_samples == approx(percentile, rel=1e-1)
