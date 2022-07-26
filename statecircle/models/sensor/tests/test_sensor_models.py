#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import pytest
import numpy as np

from ..base import DummySensorModel
from statecircle.models.measurement.clutter import PoissonClutterModel
from statecircle.models.transition.linear import ConstantVelocityModel
from statecircle.models.measurement.linear import LinearMeasurementModel
from statecircle.datasets.base import SimulatedGroundTruthDataGenerator


@pytest.fixture()
def input_paras():
    P_D = 0.7
    lambda_clutter = 20
    scope = np.array([[0, 1000], [0, 1000]])
    clutter_model = PoissonClutterModel(P_D, lambda_clutter, scope)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0], sigma=5)
    return clutter_model, measurement_model

@pytest.fixture()
def make_data_generator():
    initial_states = np.random.rand(4, 5)
    birth_times = [10, 20, 30, 40, 50]
    death_times = [40, 50, 60, 70, 80]
    time_scope = [0, 100]
    transition_model = ConstantVelocityModel(sigma=5)
    data_gen = SimulatedGroundTruthDataGenerator(initial_states, birth_times, death_times,
                                                 time_scope, transition_model, noisy=False)
    return data_gen


def test_dummy_sensor_model(input_paras, make_data_generator):
    clutter_model, measurement_model = input_paras
    data_gen = make_data_generator
    sensor_model = DummySensorModel(clutter_model, measurement_model)
    meas_data, obj_meas_data, clutter_data = sensor_model.detect(data_gen.gt_series)
    for meas, obj_meas, c_meas in zip(meas_data.datum, obj_meas_data, clutter_data):
        assert meas.shape[1] == obj_meas.shape[1] + c_meas.shape[1]

    data_reader = sensor_model.detect_iter(data_gen)
    print(len(list(data_reader)))
    for meas in data_reader:
        meas_data, obj_meas, clutter_meas = meas
        assert meas_data.shape[1] == obj_meas.shape[1] + clutter_meas.shape[1]


