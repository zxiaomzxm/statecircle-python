#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import pytest
import numpy as np

from ..base import MeasurementReader
from statecircle.models.sensor.base import DummySensorModel
from statecircle.models.measurement.clutter import PoissonClutterModel
from statecircle.models.measurement.linear import LinearMeasurementModel
from statecircle.datasets.base import SimulatedGroundTruthDataGenerator
from statecircle.models.transition.linear import ConstantVelocityModel


@pytest.fixture()
def data_generator():
    initial_states = np.random.rand(4, 5)
    birth_times = [10, 20, 30, 40, 50]
    death_times = [40, 50, 60, 70, 80]
    time_scope = [0, 100]
    transition_model = ConstantVelocityModel(sigma=5)
    return SimulatedGroundTruthDataGenerator(initial_states, birth_times, death_times,
                                             time_scope, transition_model, noisy=False)

@pytest.fixture()
def sensor_model():
    P_D = 0.7
    lambda_clutter = 20
    scope = np.array([[0, 1000], [0, 1000]])
    clutter_model = PoissonClutterModel(P_D, lambda_clutter, scope)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0], sigma=5)
    return DummySensorModel(clutter_model, measurement_model)

def test_measurement_model(data_generator, sensor_model):
    reader = MeasurementReader(data_generator, sensor_model)
    assert len(reader) == len(data_generator)
    for meas in reader:
        pass

    meas = next(iter(reader))
    assert meas is not None

    for meas in reader:
        assert meas.shape[0] == sensor_model.measurement_model.ndim_meas

    for meas, obj_meas, clutter_meas in reader.truth_meas_generator:
        assert meas.shape[1] == obj_meas.shape[1] + clutter_meas.shape[1]