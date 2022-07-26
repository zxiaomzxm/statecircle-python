#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import pytest
import numpy as np

from ..base import SimulatedGroundTruthDataGenerator
from statecircle.models.transition.linear import ConstantVelocityModel


@pytest.fixture()
def input_paras():
    initial_states = np.random.rand(4, 5)
    birth_times = [10, 20, 30, 40, 50]
    death_times = [40, 50, 60, 70, 80]
    time_scope = [0, 100]
    transition_model = ConstantVelocityModel(sigma=5)
    return initial_states, birth_times, death_times, time_scope, transition_model

def test_dummy_gt_data_generator(input_paras):
    initial_states, birth_times, death_times, time_scope, transition_model = input_paras
    data_gen = SimulatedGroundTruthDataGenerator(initial_states, birth_times, death_times,
                                                 time_scope, transition_model, noisy=False)

    for data in data_gen:
        if data is not None:
            print(data.state.shape)

    for timestamp, datum in zip(data_gen.gt_series.timestamps, data_gen.gt_series.datum):
        for data in datum:
            assert timestamp == data.timestamp

    assert len(data_gen.gt_series.datum[5]) == 0
    assert len(data_gen.gt_series.datum[15]) == 1
    assert len(data_gen.gt_series.datum[25]) == 2
    assert len(data_gen.gt_series.datum[35]) == 3
    assert len(data_gen.gt_series.datum[45]) == 3
    assert len(data_gen.gt_series.datum[55]) == 3
    assert len(data_gen.gt_series.datum[65]) == 2
    assert len(data_gen.gt_series.datum[75]) == 1
    assert len(data_gen.gt_series.datum[85]) == 0







