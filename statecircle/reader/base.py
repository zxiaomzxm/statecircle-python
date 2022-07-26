#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from ..base import Base


class Reader(Base):
    r""""Reader base class"""


class MeasurementReader(Base):
    r"""Measurement reader"""
    def __init__(self, data_generator, sensor_model):
        self.data_generator = data_generator
        self.sensor_model = sensor_model

    def __len__(self):
        return len(self.data_generator)

    def __iter__(self):
        for meas, *_ in self.sensor_model.detect_iter(self.data_generator):
            yield meas

    def truth_meas_generator(self):
        for meas, obj_meas, clutter_meas in self.sensor_model.detect_iter(self.data_generator):
            yield meas, obj_meas, clutter_meas




