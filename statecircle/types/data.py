#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np

from .base import Type


class GroundTruthData(Type):
    r"""Ground truth data base type"""

    def __init__(self, timestamp, states):
        r"""

        Parameters
        ----------
        timestamp : datatime
        states : State
            set of states
        """
        self.timestamp = timestamp
        self.states = states


class MeasurementData(Type):
    r"""Measurement data base type"""

    def __init__(self, timestamp, meas):
        r"""

        Parameters
        ----------
        timestamp
        meas
        """
        self.timestamp = timestamp
        self.meas = meas


class DataSeries(Type):
    r"""Ground truth data series"""

    def __init__(self, time_len):
        self.timestamps = [None for _ in range(time_len)]
        self.datum = [[] for _ in range(time_len)]
        self.num = np.zeros(time_len, dtype=np.int_)
        
    def __getitem__(self, idx):
        return GroundTruthData(self.timestamps[idx], self.datum[idx])


class GroundTruthSeries(DataSeries):
    r"""Ground truth series"""


class MeasurementSeries(DataSeries):
    r"""Measurement series"""
