#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from abc import abstractmethod
import numpy as np

from ..base import Base
from statecircle.types.data import GroundTruthData, GroundTruthSeries


class SENSOR_TYPE:
    CAMERA = 0
    MOBILEYE = 1
    RADAR = 2
    CONTI_RADAR = 3
    WHST_RADAR = 4
    UNKNOWN = -1
    FUSED = 666

    CAMERA_TYPES = [CAMERA, MOBILEYE, FUSED]
    RADAR_TYPES = [RADAR, CONTI_RADAR, WHST_RADAR, FUSED]


class DataGenerator(Base):
    r"""Data generator base class"""

    @abstractmethod
    def generate(self):
        pass


class SimulatedGroundTruthDataGenerator(Base):
    r"""Simulated ground truth data generator"""

    def __init__(self, scene, transition_model, noisy=False):
        r"""

        Parameters
        ----------
        scene : Scennario class
            - initial_states 
            - birth_times
            - death_times
            - time_range
        transition_model
        noisy
        """
        initial_states = scene.initial_states[:, None] if scene.initial_states.ndim == 1 else scene.initial_states
        self.obj_num = initial_states.shape[1]
        self.initial_states = initial_states
        self.birth_times = scene.birth_times
        self.death_times = scene.death_times
        assert len(scene.time_range) == 2 and scene.time_range[1] >= scene.time_range[0]
        self.time_range = scene.time_range
        self.time_len = scene.time_range[1] - scene.time_range[0]
        self.transition_model = transition_model

        self.gt_series = self.initialize_trajectories(noisy)

    @property
    def ndim(self):
        return self.initial_states.shape[0]

    def __len__(self):
        return int(self.time_len)

    def initialize_trajectories(self, noisy=False):
        gt_series = GroundTruthSeries(time_len=self.time_len)

        for k in np.arange(self.time_range[0], self.time_range[1], dtype=np.int_):
            gt_series.timestamps[k] = k

        for i in range(self.obj_num):
            obj_state = self.initial_states[:, i][:, None]
            for k in np.arange(max(self.birth_times[i], self.time_range[0]),
                               min((self.death_times[i], self.time_range[1])),
                               dtype=np.int_):
                time_step = 1
                if noisy:
                    obj_state = np.random.multivariate_normal(self.transition_model.forward(obj_state, time_step)[:, 0],
                                                              self.transition_model.noise_covar(time_step=time_step))[:, None]
                else:
                    obj_state = self.transition_model.forward(obj_state, time_step)

                gt_series.datum[k].append(GroundTruthData(timestamp=k, states=obj_state))
                gt_series.num[k] += 1

        return gt_series

    def __getitem__(self, idx):
        if len(self.gt_series.datum[idx]) == 0:
            return GroundTruthData(self.gt_series.timestamps[idx], None)

        states = np.hstack([ele.states for ele in self.gt_series.datum[idx]])
        timestamp = self.gt_series.timestamps[idx]

        return GroundTruthData(timestamp, states)
