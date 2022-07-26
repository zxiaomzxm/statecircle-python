#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from abc import abstractmethod
import numpy as np

from statecircle.types.data import MeasurementSeries, MeasurementData
from ..base import Model


class SensorModel(Model):
    r""""Sensor model base class"""

    @abstractmethod
    def detect(self):
        r"""virtual detect method"""


class DummySensorModel(Model):
    r"""Generated simulated sensor data from ground truth data"""

    def __init__(self, clutter_model, measurement_model, random_seed=None, noisy=True):
        self.clutter_model = clutter_model
        self.measurement_model = measurement_model
        self.seed = random_seed or np.random.randint(2**32)
        self.noisy = noisy
        
    def detect(self, gt_series, shuffle=True):
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # initialize memory
        data_len = len(gt_series.datum)
        # meas_data = [[] for _ in range(data_len)]
        meas_data = MeasurementSeries(data_len)
        obj_meas_data = [np.empty((self.measurement_model.ndim_meas, 0)) for _ in range(data_len)]
        clutter_data = [np.empty((self.measurement_model.ndim_meas, 0)) for _ in range(data_len)]

        # generate measurements
        for k in range(data_len):
            meas_data.timestamps[k] = gt_series.timestamps[k]
            if gt_series.num[k] > 0:
                idx = np.random.rand(gt_series.num[k]) < self.clutter_model.detection_rate
                # only generate object-orienated observations for detected objects
                if any(idx):
                    obj_states = np.hstack([gt_data.states for i, gt_data in enumerate(gt_series.datum[k]) if idx[i]])
                    for i in range(obj_states.shape[1]):
                        if self.noisy:
                            meas = np.random.multivariate_normal(
                                self.measurement_model.forward(obj_states[:, i])[:, 0],
                                self.measurement_model.noise_covar())[:, None]
                        else:
                            meas = self.measurement_model.forward(obj_states[:, i])[:, 0]
                        meas_data.datum[k].append(meas)

            # number of clutter measurements
            num_clutter = np.random.poisson(self.clutter_model.lambda_clutter)

            # generate clutter
            clutter = np.tile(self.clutter_model.scope[:, 0], [num_clutter, 1]).T + \
                      np.diag(self.clutter_model.scope.dot(np.array([-1, 1]))). \
                          dot(np.random.rand(self.measurement_model.ndim_meas, num_clutter))

            if len(meas_data.datum[k]) == 0:
                # no detection measurement
                clutter_data[k] = meas_data.datum[k] = clutter
            else:
                obj_meas_data[k] = meas_data.datum[k] = np.hstack(meas_data.datum[k])
                # total measurements are the union of object detections and clutter
                meas_data.datum[k] = np.hstack((meas_data.datum[k], clutter))
                clutter_data[k] = clutter

                if shuffle:
                    meas_data.datum[k] = np.random.permutation(meas_data.datum[k].T).T

        return meas_data, obj_meas_data, clutter_data

    def detect_iter(self, data_generator, shuffle=True):
        if self.seed is not None:
            np.random.seed(self.seed)
            
        for data in data_generator:
            obj_meas = np.empty([self.measurement_model.ndim_meas, 0])
            if data.states is not None:
                idx = np.random.rand(data.states.shape[1]) < self.clutter_model.detection_rate
                # only generate object-orienated observations for detected objects
                if any(idx):
                    obj_states = data.states[:, idx]
                    for i in range(obj_states.shape[1]):
                        if self.noisy:
                            obj_meas = np.hstack((obj_meas, np.random.multivariate_normal(
                                                  self.measurement_model.forward(obj_states[:, i])[:, 0],
                                                  self.measurement_model.noise_covar())[:, None]))
                        else:
                            obj_meas = np.hstack((obj_meas, self.measurement_model.forward(obj_states[:, i])))

            # number of clutter measurements
            num_clutter = np.random.poisson(self.clutter_model.lambda_clutter)

            # generate clutter
            clutter_meas = np.tile(self.clutter_model.scope[:, 0], [num_clutter, 1]).T + \
                           np.diag(self.clutter_model.scope.dot(np.array([-1, 1]))). \
                               dot(np.random.rand(self.measurement_model.ndim_meas, num_clutter))

            meas_data = MeasurementData(timestamp=data.timestamp, meas=np.hstack((obj_meas, clutter_meas)))

            if shuffle:
                meas_data.meas = np.random.permutation(meas_data.meas.T).T

            yield meas_data, obj_meas, clutter_meas
