#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
import matplotlib.pyplot as plt

from scenarios.linear import LinearScenario
from scenarios.nonlinear import NonlinearScenario
from tools.visualizer import Visualizer
from statecircle.datasets.base import SimulatedGroundTruthDataGenerator
from statecircle.models.sensor.base import DummySensorModel
from statecircle.platform.base import TrackerPlatform
from statecircle.reader.base import MeasurementReader


# np.random.seed(9)

# build tracker

#scene = LinearScenario.caseA()
#config_path = 'config/pda_linear.yml'

scene = NonlinearScenario.caseA()
config_path = 'config/gs_nonlinear.yml'

platform = TrackerPlatform(config_path)
    
# build data generator
data_generator = SimulatedGroundTruthDataGenerator(scene,
                                                   platform.transition_model,
                                                   noisy=False)

# build sensor model
sensor_model = DummySensorModel(platform.clutter_model, platform.measurement_model)

# build data reader
data_reader = MeasurementReader(data_generator, sensor_model)

# filtering
estimates = platform.tracker.filtering(data_reader)

card_pred = [ele.shape[1] for ele in estimates]
estimates = np.hstack(estimates)

# %% visualise results
visualizer = Visualizer(data_generator, config_path)
visualizer.show_estimates(estimates)
visualizer.show_cardinality(card_pred)
visualizer.show_measurements(data_reader)

