#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 24 16:28:42 2019

@author: zhaoxm
"""
import numpy as np
import matplotlib.pyplot as plt

from scenarios.linear import LinearScenario
from scenarios.nonlinear import NonlinearScenario
from statecircle.estimator.base import EAPEstimator
from statecircle.models.birth.base import MultiObjectBirthModel, PoissonBirthModel
from statecircle.models.density.kalman import KalmanDensityModel
from statecircle.models.measurement.nonlinear import RangeBearningMeasurementModel
from statecircle.models.transition.nonlinear import SimpleCTRVModel
from statecircle.reader.base import MeasurementReader
from statecircle.reductor.gate import EllipsoidalGate
from statecircle.models.sensor.base import DummySensorModel
from statecircle.models.measurement.clutter import PoissonClutterModel
from statecircle.models.measurement.linear import LinearMeasurementModel
from statecircle.datasets.base import SimulatedGroundTruthDataGenerator
from statecircle.models.transition.linear import ConstantVelocityModel
from statecircle.reductor.hypothesis_reductor import HypothesisReductor
from statecircle.types.state import GaussianState, GaussianSumState
from statecircle.trackers.mot.phd_filter import PHDFilter
from statecircle.trackers.mot.mbm_filter import MBMFilter
from statecircle.trackers.mot.pmbm_filter import PMBMFilter
from tools.visualizer import Visualizer

seed = None

scenario = 'linear'
# build scene
if scenario == 'linear':
    scene = LinearScenario.caseC(birth_weight=0.03, birth_cov_scale=400)
    birth_model = scene.birth_model
    
    # build transition/measurement/clutter/birth models
    transition_model = ConstantVelocityModel(sigma=5)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0],
                                               sigma=10)
    clutter_model = PoissonClutterModel(detection_rate=0.9,
                                        lambda_clutter=20,
                                        scope=[[-1000, 1000], [-1000, 1000]])
else:
    scene = NonlinearScenario.caseC(birth_weight=0.03)
    birth_model = scene.birth_model
    
    # make transition/measurement/clutter/birth models
    transition_model = SimpleCTRVModel(sigma_vel=1,
                                       sigma_omega=np.pi / 180)
    measurement_model = RangeBearningMeasurementModel(sigma_range=5,
                                                      sigma_bearing=np.pi / 180,
                                                      origin=[300, 400])
    clutter_model = PoissonClutterModel(detection_rate=0.9,
                                        lambda_clutter=20,
                                        scope=[[200, 1200], [-np.pi, np.pi]])

# build data generator
data_generator = SimulatedGroundTruthDataGenerator(scene, transition_model, noisy=False)

# build sensor model
sensor_model = DummySensorModel(clutter_model, measurement_model, random_seed=seed)

# build data reader
data_reader = MeasurementReader(data_generator, sensor_model)

# build density model
density_model = KalmanDensityModel()

# gate method
gate = EllipsoidalGate(percentile=0.999)

# estimator
estimator = EAPEstimator()

# reductor
reductor = HypothesisReductor(weight_min=1e-3, merging_threshold=4, capping_num=100)

# %% build trackers & filtering
# some extra parameters
surviving_rate = 0.99
recycle_threshold = 0.1
prob_min = 1e-3
prob_estimate = 0.5

mbm_filter = MBMFilter(surviving_rate,
                       prob_min,
                       prob_estimate,
                       birth_model,
                       density_model,
                       transition_model,
                       measurement_model,
                       clutter_model,
                       gate,
                       estimator,
                       reductor)

mbm_estimates = mbm_filter.filtering(data_reader)
mbm_card_pred = [ele.shape[1] for ele in mbm_estimates]
mbm_estimates = np.hstack(mbm_estimates)

# %% visualise results
visualizer = Visualizer(data_generator, 'MBM')
visualizer.show_estimates(mbm_estimates)
visualizer.show_cardinality(mbm_card_pred)
visualizer.show_measurements(data_reader)
