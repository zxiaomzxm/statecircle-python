#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec 18 17:50:11 2019

@author: zhxm
"""

import numpy as np
import matplotlib.pyplot as plt

from scenarios.linear import LinearScenario
from scenarios.nonlinear import NonlinearScenario
from statecircle.estimator.base import EAPEstimator
from statecircle.models.birth.base import MultiObjectBirthModel
from statecircle.models.density.kalman import KalmanDensityModel
from statecircle.models.density.unscented import UnscentedDensityModel
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
from statecircle.trackers.mot.global_nearest_neighbour_tracker import GlobalNearestNeighbourTracker
from statecircle.trackers.mot.joint_probabilistic_data_association_tracker import JointProbabilisticDataAssociationTracker
from statecircle.trackers.mot.multi_hypothesis_tracker import TrackOrientedMultiHypothesisTracker
from statecircle.types.state import GaussianState, UnscentedState

seed = None
scenario = 'nonlinear'
state_func_handle = UnscentedState # Supported `GaussianState` and `UnscentedState`

if scenario == 'linear':
    # make transition/measurement/clutter/birth models
    transition_model = ConstantVelocityModel(sigma=5)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0],
                                               sigma=10)
    clutter_model = PoissonClutterModel(detection_rate=0.9,
                                        lambda_clutter=20,
                                        scope=[[-1000, 1000], [-1000, 1000]])

    scene = LinearScenario.caseB(state_func_handle=state_func_handle)
    birth_model = scene.birth_model

elif scenario == 'nonlinear':
    # make transition/measurement/clutter/birth models
    transition_model = SimpleCTRVModel(sigma_vel=1,
                                       sigma_omega=np.pi / 180)
    measurement_model = RangeBearningMeasurementModel(sigma_range=5,
                                                      sigma_bearing=np.pi / 180,
                                                      origin=[300, 400])
    clutter_model = PoissonClutterModel(detection_rate=0.9,
                                        lambda_clutter=20,
                                        scope=[[200, 1200], [-np.pi, np.pi]])

    scene = NonlinearScenario.caseB(state_func_handle=state_func_handle)
    birth_model = scene.birth_model
    
# make data generator
noisy = False
data_generator = SimulatedGroundTruthDataGenerator(scene, transition_model, noisy=noisy)

# make sensor model
sensor_model = DummySensorModel(clutter_model, measurement_model, random_seed=seed)

# make data reader
data_reader = MeasurementReader(data_generator, sensor_model)

# make density model
if state_func_handle is GaussianState:
    density_model = KalmanDensityModel()
elif state_func_handle is UnscentedState:
    density_model = UnscentedDensityModel(transition_model.state_dim, alpha=1.0, beta=2.0)
else:
    raise TypeError

# gate method
gate = EllipsoidalGate(percentile=0.999)

# estimator
estimator = EAPEstimator()

# reductor
reductor = HypothesisReductor(weight_min=1e-3, merging_threshold=2, capping_num=100)

# %% build trackers & filtering
# GNN tracker
gnn_tracker = GlobalNearestNeighbourTracker(birth_model,
                                            density_model,
                                            transition_model,
                                            measurement_model,
                                            clutter_model,
                                            gate,
                                            estimator)
gnn_estimates = gnn_tracker.filtering(data_reader)
gnn_estimates = np.stack(gnn_estimates, -1)

## JPDA tracker
jpda_tracker = JointProbabilisticDataAssociationTracker(birth_model,
                                                        density_model,
                                                        transition_model,
                                                        measurement_model,
                                                        clutter_model,
                                                        gate,
                                                        estimator,
                                                        reductor)
jpda_estimates = jpda_tracker.filtering(data_reader)
jpda_estimates = np.stack(jpda_estimates, -1)

# MH tracker
mh_tracker = TrackOrientedMultiHypothesisTracker(birth_model,
                                                 density_model,
                                                 transition_model,
                                                 measurement_model,
                                                 clutter_model,
                                                 gate,
                                                 estimator,
                                                 reductor)
mh_estimates = mh_tracker.filtering(data_reader)
mh_estimates = np.stack(mh_estimates, -1)
# %% visualise results
gt_series = data_generator.gt_series
gt_data = np.hstack([ele.states for ele in data_generator])
fig, ax = plt.subplots(1, 1, figsize=(6,6))
plot_gt = ax.plot(gt_data[0], gt_data[1], 'yo', alpha=0.2, markersize=10)
plot_gnn = ax.plot(gnn_estimates[0].T, gnn_estimates[1].T, 'r*-', alpha=0.5)
plot_jpda = ax.plot(jpda_estimates[0].T, jpda_estimates[1].T, 'g.-', alpha=0.5)
plot_mh = ax.plot(mh_estimates[0].T, mh_estimates[1].T, 'b+-', alpha=0.5)

ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend((plot_gt[0], plot_gnn[0], plot_jpda[0], plot_mh[0]), ['Ground Truth', 'GNN', 'JPDA', "MHT"])
plt.axis('equal')

# %% plot measurements
meas, obj_meas, clutter_meas = [], [], []
for meas_data, obj_meas_, clutter_meas_ in data_reader.truth_meas_generator():
    meas.append(meas_data.meas)
    obj_meas.append(obj_meas_)
    clutter_meas.append(clutter_meas_)
meas, obj_meas, clutter_meas = np.hstack(meas), np.hstack(obj_meas), np.hstack(clutter_meas)

plt.figure()
plt.plot(obj_meas[0], obj_meas[1], 'r.', alpha=0.5)

# plot clutter
plt.plot(clutter_meas[0], clutter_meas[1], 'k.', alpha=0.2)
plt.legend(['measurements', 'clutter'])
plt.show()
plt.close('all')
