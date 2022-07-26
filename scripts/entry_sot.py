#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 15:17:17 2019

@author: zhxm
"""

import numpy as np
import matplotlib.pyplot as plt

from scenarios.linear import LinearScenario
from scenarios.nonlinear import NonlinearScenario
from statecircle.estimator.base import EAPEstimator
from statecircle.models.birth.base import SingleObjectBirthModel
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
from statecircle.trackers.sot.nearest_neighbour_tracker import NearestNeighbourTracker
from statecircle.trackers.sot.probabilistic_data_association_tracker import ProbabilisticDataAssociationTracker
from statecircle.trackers.sot.gaussian_sum_tracker import GaussianSumTracker
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
                                        scope=[[0, 1000], [0, 1000]])

    scene = LinearScenario.caseA(state_func_handle=state_func_handle)
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
                                        scope=[[0, 1000], [-np.pi, np.pi]])

    scene = NonlinearScenario.caseA(state_func_handle=state_func_handle)
    birth_model = scene.birth_model



# make data generator
birth_times = [0]
death_times = [100]
time_range = [0, 100]
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
# NN tracker
nn_tracker = NearestNeighbourTracker(birth_model,
                                     density_model,
                                     transition_model,
                                     measurement_model,
                                     clutter_model,
                                     gate,
                                     estimator)
nn_estimates = nn_tracker.filtering(data_reader)
nn_estimates = np.stack(nn_estimates, -1)

# PDA tracker
pda_tracker = ProbabilisticDataAssociationTracker(birth_model,
                                                  density_model,
                                                  transition_model,
                                                  measurement_model,
                                                  clutter_model,
                                                  gate,
                                                  estimator,
                                                  reductor)
pda_estimates = pda_tracker.filtering(data_reader)
pda_estimates = np.stack(pda_estimates, -1)

# GS tracker
gs_tracker = GaussianSumTracker(birth_model,
                                density_model,
                                transition_model,
                                measurement_model,
                                clutter_model,
                                gate,
                                estimator,
                                reductor)
gs_estimates = gs_tracker.filtering(data_reader)
gs_estimates = np.stack(gs_estimates, -1)

# %% visualise results
gt_series = data_generator.gt_series
gt_data = np.hstack([data[0].states for data in gt_series.datum])
fig, ax = plt.subplots(1, 1, figsize=(6,6))
plot_gt = ax.plot(gt_data[0], gt_data[1], 'y-', linewidth=10, alpha=0.5)
plot_nn = ax.plot(nn_estimates[0], nn_estimates[1], 'r*-', alpha=0.5)
plot_pda = ax.plot(pda_estimates[0], pda_estimates[1], 'g.-', alpha=0.5)
plot_gs = ax.plot(gs_estimates[0], gs_estimates[1], 'b+-', alpha=0.5)

ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend((plot_gt[0], plot_nn[0], plot_pda[0], plot_gs[0]), ['Ground Truth', 'Nearest Neighbour', 'Probabilistic Data Associaiton', "Gaussian Sum"])
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
