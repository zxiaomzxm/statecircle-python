#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Dec 20 14:14:33 2019

@author: zhaoxm
"""
import numpy as np
import matplotlib.pyplot as plt

from scenarios.linear import LinearScenario
from scenarios.nonlinear import NonlinearScenario
from statecircle.estimator.base import EAPEstimator
from statecircle.models.birth.base import MultiObjectBirthModel, PoissonBirthModel
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
from statecircle.types.state import GaussianState, GaussianSumState, UnscentedState
from statecircle.trackers.mot.phd_filter import PHDFilter
from statecircle.trackers.mot.mbm_filter import MBMFilter
from statecircle.trackers.mot.pmbm_filter import PMBMFilter

seed = None
scenario = 'linear'
state_func_handle = UnscentedState  # supported `GaussianState` and `UnscentedState`

if scenario == 'linear':
    # make transition/measurement/clutter/birth models
    transition_model = ConstantVelocityModel(sigma=5)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0],
                                               sigma=10)
    clutter_model = PoissonClutterModel(detection_rate=0.9,
                                        lambda_clutter=20,
                                        scope=[[-1000, 1000], [-1000, 1000]])

    scene = LinearScenario.caseC(birth_weight=0.03, birth_cov_scale=400, state_func_handle=state_func_handle)
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

    scene = NonlinearScenario.caseC(birth_weight=0.03, state_func_handle=state_func_handle)
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
reductor = HypothesisReductor(weight_min=1e-3, merging_threshold=4, capping_num=100)

# %% build trackers & filtering
# PHD filter
surviving_rate = 0.99
phd_filter = PHDFilter(surviving_rate,
                       birth_model,
                       density_model,
                       transition_model,
                       measurement_model,
                       clutter_model,
                       gate,
                       estimator,
                       reductor)
phd_estimates_list = phd_filter.filtering(data_reader)
phd_estimates = np.hstack(phd_estimates_list)
phd_card_pred = [ele.shape[1] for ele in phd_estimates_list]

surviving_rate = 0.99
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
mbm_estimates_list = mbm_filter.filtering(data_reader)
mbm_estimates = np.hstack(mbm_estimates_list)
mbm_card_pred = [ele.shape[1] for ele in mbm_estimates_list]

#surviving_rate = 0.99
recycle_threshold = 0.1
prob_min = 1e-3
prob_estimate = 0.5
pmbm_filter = PMBMFilter(surviving_rate,
                         recycle_threshold,
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
pmbm_estimates_list = pmbm_filter.filtering(data_reader)
pmbm_estimates = np.hstack(pmbm_estimates_list)
pmbm_card_pred = [ele.shape[1] for ele in pmbm_estimates_list]

# %% visualise results
gt_series = data_generator.gt_series
gt_data = np.hstack([ele.states for ele in data_generator])
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plot_gt = ax.plot(gt_data[0], gt_data[1], 'yo', alpha=0.2, markersize=10)
plot_phd = ax.plot(phd_estimates[0].T, phd_estimates[1].T, 'g+', alpha=0.5)
plot_mbm = ax.plot(mbm_estimates[0].T, mbm_estimates[1].T, 'b+', alpha=0.5)
plot_pmbm = ax.plot(pmbm_estimates[0].T, pmbm_estimates[1].T, 'r+', alpha=0.5)

ax.grid()
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.legend((plot_gt[0], plot_phd[0], plot_mbm[0], plot_pmbm[0]), ['Ground Truth', 'PHD', 'MBM', 'PMBM'])
plt.axis('equal')

# %% plot cardinality
plt.figure()
plt.plot(gt_series.num, 'yo')
plt.plot(phd_card_pred, 'g+')
plt.plot(mbm_card_pred, 'b+')
plt.plot(pmbm_card_pred, 'r+')
plt.legend(['GT', 'PHD', 'MBM', 'PMBM'])
plt.grid()

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
