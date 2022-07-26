#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec 26 10:28:08 2019

@author: zhaoxm
"""
import numpy as np
import matplotlib.pyplot as plt

from scenarios.linear import LinearScenario
from scenarios.nonlinear import NonlinearScenario
from statecircle.estimator.base import EAPEstimator
from statecircle.lib.harem.debugger import plot_covariance_ellipse
from statecircle.models.birth.base import MultiObjectBirthModel, PoissonBirthModel
from statecircle.models.density.kalman import KalmanDensityModel
from statecircle.models.density.kalman_accumulated import KalmanAccumulatedDensityModel
from statecircle.models.density.unscented import UnscentedDensityModel
#from statecircle.models.density.unscented_accumulated import UnscentedAccumulatedDensityModel
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
from statecircle.trackers.mot.mbm_tracker import MBMTracker
from statecircle.types.state import GaussianState, GaussianSumState, UnscentedState
from tools.visualizer import Visualizer

seed = 666
scenario = 'linear'
state_func_handle = GaussianState  # supported `GaussianState` and `UnscentedState`

# build scene
if scenario == 'linear':
    scene = LinearScenario.caseC(birth_weight=0.005, birth_cov_scale=400,
                                 state_func_handle=state_func_handle)
    birth_model = scene.birth_model
    
    # build transition/measurement/clutter/birth models
    transition_model = ConstantVelocityModel(sigma=5)
    measurement_model = LinearMeasurementModel(mapping=[1, 1, 0, 0],
                                               sigma=10)
    clutter_model = PoissonClutterModel(detection_rate=0.9,
                                        lambda_clutter=20,
                                        scope=[[-1000, 1000], [-1000, 1000]])
else:
    scene = NonlinearScenario.caseC(birth_weight=0.001, state_func_handle=state_func_handle)
    birth_model = scene.birth_model
    
    # make transition/measurement/clutter/birth models
    transition_model = SimpleCTRVModel(sigma_vel=5,
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
if state_func_handle is GaussianState:
    density_model = KalmanAccumulatedDensityModel(traceback_range=3)
elif state_func_handle is UnscentedState:
    raise NotImplementedError
    # density_model = UnscentedAccumulatedDensityModel(transition_model.state_dim, alpha=1.0, beta=2.0)
else:
    raise TypeError

# gate method
gate = EllipsoidalGate(percentile=0.999)

# estimator
estimator = EAPEstimator()

# reductor
reductor = HypothesisReductor(weight_min=0.01, merging_threshold=4, capping_num=100)

# %% build trackers & filtering
# some extra parameters
prior_birth = False
surviving_rate = 0.99
recycle_threshold = 0.1
prob_min = 0.01
prob_estimate = 0.5

mbm_filter = MBMTracker(prior_birth,
                        surviving_rate,
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

#%% ploting
animation = False
show_birth = True

gt_datum = np.concatenate(data_generator.gt_series.datum, axis=-1)
true_state = np.concatenate([ele.states for ele in gt_datum], -1)
for k, (_, obj_meas_data, clutter_data) in enumerate(data_reader.truth_meas_generator()):
    if not animation:
        k = scene.time_range[-1] - 1
    MBM_estimated_state = np.hstack(mbm_estimates[k]).squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_gt = ax.plot(true_state[0], true_state[1], 'yo', alpha=0.2, markersize=10)
    
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
#    # plot birth region
#    if prior_birth and show_birth:
#        for birth_state in birth_model:
#            plot_birth = plot_covariance_ellipse(birth_state.x[:2], birth_state.P[:2,:2], 'b', ax, 3)
#    else:
#        plot_birth = None
        
    # plot tracks
    for track in mbm_estimates[k]:
        range_t = track['range'][-1] - track['range'][0] + 1
        # if range_t > -1:
        plt.plot(track['trajectory'][0], track['trajectory'][1], '-')
    
    # plot obejct measurements
    state_meas = measurement_model.reverse(obj_meas_data)
    plot_meas = ax.plot(state_meas[0], state_meas[1], 'r*', alpha=1)
    # plot clutter
    state_clutter = measurement_model.reverse(clutter_data)
    plot_clutter = ax.plot(state_clutter[0], state_clutter[1], 'k.', alpha=1)

    ax.legend((plot_gt[0], plot_meas[0], plot_clutter[0]),
              ['ground truth', 'detections', 'clutter'],
              loc='upper left')
        
    ax.set_xlim([-1000, 1000])
    ax.set_ylim([-1000, 1000])
#    plt.axis('equal')
#    plt.savefig('snapshot/results/track_{:04d}.png'.format(k))
    
    plt.show()
    print('step: {}'.format(k))
    plt.close('all')
    
    if not animation:
        break

#%% plot cardinality
plt.figure()
plt.plot(data_generator.gt_series.num, 'yo')
#PMBM_card_pred = [for ele in track for step, track in enumerate(PMBMEstimates)]
PMBM_card_pred = []
for step, tracks in enumerate(mbm_estimates):
    valid_track_num = 0
    for track in tracks:
        range_t = track['range'][-1] - track['range'][0] + 1
        if track['range'][-1] == step:
            valid_track_num += 1
    PMBM_card_pred.append(valid_track_num)
plt.plot(PMBM_card_pred, 'b+')
plt.legend(['GT', 'PMBM'])
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


    
