#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep 11 09:27:40 2020

@author: zhxm
"""
import numpy as np
from copy import deepcopy

from statecircle.types.state import BernoulliTrajectory, GaussianAccumulatedState
from statecircle.models.density.kalman_accumulated import KalmanAccumulatedDensityModel
from statecircle.models.transition.linear import ConstantVelocityModel
from statecircle.models.measurement.clutter import PoissonClutterModel
from statecircle.models.measurement.linear import LinearMeasurementModel
from statecircle.utils.common import normalize_log_weights
#%%
# hyper parameters
state_dim = 4
meas_dim = 2

prob_min = 0.01
surviving_rate = 0.95
traceback_range = 5
trans_sigma = 10
meas_mapping = [1, 1, 0, 0]
meas_sigma = 10
time_step = 0.1

# clutter hyper parameters
detection_rate = 0.9
lambda_clutter = 20
scope = [[-1000, 1000], [-1000, 1000]]

# bith hyper parameters
birth_weight = 0.001
birth_var = 400.0

#%% initialize models
density_model = KalmanAccumulatedDensityModel(traceback_range = traceback_range)
transition_model = ConstantVelocityModel(trans_sigma)
measurement_model = LinearMeasurementModel(meas_mapping, meas_sigma)
clutter_model = PoissonClutterModel(detection_rate, lambda_clutter, scope)

#%% test methods
def Bern_predict(Bern, time_step):
    """performs prediiction step for a Bernoulli trajectory component
    Parameters
    ----------
    Bern : BernoulliTrajectory

    Notes: inplace prediction
    """
    if Bern.w_death[-1] >= prob_min:
        Bern.state = density_model.predict(Bern.state, time_step, transition_model)
        # predicted time of death
        Bern.t_death = np.append(Bern.t_death, Bern.t_death[-1] + 1)
        Bern.w_death = np.append(Bern.w_death[:-1],
                                 (Bern.w_death[-1] * (1 - surviving_rate),
                                  Bern.w_death[-1] * surviving_rate))
    return Bern

def PPP_detected_update(step, log_w, state, mea):
    r"""creates a new local hypothesis by updating the PPP with measurement and
       calculates the corresponding likelihood

    Parameters
    ----------
    step : scalar
        iter time step
    indices : bool matrix
        represent which ppp component contains the measurement
    mea : ndarray[meas_dim, 1]

    Returns
    -------
    Bern_upd, likelihood
    """
    detection_rate = clutter_model.detection_rate
    intensity_clutter = clutter_model.intensity_clutter

    # 1) for each mixture component in the PPP intensity, perform Kalman update and
    # calculate the predicted likelihood for each detection inside the corresponding
    # gate
    log_w = [np.log(detection_rate) + log_w + \
             density_model.predicted_log_likelihood(state, mea, measurement_model)]
    
    log_w, log_sum_w = normalize_log_weights(log_w)
    rho = np.exp(log_sum_w)

    # 2) perform Gaussian moment matching for the updated object state
    # densities resulted from being updated by the same detection
    Bern = BernoulliTrajectory()
    Bern.state = state
    
    # 3) the returned likelihood should be the sum of the predicted
    # likelihoods calculated for each mixture component in the PPP
    # intensity and the clutter intensity
    lik_new = np.log(intensity_clutter + rho)

    # 4) the return existence probability of the Bernoulli component
    # is the ratio between the sum of the predicted likelihoods and
    # the returned likelihood
    Bern.prob = rho / (intensity_clutter + rho)
    
    Bern.t_birth = step
    Bern.t_death = np.array([step])
    Bern.w_death = np.array([1.])

    return Bern, lik_new

def Bern_detected_update_lik(Bern, meas):
    """calculates the predicted likelihood for a given local hypothesis

    Parameters
    ----------
    Bern : BernoulliTrajectory
    meas : ndarray[meas_dim, num_meas]

    Returns
    -------
    detected likelihoods
    """
    lik_detected = density_model.predicted_log_likelihood(Bern.state, meas, measurement_model) + \
                   np.log(clutter_model.detection_rate) + np.log(Bern.prob) + np.log(Bern.w_death[-1])
    return lik_detected

def Bern_detected_update_state(Bern, meas):
    """creates the new local hypothesis due to measurement update

    Parameters
    ----------
    Bern : BernoulliTrajectory
    meas : ndarray[meas_dim, num_meas]

    Returns
    -------
    Bern_det_upd
    """

    # creates the new local hypothesis due to measurement update
    # udpated time of death
    Bern_det_upd = BernoulliTrajectory(prob=1.,
                                       state=density_model.update(Bern.state, meas, measurement_model),
                                       t_birth=Bern.t_birth,
                                       t_death=Bern.t_death[-1:],
                                       w_death=np.array([1.]))

    return Bern_det_upd

def Bern_undetected_update(Bern):
    """caculates the likelihood of missed detection, and creates new local 
    hypotheses due to missed detection.
    Parameters
    ----------
    Bern : BernoulliTrajectory

    Returns
    -------
    BernTrajectory, likelihood
    """
    detection_rate = clutter_model.detection_rate
    Bern_undet_upd = deepcopy(Bern)
    l_nodetect = Bern.prob * (1 - detection_rate * Bern.w_death[-1])
    lik_undetected = 1 - Bern.prob + l_nodetect
    Bern_undet_upd.prob = np.exp(np.log(l_nodetect) - np.log(lik_undetected))
    lik_undetected = np.log(lik_undetected)

    # updated time of death
    Bern_undet_upd.w_death = np.append(Bern.w_death[:-1], Bern.w_death[-1] * (1 - detection_rate)) / \
                             (1 - detection_rate * Bern.w_death[-1])

    return Bern_undet_upd, lik_undetected
#%% testcases
"""
Bern
- state
  - mean
  - cov
- t_birth
- t_death
- w_death

"""
def generate_Bern(state_dim):
    mean = np.ones((state_dim, 1))
    cov = np.eye(state_dim)
    return BernoulliTrajectory(prob = 0.5, 
                               state = GaussianAccumulatedState(mean, cov),
                               t_birth = 0,
                               t_death = [0.0],
                               w_death = [1.0])
    
def generate_mea(meas_dim):
    return (np.arange(meas_dim) + 1)[:, None]

# test Bern_predict
Bern = generate_Bern(state_dim)
print("initial Bern:")
print(Bern)
Bern = Bern_predict(Bern, time_step=time_step)
print("predicted Bern:")
print(Bern)
Bern = Bern_predict(Bern, time_step=time_step)
print("predicted Bern twice:")
print(Bern)

# test birth and PPP_detected_update
mea = generate_mea(meas_dim)
cov = birth_var * np.eye(state_dim)
mean = measurement_model.reverse(mea)
birth_state = GaussianAccumulatedState(mean, cov)
birth_Bern, birth_lik = PPP_detected_update(step=0, 
                                            log_w=np.log(birth_weight),
                                            state=birth_state,
                                            mea=mea)
print("birth_Bern:\n", birth_Bern)
print("birth_lik: ", birth_lik)

# test Bern_detected_update
Bern = generate_Bern(state_dim)
mea = generate_mea(meas_dim)
lik_detected = Bern_detected_update_lik(Bern, mea)
Bern_detected = Bern_detected_update_state(Bern, mea)
print("Bern_detected:\n", Bern_detected)
print("lik_detected: ", lik_detected)

# test Bern_undetected_update
Bern = generate_Bern(state_dim)
Bern_undet_upd, lik_undetected = Bern_undetected_update(Bern)
print("Bern_undet_upd:\n", Bern_undet_upd)
print("lik_undetected: ", lik_undetected)
