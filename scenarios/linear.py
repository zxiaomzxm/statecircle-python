#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 24 15:20:19 2019

@author: zhaoxm
"""
import numpy as np

from .base import BaseScenario
from statecircle.models.birth.base import SingleObjectBirthModel, MultiObjectBirthModel, PoissonBirthModel, BernoulliBirthModel
from statecircle.types.state import GaussianState, GaussianSumState, BernoulliState, GaussianMixtureState, GaussianState


class LinearScenario(BaseScenario):
    
    @classmethod
    def caseA(cls, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [0]
        tdeath = [100]
        state_dim = 4
        initial_states = np.array([[0, 0, 10, 10]]).T
        
        # make birth model
        birth_cov = 10 * np.eye(state_dim)
        birth_model = SingleObjectBirthModel(initial_states, birth_cov, state_func_handle)
        
        return LinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)

    @classmethod
    def caseB(cls, state_func_handle=GaussianState):
        # create ground truth model
        birth_num = 5
        time_range = [0, 100]
        tbirth = [0] * birth_num
        tdeath = [100] * birth_num
        initial_states = np.array([[0, 0, 0, -10],
                                   [400, -600, -10, 5],
                                   [-800, -200, 20, -5],
                                   [0, 0, 7.5, -5],
                                   [-200, 800, -3, -15]]).T
        # make birth model
        state_dim = 4
        birth_cov = np.eye(state_dim)
        birth_model = MultiObjectBirthModel(initial_states, birth_cov, state_func_handle)
    
        return LinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)
    
    @classmethod
    def caseC(cls, birth_weight=0.03, birth_cov_scale=400, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [1,  0,   1,  20,  20,  20,  40,  40,  60,  60,  80,  80]
        tdeath = [70, 100, 70, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        initial_states = np.array([[0, 0, 0, -10],
                                   [400, -600, -10, 5],
                                   [-800, -200, 20, -5],
                                   [400, -600, -7, -4],
                                   [400, -600, -2.5, 10],
                                   [0, 0, 7.5, -5],
                                   [-800, -200, 12, 7],
                                   [-200, 800, 15, -10],
                                   [-800, -200, 3, 15],
                                   [-200, 800, -3, -15],
                                   [0, 0, -20, -15],
                                   [-200, 800, 15, -5]]).T
    
        # build birth model
        state_dim = 4
        birth_log_weights = np.log(birth_weight * np.ones(state_dim))
        birth_num = 4
        birth_states = [None] * birth_num
        
        birth_cov = birth_cov_scale * np.eye(state_dim)
        birth_states[0] = state_func_handle(np.array([0, 0, 0, 0]), birth_cov)
        birth_states[1] = state_func_handle(np.array([400, -600, 0, 0]), birth_cov)
        birth_states[2] = state_func_handle(np.array([-800, -200, 0, 0]), birth_cov)
        birth_states[3] = state_func_handle(np.array([-200, 800, 0, 0]), birth_cov)
        intensity = GaussianSumState(birth_log_weights, birth_states)
        
        birth_model = PoissonBirthModel(intensity)
        return LinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)

    @classmethod
    def caseD(cls, birth_prob=0.1, birth_cov_scale=400, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [10]
        tdeath = [80]
        state_dim = 4
        initial_states = np.array([[0, 0, 10, 10]]).T

        # make birth model
        birth_cov = birth_cov_scale * np.eye(state_dim)
        birth_gaussian = state_func_handle(initial_states, birth_cov)
        bern = BernoulliState(prob=birth_prob,
                              state=GaussianMixtureState(log_weights=np.array([0.]),
                                                         gaussian_states=[birth_gaussian]))
        birth_model = BernoulliBirthModel(bern)

        return LinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)
    
    @classmethod
    def caseE(cls, birth_weight=0.03, birth_cov_scale=400, state_func_handle=GaussianState):
        # build birth model
        time_range = [0, 100]
        tbirth = [10]
        tdeath = [80]
        initial_states = np.array([[0, 0, 0, 0, 0, 0]]).T
        
        state_dim = 6
        birth_num = 6
        birth_log_weights = np.log(birth_weight * np.ones(state_dim))
        birth_states = [None] * birth_num
        birth_cov = birth_cov_scale * np.eye(state_dim)
        birth_states[0] = state_func_handle(np.ones(state_dim), birth_cov)
        birth_states[1] = state_func_handle(np.ones(state_dim), birth_cov)
        birth_states[2] = state_func_handle(np.ones(state_dim), birth_cov)
        birth_states[3] = state_func_handle(np.ones(state_dim), birth_cov)
        birth_states[4] = state_func_handle(np.ones(state_dim), birth_cov)
        birth_states[5] = state_func_handle(np.ones(state_dim), birth_cov)
        intensity = GaussianSumState(birth_log_weights, birth_states)

        birth_model = PoissonBirthModel(intensity)
        return LinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)

    @classmethod
    def caseF(cls, birth_weight=0.03, birth_cov_scale=400, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [1]
        tdeath = [100]
        initial_states = np.array([[0, 0, 10, 0]]).T
    
        # build birth model
        state_dim = 4
        birth_num = 1
        birth_log_weights = np.log(birth_weight * np.ones(birth_num))
        birth_states = [None] * birth_num
        
        birth_cov = birth_cov_scale * np.eye(state_dim)
        birth_states[0] = state_func_handle(np.array([0, 0, 0, 0]), birth_cov)
        intensity = GaussianSumState(birth_log_weights, birth_states)
        
        birth_model = PoissonBirthModel(intensity)
        return LinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)
    