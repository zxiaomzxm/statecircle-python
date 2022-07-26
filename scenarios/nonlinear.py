#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 24 15:54:01 2019

@author: zhaoxm
"""
import numpy as np

from .base import BaseScenario
from statecircle.models.birth.base import SingleObjectBirthModel, MultiObjectBirthModel, PoissonBirthModel, \
    BernoulliBirthModel
from statecircle.types.state import GaussianState, GaussianSumState, BernoulliState, GaussianMixtureState


class NonlinearScenario(BaseScenario):

    @classmethod
    def caseA(cls, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [0]
        tdeath = [100]
        initial_states = np.array([[0, 0, 10, 0, np.pi / 180]]).T

        # make birth model
        birth_cov = np.diag(np.array([1, 1, 1, 1 * np.pi / 180, 1 * np.pi / 180]) ** 2)
        birth_model = SingleObjectBirthModel(initial_states, birth_cov, state_func_handle)

        return NonlinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)

    @classmethod
    def caseB(cls, state_func_handle):
        # create ground truth model
        time_range = [0, 100]
        birth_num = 4
        tbirth = [0] * birth_num
        tdeath = [100] * birth_num
        initial_states = np.array([[0, 0, 5, 0, np.pi / 180],
                                   [20, 20, -20, 0, np.pi / 90],
                                   [-20, 10, -10, 0, np.pi / 360],
                                   [-10, -10, 8, 0, np.pi / 270]]).T

        # make birth model
        birth_cov = np.diag(np.array([1, 1, 1, 1 * np.pi / 180, 1 * np.pi / 180]) ** 2)
        birth_model = MultiObjectBirthModel(initial_states, birth_cov, state_func_handle)

        return NonlinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)

    @classmethod
    def caseC(cls, birth_weight=0.3, birth_cov_scale=10, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [0, 20, 40, 60]
        tdeath = [50, 70, 90, 100]
        initial_states = np.array([[0, 0, 5, 0, np.pi / 180],
                                   [20, 20, -10, 0, np.pi / 90],
                                   [-20, 10, -10, 0, np.pi / 360],
                                   [-10, -10, 8, 0, np.pi / 270]]).T

        # build birth model
        birth_num = 4
        birth_log_weights = np.log(birth_weight * np.ones(birth_num))

        birth_cov = birth_cov_scale * np.diag(np.array([1, 1, 1, 1 * np.pi / 90, 1 * np.pi / 90]) ** 2)
        birth_states = [None] * birth_num
        birth_states[0] = state_func_handle(np.array([0, 0, 5, 0, np.pi / 180]), birth_cov)
        birth_states[1] = state_func_handle(np.array([20, 20, -10, 0, np.pi / 90]), birth_cov)
        birth_states[2] = state_func_handle(np.array([-20, 10, -10, 0, np.pi / 360]), birth_cov)
        birth_states[3] = state_func_handle(np.array([-10, -10, 8, 0, np.pi / 270]), birth_cov)
        intensity = GaussianSumState(birth_log_weights, birth_states)
        birth_model = PoissonBirthModel(intensity)

        return NonlinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)

    @classmethod
    def caseD(cls, birth_prob=0.1, state_func_handle=GaussianState):
        time_range = [0, 100]
        tbirth = [10]
        tdeath = [80]
        initial_states = np.array([[0, 0, 10, 0, np.pi / 180]]).T

        # make birth model
        birth_cov = np.diag(np.array([1, 1, 1, 1 * np.pi / 180, 1 * np.pi / 180]) ** 2)
        birth_gaussian = state_func_handle(initial_states, birth_cov)
        birth_bern = BernoulliState(prob=birth_prob,
                                    state=GaussianMixtureState(log_weights=np.array([0.]),
                                                               gaussian_states=[birth_gaussian]))

        birth_model = BernoulliBirthModel(birth_bern)

        return NonlinearScenario(time_range, tbirth, tdeath, initial_states, birth_model)
