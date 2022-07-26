#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 24 15:59:55 2019

@author: zhaoxm
"""
class BaseScenario:
    def __init__(self, time_range, birth_times, death_times, initial_states, birth_model):
        r""" Nonlinear scenerio class
        
        Attributes
        ----------
        birth_times : list[#states]
        death_times : list[#states]
        time_range : list[#states]
        initial_states : 2darray[state_dim, #states]
        """
        assert len(birth_times) == len(death_times) == initial_states.shape[1]
        assert len(time_range) == 2 and time_range[1] >= time_range[0]
        assert time_range[0] <= min(birth_times)
        assert time_range[1] >= max(death_times)
        
        self.birth_times = birth_times
        self.death_times = death_times
        self.time_range = time_range
        self.initial_states = initial_states
        self.state_dim, self.birth_num = initial_states.shape
        self.birth_model = birth_model