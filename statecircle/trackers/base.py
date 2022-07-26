#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 15:02:46 2019

@author: zhxm
"""
from abc import abstractmethod

from ..base import Base


class Filter(Base):
    """ Filter class
    Filter can estimate current state using the history measurements up to current
    time, but can not output the whole trajectory
    
    Attributes
    ----------
    
    """
    def __init__(self, birth_model, density_model, transition_model,
                 measurement_model, clutter_model, gate, estimator, reductor=None):
        self.birth_model = birth_model
        self.density_model = density_model
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.clutter_model = clutter_model
        self.gate = gate
        self.estimator = estimator
        self.reductor = reductor
        self.timestamp = None

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    def birth(self, *args, **kwargs):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def estimate(self, *args, **kwargs):
        pass

    def reduction(self, *args, **kwargs):
        pass

    @abstractmethod
    def filtering(self, *args, **kwargs):
        pass


class Tracker(Filter):
    """ Tracker class
    Tracker can estimate current state and current state label at the same time
    using the history measurements up to currnet time
    
    Attributes
    ----------
    """

class SingleObjectTracker(Tracker):
    """ Single Object Tracker """


class MultiObjectFilter(Filter):
    """ Multi-object filter """


class MultiObjectTracker(Tracker):
    """ Multi-Object trackers """
