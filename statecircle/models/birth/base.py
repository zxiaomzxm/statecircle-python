#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from abc import abstractmethod
import numpy as np

from ..base import Model
from statecircle.types.state import GaussianState, PoissonState


class BirthModel(Model):
    r"""Birth model"""
    @abstractmethod
    def birth(self):
        r"""virtual birth method"""


class SingleObjectBirthModel(BirthModel):
    def __init__(self, initial_state, birth_cov, state_func_handle=GaussianState):
        r"""

        Parameters
        ----------
        initial_state : state
        birth_cov : array
        """
        initial_state = np.array(initial_state)
        initial_state = initial_state[:, None] if initial_state.ndim == 1 else initial_state
        self.mean = self.initial_state = initial_state
        self.cov = birth_cov
        self.state_func_handle = state_func_handle

    def birth(self):
        return self.state_func_handle(self.mean, self.cov)

    @property
    def ndim(self):
        return self.mean.shape[0]


class MultiObjectBirthModel(BirthModel):
    def __init__(self, initial_states, birth_cov, state_func_handle=GaussianState):
        r"""

        Parameters
        ----------
        initial_states : array(ndim_state, num_state)
        birth_cov: array
        """
        self.means = initial_states
        self.covs = [birth_cov for _ in range(self.means.shape[1])]
        self.state_func_handle = state_func_handle

    def birth(self):
        return [self.state_func_handle(mean, cov) for mean, cov in zip(self.means.T, self.covs)]

class BernoulliBirthModel(BirthModel):
    def __init__(self, bern):
        self.bern = bern
        
    def birth(self):
        return self.bern
        
class MultiBernoulliBirthModel(BirthModel):
    def __init__(self, multi_bern):
        self.multi_bern = multi_bern
    
    def birth(self):
        return self.multi_bern
    

class PoissonBirthModel(BirthModel):
    def __init__(self, intensity):
        self.intensity = intensity

    def birth(self):
        return PoissonState(self.intensity)
