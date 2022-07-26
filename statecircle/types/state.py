#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 11:48:08 2019

@author: zhxm
"""

import numpy as np
from pytest import approx
from numpy import ndarray
from scipy.special import logsumexp

from statecircle.utils.common import normalize_log_weights
from .base import Type, Particles, SigmaPoints


class State(Type):
    r""" State class
    Generic state definition
    """
    @property
    def ndim(self):
        return self.mean.shape[0]


class LabeledState(State):
    r"""Labeled state for trackers based on labeled RFS state"""
    def __init__(self, label):
        self.label = np.array(label, dtype=np.int)
        
    def append(self, label):
        self.label = np.append(self.label, label)
        
    @property
    def num_label(self):
        return len(self.label)


"""
=========================
Ordinay states defination
=========================
"""


class GaussianState(State):
    r""" Gaussian state defination
    
    Attributes
    ----------
    mean : vector(ndim, 1)
        Gaussian mean vector
    cov : Matrix(ndim, ndim)
        Gaussian covariance matrix
    """

    def __init__(self, mean=None, cov=None):
        if isinstance(mean, ndarray):
            mean = mean[:, None] if mean.ndim == 1 else mean
        self.mean = mean
        self.cov = cov

    def __repr__(self):
        return "{}(mean: {}, cov: {})".format(type(self).__name__, self.mean, self.cov)
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if np.all(self.mean == other.mean) and np.all(self.cov == other.cov):
                return True
            else:
                return False
        else:
            return False

    __str__ = __repr__

class GaussianAccumulatedState(GaussianState):
    r"""Gaussian accumluated state defination

    Attributes
    ----------
    means : vector(ndim, n)
        Gaussian means with n accumulated density
    cov_aug : Matrix(ndim * n, ndim * n)
        Gaussian accumulated covariance matrix
    """
    def __len__(self):
        return self.mean.shape[1]


class GaussianLabeledState(GaussianState, LabeledState):
    r"""Gaussian labeled state"""

    def __init__(self, label, *args, **kwargs):
        GaussianState.__init__(self, *args, **kwargs)
        LabeledState.__init__(self, label)


class GaussianSumState(State):
    r""" Gaussian sum state, a intensity state
    
    Attributes
    ----------
    weights : `Vector` (num_components)
        Gaussian components weights, not normalized to 1
    gaussian_states : List of `GaussianState` (num_components)
        A list contains Gaussian states
    """

    def __init__(self, log_weights=[], gaussian_states=[]):
        assert len(log_weights) == len(gaussian_states)
        self.log_weights = log_weights
        self.gaussian_states = gaussian_states

    @property
    def mean(self):
        means = np.hstack([ele.mean for ele in self.gaussian_states])
        return (self.weights * means).sum(-1, keepdims=True)

    @property
    def cov(self):
        means = np.stack([ele.mean for ele in self.gaussian_states])
        covs = np.stack([ele.cov for ele in self.gaussian_states], axis=-1)
        cov_sum = (self.weights * covs).sum(-1)
        delta_mean = self.mean - means
        cov_mean_spread = delta_mean.dot(np.diag(self.weights).dot(delta_mean.T))
        return cov_sum + cov_mean_spread

    @property
    def weights(self):
        return np.exp(self.log_weights)
    
    @property
    def num_comps(self):
        # number of components
        return len(self.log_weights)

    def __add__(self, other):
        log_weights = np.hstack((self.log_weights, other.log_weights))
        states = self.gaussian_states + other.gaussian_states
        return GaussianSumState(log_weights, states)

    def __radd__(self, other):
        log_weights = np.hstack((other.log_weights, self.log_weights))
        states = other.gaussian_states + self.gaussian_states
        return GaussianSumState(log_weights, states)

    def __iadd__(self, other):
        return self.__add__(other)


    def integral(self):
        r"""
        return Gaussian sum integral

        Returns
        -------
        State integral
        """
        return np.exp(logsumexp(self.log_weights))


class GaussianMixtureState(GaussianSumState):
    r""" Gaussian mixture state, a density state
    
    Attributes
    ----------
    weights : `Vector` (num_components)
        Gaussian components weights, summation nomalized to 1
    gaussian_states : List of `GaussianState` (num_components)
        A list contains Gausian states
    """
    # TODO: testcase

    def __init__(self, log_weights=[], gaussian_states=[]):
        super().__init__(log_weights, gaussian_states)
        # TODO: log filed computing
        self.log_weights, _ = normalize_log_weights(log_weights)

    def __add__(self, other):
        log_weights = np.hstack((self.log_weights, other.log_weights))
        states = self.gaussian_states + other.gaussian_states
        return GaussianMixtureState(log_weights, states)

    def __radd__(self, other):
        log_weights = np.hstack((other.log_weights, self.log_weights))
        states = other.gaussian_states + self.gaussian_states
        return GaussianMixtureState(log_weights, states)

    def __iadd__(self, other):
        return self.__add__(other)


class ParticleState(State):
    r""" Particle state defination
    
    Attributes
    ----------
    mean : Vector(ndim, 1)
        Particles statistical mean
    cov : Matrix(ndim.ndim)
        Particles statistical covariance matrix
    particles : `Particles` instance
        Contains attributes `pts` and `weights`
    """

    def __init__(self, num_particles=1000):
        self.num_particles = num_particles
        self.particles = Particles(self.num_particles)
        self._update_particles_state(self.particles)

    def update_particles_state(self, particles):
        self._mean = particles.mean(1)
        self._cov = particles.cov(ddof=0)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov


class UnscentedState(GaussianState):
    r"""Unscented State class
    
    Attributes
    ----------
    mean : Vector(ndim, 1)
        Particles statistical mean
    cov : Matrix(ndim.ndim)
        Particles statistical covariance matrix
    sigma : `SigmaPoints` instance
        Contains attributes `points`, `mean_weights` and `cov_weights`
    inno_cov: Matrix(ndim_meas, ndim_meas)
        Innovation convariance matrix
    """

    def __init__(self, mean, cov, sigma_pts=None):
        if isinstance(mean, ndarray):
            mean = mean[:, None] if mean.ndim == 1 else mean
        self.mean = mean
        self.cov = cov
        self.sigma = sigma_pts
        self.inno_cov = None

    @property
    def pts_mean(self):
        return self.sigma.mean()

    @property
    def pts_cov(self):
        return self.sigma.cov()


"""
==========================================
Random finite sets based states definition
==========================================
"""


class Probability:
    r""" Probability data descriptor
    
        assert isinstance(prob, float)
        assert 0. <= prob <= 1.
    """

    def __get__(self):
        pass

    def __set__(self):
        pass

    def __delete__(self):
        pass

    def __set_name__(self):
        pass


class BernoulliState(State):
    r""" Bernoulli state class
    
    Attributes
    ----------
    prob : float in range [0, 1]
        Existance probabilty
    state : `State`
        State instance, can be arbitray valid density representaion
    r"""

    #    prob = Probabilty()
    def __init__(self, prob=None, state=None):
        if prob is not None:
            assert isinstance(prob, float) and 0. <= prob <= 1., "Invalid probability input"
        self.prob = prob

        if state is not None:
            assert isinstance(state, State), "Bernoulli state must be valid state representation"
        self.state = state
        
    @property
    def ndim(self):
        return self.state.ndim


class BernoulliTrajectory(BernoulliState):
    r""" Bernoulli trajectory class

    Attributes
    ----------
    prob : float in range [0, 1]
        Existance probabilty
    state : `State`
        State instance, can be arbitray valid density representaion
    t_birth : timestamp
        birth time
    t_death : vector[timestamp]
        death times
    w_death: vector[timestamp]
        death weights
    r"""

    #    prob = Probabilty()
    def __init__(self, prob=None, state=None, t_birth=None, t_death=None, w_death=None):
        super(BernoulliTrajectory, self).__init__(prob, state)
        self.t_birth = t_birth
        self.t_death = t_death
        self.w_death = w_death


class MultiBernoulliState(State):
    r""" Multi-Bernoulli state class
    Multi-Bernoulli random finite sets
    
    Attributes
    ----------
    berns : List of `BernoulliState` (num_berns)
        A list contains Bernnoulli states
    """

    def __init__(self, num_berns):
        self.berns = [BernoulliState() for i in range(num_berns)]


class MultiBernoulliMixtureState(State):
    r""" Multi-Bernoulli mixture state class
    Multi-Bernoulli mixture random finite sets
    
    Attributes
    ----------
    weights : `Vector` (num_components)
        Bernoulli components weights
    multi_berns : List of `MultiBernoulliState` (num_components)
        A list contains Bernnoulli states
    """

    def __init__(self, num_berns, num_components):
        assert num_components > 0, "num_components must be positive"
        self.weights = np.ones(num_components) / num_components
        self.multi_berns = [MultiBernoulliState(num_bern) for num_bern in num_berns]


class PoissonState(State):
    r""" Poisson state class
    Poisson random finite sets/Poisson point process representation
    
    Attributes
    ----------
    intensity : `State`
        Uniform intensity or Gaussian sum intensity, note NOT density
    """

    def __init__(self, intensity):
        self.intensity = intensity

    @property
    def mean(self):
        return self.intensity.integral()


