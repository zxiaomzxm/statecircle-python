#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Dec  5 11:49:17 2019

@author: zhxm
"""

from ..base import Base


class Type(Base):
    r"""Base Type"""

    def __repr__(self):
        attrs = ("{}:{!r}".format(name, value) for name, value in self.__dict__.items())
        return "{}({})".format(type(self).__name__, ", ".join(attrs))

    __str__ = __repr__


class Particles(Type):
    r""" Particle class
    Weighted particles used in particle filter
    
    Attributes
    ----------
    pts : Matrix(2, num)
        paticle potins
    weights : Vector[num]
        particle weights
    """

    def __init__(self, pts, weights):
        self.pts = pts
        self.weights = weights


class SigmaPoints(Particles):
    r""" Sigma Points
    Sigma points used in unscented Kalman filter

    NOTE: number of sigma points = 2 * state_dim + 1
    
    Attributes
    ----------
    pts : Matrix(2, num)
        Particle potins
    mean_weights : Vector(num)
        Particle first-order weights
    cov_weights : Vector(num)
        Particle second-order weights
    """

    def __init__(self, points, mean_weighs, cov_weights):
        self.points = points
        self.mean_weights = mean_weighs
        self.cov_weights = cov_weights

    @property
    def mean(self):
        return self.points.dot(self.mean_weights)

    @property
    def cov(self):
        res = (self.points - self.mean)
        return res.dot(self.cov_weights).dot(res.T)
