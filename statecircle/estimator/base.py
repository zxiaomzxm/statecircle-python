#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""

from abc import abstractmethod

from ..base import Base

class Estimator(Base):
    r"""estimator base class"""

    @abstractmethod
    def __call__(self, state):
        r"""Extract estimate from density"""


class EAPEstimator(Estimator):
    r"""Expected a posterior estimator"""

    def __call__(self, state):
        return state.mean


class MAPEstimator(Estimator):
    r"""Maximum a posterior estimator"""

    def __call__(self, state):
        return state.max()
