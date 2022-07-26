#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from abc import abstractmethod

from ..base import Model


class ConjugateDensityModel(Model):
    r"""Conjugate Density model

    The density is multi-object conjugate under predict and update steps
    """

    @abstractmethod
    def predict(self, state, time_step, transition_model):
        r"""

        Parameters
        ----------
        state
        transition_model

        Returns
        -------

        """

    @abstractmethod
    def update(self, state, measurement_model):
        r"""

        Parameters
        ----------
        state
        measurement_model

        Returns
        -------

        """

    @abstractmethod
    def predicted_log_likelihood(self, state, meas, measurement_model):
        r"""

        Parameters
        ----------
        state
        meas
        measurement_model

        Returns
        -------

        """



