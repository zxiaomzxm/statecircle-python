#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from ..base import Model
from abc import abstractmethod

class MeasurementModel(Model):
    r"""Base measurement models class"""

    @property
    @abstractmethod
    def ndim_meas(self):
        r"""measurement dimentsion"""


