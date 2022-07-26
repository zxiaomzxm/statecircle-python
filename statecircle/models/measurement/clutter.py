#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
from scipy.stats import poisson

from statecircle.models.base import Model


class ClutterModel(Model):
    r"""Clutter model base class"""


class PoissonClutterModel(ClutterModel):
    r"""Poisson sensor model"""

    def __init__(self, detection_rate, lambda_clutter, scope):
        self.detection_rate = detection_rate
        self.lambda_clutter = lambda_clutter
        self.scope = np.array(scope)

        volumn = np.prod(self.scope[:, 1] - self.scope[:, 0])
        self.intensity_clutter = lambda_clutter / volumn
        self.density = 1 / volumn
        self.cardinality_pmf = poisson(lambda_clutter).pmf
        