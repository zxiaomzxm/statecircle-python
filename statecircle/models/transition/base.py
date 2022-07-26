#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
from ..base import Model


class TransitionModel(Model):
    r"""Base transition models class"""
    def __init__(self, *args, **kwargs):
        pass

    def jacobian(self, x):
        r"""Virtual Jacobian matrix for transition forward function"""
        raise NotImplementedError

    def forward(self, x, noise=None):
        r"""Virtual transition forward method for transition models

        .. math::

            x_{t+1} = f_t(x_t) + (u_t), \ \ \ \  u_t \sim \mathcal{N}(0, Q_t)
        """
        raise NotImplementedError

    def reverse(self, z):
        r"""Virtial transition reverse method for transition models

        .. math::

            x_t = f_t^{-1}(x_{t+1})
        """
        raise NotImplementedError
