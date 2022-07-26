#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""

from ..base import Base
from abc import abstractmethod


class Model(Base):
    r"""Virtual base models model"""

    @property
    def ndim(self):
        r"""return state dimension"""
        raise NotImplementedError

    def forward(self):
        r"""virtual forward function"""
        raise NotImplementedError

    def reverse(self ):
        r"""reverse function, not virtual method"""
        raise NotImplementedError



