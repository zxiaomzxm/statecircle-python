#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Dec  4 20:34:02 2019

@author: zhxm
"""

from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class BaseMeta(ABCMeta):
    r"""BaseMeta meta class
    """

    # def __prepare__(mcls, name, bases, **kwargs):
    #     return OrderedDict()

    def __new__(mcls, name, bases, namespace, **kwargs):
        return super().__new__(mcls, name, bases, namespace, **kwargs)


class Base(metaclass=BaseMeta):
    r"""StateCircle base meta class
    """
