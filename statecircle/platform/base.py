#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: zhxm
"""
import numpy as np
import yaml

from ..base import Base
from statecircle.utils.common import import_model_with_paras, import_model, AttrDict


class Platform(Base):
    r"""Base platform class"""


class TrackerPlatform(Platform):
    r"""Single Object Tracker Platform"""
    cfg = None

    def __init__(self, config_path):
        with open(config_path) as f:
            TrackerPlatform.cfg = AttrDict.from_dict(yaml.load(f))

        print('building {}...\ntracker name: {}'.format(self.cfg.TRACKER.type, self.cfg.TRACKER.name))
        # build modules
        self.build_tracker_modules()

        # sanity check
        self.sanity_check()

        # build tracker
        self.tracker = import_model(self.cfg.TRACKER.type)(**self.__dict__)

    def build_tracker_modules(self):
        for k, v in self.cfg.TRACKER.items():
            if isinstance(v, AttrDict) and 'type' in v:
                setattr(self, k.lower(), import_model_with_paras(v))

    def sanity_check(self):
        mandatory_modules = ['birth_model', 'density_model', 'transition_model',
                             'measurement_model', 'clutter_model', 'gate', 'estimator']
        for module_name in mandatory_modules:
            assert module_name in self.__dict__.keys()

        assert self.birth_model.ndim == self.transition_model.ndim
