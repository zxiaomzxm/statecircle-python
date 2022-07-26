#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:56:16 2019

@author: zhaoxm
"""
from importlib import import_module
import numpy as np

from scipy.stats import multivariate_normal as mvn
from copy import deepcopy
from collections import OrderedDict


class IDontKnowHowToDoIt(Exception):
    pass



def rotate2D(x, theta, origin=[0, 0]):
    r"""rotate 2d points
    
    Parameters
    ----------
    x : array[2, #points]
    theta : float
    origin : point
    """
    if len(x) == 0:
        return x
    theta *= np.pi / 180
    origin = np.atleast_2d(origin).T
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat = np.array([[c, -s], [s, c]])
    return rot_mat.dot((x - origin)) + origin

def init(self, **kwargs):
    for k, v in kwargs.items():
        setattr(self, k, v)
        
def repr_(self):
    return format(self.__dict__)

def str_(self):
    return format(self.__dict__)

class AttrDict(OrderedDict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value

    @classmethod
    def from_dict(cls, dict_):
        for k, v in dict_.items():
            if isinstance(v, dict):
                dict_[k] = cls.from_dict(v)
        return AttrDict(dict_)
    
    def deepcopy(self):
        return deepcopy(self)
        
    def remove(self, name):
        bak = self.deepcopy()
        bak.pop(name)
        return bak

def list_index(input_list, index):
    return [input_list[i] for i in index]

def list_logical_index(input_list, logical_index):
    return [input_list[i] for i in range(len(logical_index)) if logical_index[i]]
            
def log_mvnpdf(x, mu, cov):
    return np.log(mvn(mu, cov).pdf(x))

def normalize_log_weights(log_w):
    if isinstance(log_w, list):
        log_w = np.array(log_w)
    if len(log_w) == 0:
        log_sum_w = np.empty([0])
        log_w = np.empty([0])
    elif len(log_w) == 1:
        log_sum_w = log_w[0]
        log_w = np.array([0.])
    else:
        if np.max(log_w) == np.inf:
            # corner case 1
            inf_idx = log_w == np.inf
            log_w[inf_idx] = 0 - np.log(np.sum(inf_idx))
            log_w[~inf_idx] = -np.inf
            log_sum_w = np.inf
        elif np.all(log_w == -np.inf):
            # corner case 2
            log_sum_w = float(len(log_w))
            log_w = np.zeros_like(log_w) - np.log(len(log_w))
        else:
            idx = np.argsort(-log_w)
            log_w_max = log_w[idx[0]]
            log_sum_w = log_w_max + np.log(1 + np.sum(np.exp(log_w[idx[1:]] - log_w_max)))
            log_w -= log_sum_w
        
    return log_w, log_sum_w

def has_duplicated_ele(x):
    assert isinstance(x, np.ndarray)
    assert x.ndim == 1
    return not len(np.unique(x)) == len(x)

def generate_trajectories(states, labels, keep_len=3):
    trajs = {}
    for state, label in zip(states, labels):
        for s, l in zip(state.T, label):
            if l in trajs.keys():
                trajs[l].append(s)
            else:
                trajs[l] = [s]
    
    trajs_prune = {}
    for k, v in trajs.items():
        if len(v) >= keep_len:
            trajs_prune[k] = np.stack(v, 1)
    return trajs_prune

def weights_outer_sum(weight, res):
    # auxiliary function in unscented density
    # return np.sum(weight * np.stack([np.outer(res[:,i], res[:,i]) \
    #                  for i in range(res.shape[1])], axis=-1), axis=-1)
    return res.dot(np.diag(weight).dot(res.T))
    
def is_pd(mat):
    # check the input matrix is posive definite
    return np.all(np.linalg.eigvals(mat) > 0)

def import_model(path):
    p, m = path.rsplit('.', 1)
    mod = import_module(p)
    return getattr(mod, m)

def import_model_with_paras(path):
    paras = path.remove('type')
    for k, v in paras.items():
        if isinstance(v, str) and v.startswith('<') and v.endswith('>'):
            paras[k] = eval(v.strip('<>'))
    return import_model(path.type)(**paras)

def tuple2int(x, y):
    return tuple([np.round(ele).astype(np.int) for ele in [x, y]])

def rem2pi(x):
    return x - 2 * np.pi * np.round(x / 2 / np.pi)
