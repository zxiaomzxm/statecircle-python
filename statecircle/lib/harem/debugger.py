#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:40:00 2019

@author: zhxm
"""

import numpy as np
import inspect
from pprint import pformat
from time import time
from functools import wraps
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import cv2
from scipy.stats import chi2, multivariate_normal
import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class Debugger:
    
    fg_color_dict = dict(zip('krgybmcw', range(30,38)))
    bg_color_dict = dict(zip('krgybmcw', range(40,48)))
    
    def __init__(self):
        pass
    
    @staticmethod
    def retrieve_var_info(var):
        stacks = inspect.stack()
        try:
            call_func = stacks[1].function
            code = stacks[2].code_context[0]
            start_index = code.index(call_func)
            start_index = code.index("(", start_index + len(call_func)) + 1
            end_index = code.rindex(")")
            var_name = code[start_index:end_index].strip()
            return var_name, \
                   stacks[2].filename, \
                   stacks[2].lineno
        except:
            return "[VAR NAME NOT FIND]", "None", -1
    
    @staticmethod
    def dprint(var, concise=True)->None:
        var_name, filename, lineno = Debugger.retrieve_var_info(var)
        var_type = str(type(var)).split()[1].strip('\'>')
        if concise:
            filename = filename.rsplit('/')[-1]
            print('\n[{}: L{}][{}] {}:\n{}'.format(filename, lineno, var_type, Debugger.cformat(var_name), pformat(var)))
        else:
            print('[{}] {}'.format(var_type, Debugger.cformat(var_name)), end="\t")
            if hasattr(var, '__len__'):
                print('[len]: {}'.format(len(var)), end="\t")
            if hasattr(var, 'shape'):
                print('shape: {}'.format(var.shape), end="\t")
            print('[content]:\n{}'.format(pformat(var)))
        
    @staticmethod
    def cformat(in_str, font=1, fg_color='r', bg_color='y'):
        assert font in [0, 1, 4, 5, 7, 22, 24, 25, 27]
        if fg_color != '':
            assert fg_color in Debugger.fg_color_dict.keys()
            fg_idx = Debugger.fg_color_dict[fg_color]
        else:
            fg_idx = ''
        
        if bg_color != '':  
            assert bg_color in Debugger.bg_color_dict.keys()
            bg_idx = Debugger.bg_color_dict[bg_color]
        else:
            bg_idx = ''

        format_str = ('\033[{};{};{}m' + in_str + '\033[0m').format(font, fg_idx, bg_idx)
        return format_str
    
    @staticmethod
    def cprint(in_str, font=1, fg_color='r', bg_color='y'):
        format_str = Debugger.cformat(in_str, font, fg_color, bg_color)
        print(format_str)


def timer_decorater(click=False):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if click:
                tic = time()
                result = func(*args, **kwargs)
                toc = time() - tic
                print('{:<10} cost ={:>10.4f} ms'.format('[' + func.__name__ + ']', toc * 1e3))
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return inner
        

class timer:
    def __init__(self, click=False):
        self.click = click
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if self.click:
                tic = time()
                result = func(*args, **kwargs)
                toc = time() - tic
                print('{:<10} cost ={:>10.4f} ms'.format('[' + func.__name__ + ']', toc * 1e3))
                return result
            else:
                return func(*args, **kwargs)
        return wrapper

class prev_state:
    def __init__(self, initial_state=None, exception_type=Exception, save_prev=True):
        self.prev_state = initial_state
        self.exception_type = exception_type
        
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                self.prev_state = result
            except self.exception_type:
                result = self.prev_state
            return result
        return wrapper
    
def covariance_ellipse(P, thresh):
    eig, U = np.linalg.eig(P)
    orientation = np.math.atan2(U[1, 0], U[0, 0])
    width, height = np.sqrt(eig * thresh)
    return orientation, width, height

def plot_covariance_ellipse(mean, cov, percentile=0.95, color='b', ax=None, plot_center=False, 
                            plot_edge=False, scale=1, **kwargs):
    """ plot the covariance ellipse where mean is a (x,y) tuple for the mean
        of the covariance (center of ellipse)
        cov is a 2x2 covariance matrix
    """
    linewidth = kwargs.get('linewidth') or 2
    linestyle = kwargs.get('linestyle')  or '-'
    alpha= kwargs.get('alpha') or 0.7
    
    scale = _pair(scale)
    thresh = chi2.ppf(percentile, 2)
    angle, width, height = covariance_ellipse(cov, thresh)
    width *= scale[0]
    height *= scale[1]
    angle = np.degrees(angle)
    ax = ax or plt.gca()
    e = Ellipse(mean, 2 * width, 2 * height, angle, fill=False, edgecolor=color, linewidth=linewidth,
                linestyle=linestyle,  alpha=alpha)
    if plot_center:
        center_obj = ax.scatter(*mean, marker='+', color=color)
    else:
        center_obj = None
        
    if plot_edge:
        e_aug = Ellipse(mean, width, height, angle, fill=False, edgecolor='k', linewidth=5)
        ax.add_patch(e_aug)
    ax.add_patch(e)       
    return center_obj or e

def plot_covariance_ellipse_cv(image, mean, cov, percentile=0.995, color=(255, 255, 255), scale=1):
    """ opencv version
        plot the covariance ellipse where mean is a (x,y) tuple for the mean
        of the covariance (center of ellipse)
        cov is a 2x2 covariance matrix
    """
    scale = _pair(scale)
    thresh = chi2.ppf(percentile, 2)
    angle, width, height = covariance_ellipse(cov, thresh)
    width *= scale[0]
    height *= scale[1]
    width = np.round(width).astype(np.int)
    height= np.round(height).astype(np.int)
    angle = np.degrees(angle)
    
    cv2.ellipse(image, tuple(mean), (width, height), angle, 0, 360, color, 1)
    return image
    
if __name__ == "__main__":
    test_var = 1
    test_list = [1,2,3]
    test_dict = {'a': 1, 'b': 2}
    test_tuple = (1, 2, 3)
    test_set = set([1, 2, 3])
    
    Debugger.dprint(test_var)
    Debugger.dprint(test_list, False)
    Debugger.dprint(test_list, True)
    Debugger.dprint(test_dict, False)
    Debugger.dprint(test_tuple)
    Debugger.dprint(test_set)
    
    test_mat = np.random.rand(3,4)
    Debugger.dprint(test_mat, False)
    
    
    Debugger.cprint('test1', font=1, fg_color='w', bg_color='k')
    Debugger.cprint('test2')
    Debugger.cprint('test3', font=1, fg_color='b', bg_color='')
    
    mean = [0, 0]
    cov = 20**2*np.array([[1.5, -0.5] ,[-0.5, 0.5]])
    plot_covariance_ellipse(mean, cov, 0.4, 'k')
    plot_covariance_ellipse(mean, cov, 0.95, 'g')
    plot_covariance_ellipse(mean, cov, 0.99, 'r')
    rvs = multivariate_normal.rvs(mean, cov, 1000).T
    plt.plot(rvs[0], rvs[1], 'b.', markersize=2)
#    plot_covariance_ellipse([0,0], 400*np.eye(2), 0.9, plot_edge=True)
    plt.axis('equal')

    T = {'a':1, 'c':3}
    
    @timer(True)
    @prev_state(10)
    def foo(idx):
        return T[idx]
    
    print(foo('a1'))
    print(foo('b'))
    print(foo('c'))


