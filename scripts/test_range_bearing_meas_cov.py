#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Jun  2 10:45:27 2020

@author: zhaoxm
"""

#%%
import numpy as np
from scipy.stats import chi2
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
def covariance_ellipse(P, thresh):
    U, s, v = np.linalg.svd(P)
    orientation = np.math.atan2(U[1,0], U[0,0])
    width, height = 2 * np.sqrt(s * thresh)
    return orientation, width, height

def plot_covariance_ellipse(mean, cov, percentile=0.95, color='b', ax=None, plot_center=False, plot_edge=False):
    """ plot the covariance ellipse where mean is a (x,y) tuple for the mean
        of the covariance (center of ellipse)
        cov is a 2x2 covariance matrix
    """
    thresh = chi2.ppf(percentile, 2)
    angle, width, height = covariance_ellipse(cov, thresh)
    angle = np.degrees(angle)
    ax = ax or plt.gca()
    
    e = Ellipse(mean, width, height, angle, fill=False, edgecolor=color, linewidth=2, linestyle='-',  alpha=0.7)
    if plot_center:
        center_obj = ax.scatter(*mean, marker='+', color=color)
    else:
        center_obj = None
        
    if plot_edge:
        e_aug = Ellipse(mean, width, height, angle, fill=False, edgecolor='k', linewidth=5)
        ax.add_patch(e_aug)
    ax.add_patch(e)       
    return center_obj or e



def draw_ellipse(var_x, var_y, r, phi):

    R = np.diag([var_x, var_y])
    
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    J = np.array([[np.cos(phi), -r * np.sin(phi)],
              [np.sin(phi), r * np.cos(phi)]])
    
    cov = J.dot(R).dot(J.T) 
    return np.array([x, y]), cov


var_x = 1
var_y = 0.001
r = 10
phi = np.pi / 3
res = draw_ellipse(var_x, var_y, r, phi)
fig, ax = plt.subplots(1, 1)
plot_covariance_ellipse(*res, plot_center=True)
plt.axis([-30, 30, 0, 50])
ax.set_aspect(1)

