#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri May 15 17:03:24 2020

@author: zhaoxm
"""
#%%
import numpy as np
#from statecircle.lib.harem.debugger import plot_covariance_ellipse
from scipy.stats import multivariate_normal, chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
mvn = multivariate_normal

def covariance_ellipse(P, thresh):
    U, s, v = np.linalg.svd(P)
    orientation = np.math.atan2(U[1,0], U[0,0])
    width, height = 2 * np.sqrt(s * thresh)
    return orientation, width, height


def plot_covariance_ellipse(mean, cov, percentile=0.95, color='b', ax=None, plot_center=True, plot_edge=False, scale=1):
    """ plot the covariance ellipse where mean is a (x,y) tuple for the mean
        of the covariance (center of ellipse)
        cov is a 2x2 covariance matrix
    """
    thresh = chi2.ppf(percentile, 2)
    angle, width, height = covariance_ellipse(cov, thresh)
    angle = np.degrees(angle)
    width *= scale
    height *= scale
    ax = ax or plt.gca()
    
    e = Ellipse(mean, width, height, angle, fill=False, edgecolor=color, linewidth=2)
    if plot_center:
        center_obj = ax.scatter(*mean, marker='+', color=color)
    else:
        center_obj = None
    if plot_edge:
        e_aug = Ellipse(mean, width, height, angle, fill=False, edgecolor='k', linewidth=5)
        ax.add_patch(e_aug)
    ax.add_patch(e)       
    return center_obj

#%% initial state
m = np.array([0, 0, 1, 0.5])
P = np.diag([100, 100, 100, 100])
#%% transition model
dt = 1
F = np.array([[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]])
sigma  = 1
Q =  sigma ** 2 * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                            [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                            [dt ** 3 / 2, 0, dt ** 2, 0],
                            [0, dt ** 3 / 2, 0, dt ** 2]])
#%% measurement model
meas_dim = 2
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
est_sigma = 3
est_R = est_sigma ** 2 * np.diag([1, 2])

real_sigma = 5
real_R = real_sigma ** 2 * np.diag([1, 2])

#%% generate dataset
T = 100
x0 = np.array([0, 0, 1, 1])
x = x0
meas = []
for t in range(T):
    x = F.dot(x) + mvn.rvs(cov=Q)
    z = H.dot(x) + mvn.rvs(cov=real_R)
    meas.append(z)
meas = np.array(meas)

#%% plot measurements
plt.plot(meas[:,0], meas[:,1], '.')

#%% NIS(NOrmalized Innovation Squared)
def NIS(z_pred, mea, S):
    res = (z_pred - mea)
    # res.T * S^{-1} * res
    return res.dot(np.linalg.inv(S)).dot(res)

#%%
scope = 100
percentile = 0.999
nis_thresh = chi2.ppf(percentile, meas_dim)
NIS_trace = []
for i in range(T):
    # predict
    m_pred = F.dot(m)
    P_pred = F.dot(P).dot(F.T) + Q
    
    # update
    z_pred = H.dot(m_pred)
    S = H.dot(P_pred).dot(H.T) + est_R
    K =  P_pred.dot(H.T).dot(np.linalg.inv(S))
    
    m_upd = m_pred + K.dot(meas[i] - z_pred)
    P_upd = P_pred - K.dot(S).dot(K.T)
    
    m = m_upd
    P = P_upd
    
    NIS_trace.append(NIS(z_pred, meas[i], S))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#    ax.plot(m[0], m[1], 'b.')
    ax[0].plot(meas[i][0], meas[i][1], 'rx')
    plot_covariance_ellipse(m[:2], P[:2,:2], percentile=0.999, color='b', 
                            ax=ax[0], plot_edge=False, scale=1)
    plot_covariance_ellipse(z_pred, S, percentile=0.999, color='r', 
                            ax=ax[0], plot_center=False, plot_edge=False, scale=1)
    ax[0].axis([m[0] - scope, m[0] + scope, m[1] - scope, m[1] + scope])
#    ax[0].axis('equal')
    ax[1].hlines(nis_thresh, 0, T, 'r')
    ax[1].plot(NIS_trace)
    ax[0].grid(True)
    ax[1].grid(True)
    plt.show()
    print('')
    plt.close('all')
    