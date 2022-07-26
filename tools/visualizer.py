#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Dec 24 16:13:12 2019

@author: zhaoxm
"""
import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, data_generator, name='trial'):
        self.data_generator = data_generator
        self.name = name
        
    def show_estimates(self, estimates):
        gt_data = np.hstack([ele.states for ele in self.data_generator])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plot_gt = ax.plot(gt_data[0], gt_data[1], 'yo', alpha=0.2, markersize=10)
        plot_est = ax.plot(estimates[0], estimates[1], 'g+', alpha=0.5)
        
        ax.grid()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.legend((plot_gt[0], plot_est[0]), ['Ground Truth', self.name])
        plt.axis('equal')
    
    def show_cardinality(self, card_pred):
        plt.figure()
        plt.plot(self.data_generator.gt_series.num, 'yo')
        plt.plot(card_pred, 'b+')
        plt.legend(['GT', self.name])
        plt.grid()
    
    def show_measurements(self, data_reader):
        meas, obj_meas, clutter_meas = [], [], []
        for meas_data, obj_meas_, clutter_meas_ in data_reader.truth_meas_generator():
            meas.append(meas_data.meas)
            obj_meas.append(obj_meas_)
            clutter_meas.append(clutter_meas_)
        meas, obj_meas, clutter_meas = np.hstack(meas), np.hstack(obj_meas), np.hstack(clutter_meas)
        
        plt.figure()
        plt.plot(obj_meas[0], obj_meas[1], 'r.', alpha=0.5)
        
        # plot clutter
        plt.plot(clutter_meas[0], clutter_meas[1], 'k.', alpha=0.2)
        plt.legend(['measurements', 'clutter'])
        plt.show()
        plt.close('all')