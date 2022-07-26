#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Sun May 24 16:06:40 2020

@author: zhaoxm
"""
import numpy as np
#%%
meas_data = data_reader[302]
print(meas_data.sensor_type)
for obj in meas_data.objects:
    vx = obj.VeloX
    vy = obj.VeloY
    x = obj.WldX
    y = obj.WldY
    thetap = np.arctan2(y,x)
    thetav = np.arctan2(vy,vx)
    print('x:{}, y:{}'.format(x, y))
    print('vx:{}, vy:{}'.format(vx, vy))
    print('p:{} vs v:{}'.format(thetap * 180/np.pi, thetav*180/np.pi))
    
    #%%
dd = {
     'WldX': -1.28395,
     'WldY': 87.4402,
     'WldWidth': 0.0,
     'WldHeight': 1.6,
     'Dist': 87.4497,
     'Angle': -89.1587,
     'VeloX': 0.107444,
     'VeloY': -14.7395,
     }

np.arctan2(dd['WldY'], dd['WldX']) * 180 / np.pi
np.arctan2(dd['VeloY'], dd['VeloX']) * 180 / np.pi
     