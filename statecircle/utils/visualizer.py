#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Thu Jun  4 20:42:41 2020

@author: zhxm
"""
from os.path import join as opj
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from pathlib import Path

from statecircle.utils.common import tuple2int
from statecircle.lib.harem.debugger import plot_covariance_ellipse, plot_covariance_ellipse_cv

FONT = (cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN)

class PLTVisualizer:
    def __init__(self, data_path, show_map=False, map_scope=100):
        self.data_path = data_path
        self.img_name_len = len(next(Path(data_path).rglob('*.jpg')).stem)
        
        self.fig = plt.figure(figsize=(12, 6))
         # handy parameters
        left, bottom = 0.02, 0.07
        width, height = 0.7, 1 - 2 * bottom
        rect_image = [left, bottom, width, height]
        rect_radar = [left + width + 0.02, bottom, 1 - left - width - 0.05, height]
        self.image_height, self.image_width = 360, 640
        self.image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        self.scope = [-40, 40, 0, 180]
        
        ax_image = plt.axes(rect_image)
        ax_image.tick_params(direction='in', top=True, right=True)
        ax_radar = plt.axes(rect_radar)
        ax_radar.tick_params(direction='in', top=True, right=True)
        ax_radar.yaxis.tick_right()
        ax_radar.yaxis.set_label_position('right')
        
        self.fig.canvas.mpl_connect('key_press_event', self.exit_press)
        
        self.ax_image = ax_image
        self.ax_radar = ax_radar
        
        self.show_map = show_map
        self.plot_track = [None]
        if show_map:
            self.fig_map, self.ax_map = plt.subplots(1, 1, figsize=(7, 7))
            self.fig_map.tight_layout()
            self.map_scope = map_scope
            self.vision_coords = []
            self.radar_coords = []
            
            self.plot_camera_meas = [None]
            self.plot_radar_meas = [None]
            self.plot_stable_lm = [None]
            self.plot_mainobj = [None]
            self.plot_track_map = [None]
            
        self.exit = False
    
    def exit_press(self, event):
        print('key pressed: ', format(event.key))
        if event.key in ['escape']:
            print('exit!!!')
            self.exit = True
        
    def visualize(self, idx, data, show_id=False, show_radar=True, show_camera=True):
        self.ax_image.cla()
        self.ax_radar.cla()
        
        # draw image map
        format_str = '{:0' + str(self.img_name_len) + 'd}.jpg'
        image_path = opj(self.data_path, format_str.format(data.frame_id))
        self.image = cv2.imread(image_path) if os.path.exists(image_path) else self.image
        self.image_height, self.image_width, _ = self.image.shape
        self.ax_image.imshow(self.image)
    
        # draw bounding box and radar detection points
        # draw vision bounding box
        for obj in data.vision_objects:
            rec = Rectangle((obj.ImgX, obj.ImgY), obj.ImgWidth, obj.ImgHeight, color='r', fill=False)
            self.ax_image.add_patch(rec)
    
        # draw radar detection points
        rc = data.radar_image_coords
        valid_idx = (rc[0] >=0) * (rc[1] >=0) * (rc[0] <=self.image_width) * (rc[1] <= self.image_height)
        rc = rc[:, valid_idx]
        self.ax_image.plot(rc[0], rc[1], 'yo', alpha=0.5)
    
        # draw ipm map
        if show_camera:
            self.ax_radar.plot(data.vision_world_coords[0], data.vision_world_coords[1], 'r.', alpha=0.7)
            
        if show_radar:
            def plot_radar_meas(index, color):
                self.ax_radar.plot(data.radar_world_coords[0, index], data.radar_world_coords[1, index], color+'.', alpha=0.7)
                self.ax_radar.quiver(radar_coord_velo[0, index], radar_coord_velo[1, index], 
                                     radar_coord_velo[2, index], radar_coord_velo[3, index],
                                     color=color, alpha=0.5, scale=2, scale_units='xy')
            
            radar_coord_velo = data.radar_world_coords_velo
            
            vel = data.ego_info['Velocity'] / 3.6
            radar_x_vel = radar_coord_velo[2]
            radar_y_vel = radar_coord_velo[3]
            static_velocity_res = 1.5
            motion_idx = (np.abs(radar_y_vel + vel) > static_velocity_res) # | (np.abs(radar_x_vel) > static_velocity_res)
            
            plot_radar_meas(motion_idx, 'y')
            plot_radar_meas(~motion_idx, 'c')

        if show_id:
            if show_camera:
                for obj in data.vision_objects:
                    if 0 <= obj.ImgX < self.image_width and 0 <= obj.ImgY < self.image_height:
                        self.ax_image.text(obj.ImgX, obj.ImgY, int(obj.ID), fontsize=8, color='r')
        
                    if self.scope[0] <= obj.WldX < self.scope[1] and self.scope[2] <= obj.WldY < self.scope[3]:
                        self.ax_radar.text(obj.WldX, obj.WldY, int(obj.ID), fontsize=8, color='r')
            
            if show_radar:
                for obj in data.radar_objects:
                    if 0 <= obj.ImgX < self.image_width and 0 <= obj.ImgY < self.image_height:
                        self.ax_image.text(obj.ImgX, obj.ImgY, int(obj.ID), fontsize=8, color='y')
        
                    if self.scope[0] <= obj.WldX < self.scope[1] and self.scope[2] <= obj.WldY < self.scope[3]:
                        self.ax_radar.text(obj.WldX, obj.WldY, int(obj.ID), fontsize=8, color='y')
                    
        self.ax_image.set_title('[{}]frame: {}'.format(idx, data.frame_id))
        self.ax_radar.set_title('timestamp: {:.3f}ms'.format(data.milli_seconds))
        
        return self.exit
    
    def plot_tracks(self, k, estimates, ego_trace=None, cov_scale=1):
        k_ = k if k == 0 else k - 1
        for track in estimates[k_]:
            # range_t = track['range'][-1] - track['range'][0] + 1
            if track['range'][-1] > 0 and  track['range'][-1] in [k_, k_-1, k_-2]:
                pts = np.array([track['trajectory'][0], track['trajectory'][1]])
                plot_covariance_ellipse(pts[:, -1], track['cov'][:2, :2], ax=self.ax_radar, 
                                        color='w', scale=cov_scale, linewidth=1)
                self.plot_track = self.ax_radar.plot(pts[0], pts[1], 'lightgreen')
                
                if self.show_map:
                    pts_world = trans_to_world_coords(pts, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
                    self.plot_track_map = self.ax_map.plot(pts_world[0], pts_world[1], 'lightgreen')
    
    def plot_map(self, k, merge_frame, ego_trace, 
                  landmark_trace, cov_trace, stable_landmarks, show_id=False):
        
        # ego_trace [num_pts, 2]
        cur_vision_coords = []
        for col in range(merge_frame.vision_obj_count):
            coords = merge_frame.vision_world_coords[:2, col]
            coords = trans_to_world_coords(coords, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
            cur_vision_coords.append(coords)
            self.vision_coords.append(coords)
            
        cur_radar_coords = []
        for col in range(merge_frame.radar_obj_count):
            coords = merge_frame.radar_world_coords[:2, col]
            coords = trans_to_world_coords(coords, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
            cur_radar_coords.append(coords)
            self.radar_coords.append(coords)
            
        cur_vision_coords = np.array(cur_vision_coords).T    
        cur_radar_coords = np.array(cur_radar_coords).T    
        vc = np.array(self.vision_coords).T
        rc = np.array(self.radar_coords).T
        
        ax = self.ax_map
        ax.cla()
        
        if len(vc) > 0:
            self.plot_camera_meas = ax.plot(cur_vision_coords[0], cur_vision_coords[1], 'r.', markersize=8, alpha=0.9)
            ax.plot(vc[0], vc[1], 'r.', markersize=1, alpha=0.5)
        if len(rc) > 0:
            self.plot_radar_meas = ax.plot(cur_radar_coords[0], cur_radar_coords[1], 'y.', markersize=8, alpha=0.9)
            ax.plot(rc[0], rc[1], 'y.', markersize=1, alpha=0.5)
            
        self.plot_ego = ax.plot(ego_trace[:k, 0], ego_trace[:k, 1], color=[0, 1, 1])
        self.plot_lm = ax.plot(landmark_trace[k][0], landmark_trace[k][1], '.', color=[0, 1, 1], markersize=5, alpha=0.5)
        
        for i, lm in enumerate(landmark_trace[k].T):
            # ax.text(*lm, lbl, verticalalignment='top', fontsize=8, color='r')
            cov_ = cov_trace[k][3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
            plot_covariance_ellipse(lm, cov_, percentile=0.999, ax=ax)
        self.plot_cov = plot_covariance_ellipse(ego_trace[k][:2], cov_trace[k][:2, :2], percentile=0.999, ax=ax)
        
        if k > 0:
            stable_lms = np.hstack(stable_landmarks[:k])
            self.plot_stable_lm = ax.plot(stable_lms[0], stable_lms[1], '.', color='chartreuse', markersize=3)
        
        self.end_point = ego_trace[k]
    
    def show(self, show=True, save_name=None, pause=1e-3):
        self.ax_image.axis([0, self.image_width, self.image_height, 0])
        self.ax_radar.axis(self.scope)
        # self.ax_radar.set_aspect(1)
        self.ax_radar.grid()
        self.ax_radar.set_xlabel('x (m)')
        self.ax_radar.set_ylabel('y (m)')
        self.ax_radar.legend(['cam', 'radar', 'static'], loc='upper right')
        self.ax_radar.grid(linestyle='--', alpha=0.4)
        self.fig.canvas.draw_idle()
        
        if self.show_map:
            ax = self.ax_map
            end_point = self.end_point
            ax.plot(end_point[0], end_point[1], 'ws', color=[0, 1, 1], markersize=3)
            ax.set_xlim([end_point[0] - self.map_scope, end_point[0] + self.map_scope])
            ax.set_ylim([end_point[1] - self.map_scope, end_point[1] + self.map_scope])
            ax.legend((self.plot_camera_meas[0], self.plot_radar_meas[0], self.plot_ego[0], 
                       self.plot_lm[0], self.plot_stable_lm[0], 
                       self.plot_track_map[0], self.plot_cov, self.plot_mainobj[0]),
                  ['camera meas.', 'radar meas.', 'ego trace', 'landmarks', 'stable landmarks', 'tracks',
                   'gating cov', 'main obj.'],
                  loc='upper right')
            ax.set_aspect(1)
            ax.grid(linestyle='--', alpha=0.4)
            self.fig_map.canvas.draw_idle()
            
        if show:
            plt.pause(pause/1e3)
            
        if save_name is not None:
            self.fig.savefig(save_name)
            
            if self.show_map:
                map_save_path = (Path(save_name).parent / 'maps')
                map_save_path.mkdir(exist_ok=True)
                map_save_name = str(map_save_path / Path(save_name).name)
                self.fig_map.savefig(map_save_name)


class CVVisualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_name_len = len(next(Path(data_path).rglob('*.jpg')).stem)
        self.image_height, self.image_width = 360, 640
        self.radar_image_height = self.image_height
        self.radar_image_width = self.radar_image_height // 2
        self.scope = [-10, 10, 0, 220]
        self.radar_scale = [self.radar_image_width / (self.scope[1] - self.scope[0]), 
                            self.radar_image_height / (self.scope[3] - self.scope[2])] 
        
    def visualize(self, idx, data, show_id=False, show_radar=True, show_camera=True, verbose=False):
        # handy parameters
        if verbose:
            print('frame id:{}, #vision meas: {}, #radar meas: {}'\
                  .format(data.frame_id, len(data.vision_objects), len(data.radar_objects)))
            
        def trans_radar_meas_to_scale_image(x, y):
            x -= self.scope[0]
            x *= self.radar_scale[0]
            y *= self.radar_scale[1]
            return x, y
        
        def radar_meas_wrapper(x, y):
            return tuple2int(*trans_radar_meas_to_scale_image(x, y))
        
        # draw image map
        format_str = '{:0' + str(self.img_name_len) + 'd}.jpg'
        image_path = opj(self.data_path, format_str.format(data.frame_id))
        if os.path.exists(image_path):
            vision_img = cv2.imread(image_path)
            self.prev_img = vision_img
        else:
            if hasattr(self, 'prev_img'):
                vision_img = self.prev_img
            else:
                vision_img = np.zeros((self.image_height, self.image_width, 3))
            
        image_height, image_width, _ = vision_img.shape
    
        # draw bounding box and radar detection points
        # draw vision bounding box
        for obj in data.vision_objects:
            cv2.rectangle(vision_img, tuple2int(obj.ImgX, obj.ImgY), 
                          tuple2int(obj.ImgX + obj.ImgWidth - 1, obj.ImgY + obj.ImgHeight - 1),
                          (0,0,255), 1)
    
        # draw radar detection points
        rc = data.radar_image_coords
        valid_idx = (rc[0] >=0) * (rc[1] >=0) * (rc[0] <=image_width) * (rc[1] <= image_height)
        rc = rc[:, valid_idx]
        radius = 2
        thickness = -1
        font_scale = 0.4
        font_thickness = 1
        for col in range(rc.shape[1]):
            cv2.circle(vision_img, tuple2int(rc[:, col][0], rc[:, col][1]), radius, (0, 255, 255), thickness)
    
        # draw ipm map
        radar_img = np.zeros((self.radar_image_height, self.radar_image_width, 3), dtype=np.uint8)
        if show_camera:
            for col in range(data.vision_obj_count):
                cv2.circle(radar_img, radar_meas_wrapper(data.vision_world_coords[:, col][0], data.vision_world_coords[:, col][1]),
                       radius, (0,0,255), thickness)
                
        if show_radar:
            for col in range(data.radar_obj_count):
                vel = data.ego_info['Velocity'] / 3.6
                radar_x_vel = data.radar_objects[col].VeloX
                radar_y_vel = data.radar_objects[col].VeloY
                static_velocity_res = 1.5
                if (np.abs(radar_y_vel + vel) > static_velocity_res): # | (np.abs(radar_x_vel) > static_velocity_res):
                    circle_color = (0, 255, 255)
                else:
                    circle_color = (255, 255, 0)
                cv2.circle(radar_img, radar_meas_wrapper(data.radar_world_coords[:, col][0], data.radar_world_coords[:, col][1]),
                       radius, circle_color, thickness)
        
        if show_id:
            if show_camera:
                for obj in data.vision_objects:
                    if 0 <= obj.ImgX < image_width and 0 <= obj.ImgY < image_height:
                        cv2.putText(vision_img, str(int(obj.ID)), tuple2int(obj.ImgX, obj.ImgY), FONT[0],
                                    font_scale, (0,0,255), font_thickness)
        
                    if self.scope[0] <= obj.WldX < self.scope[1] and self.scope[2] <= obj.WldY < self.scope[3]:
                        cv2.putText(radar_img, str(int(obj.ID)), radar_meas_wrapper(obj.WldX, obj.WldY), FONT[0],
                                    font_scale, (0,0,255), font_thickness)
    
            if show_radar:
                for obj in data.radar_objects:
                    if 0 <= obj.ImgX < image_width and 0 <= obj.ImgY < image_height:
                        cv2.putText(vision_img, str(int(obj.ID)), tuple2int(obj.ImgX, obj.ImgY), FONT[0],
                                    font_scale, (0,255,255), font_thickness)
        
        
                    if self.scope[0] <= obj.WldX < self.scope[1] and self.scope[2] <= obj.WldY < self.scope[3]:
                        cv2.putText(radar_img, str(int(obj.ID)), radar_meas_wrapper(obj.WldX, obj.WldY), FONT[0],
                                    font_scale, (0,255,255), font_thickness)
                    
        self.ax_image, self.ax_radar = vision_img, radar_img
        return False
    
    def plot_tracks(self, k, estimates, ego_trace=None, cov_scale=1):
        k_ = k if k == 0 else k - 1
        for track in estimates[k_]:
            # range_t = track['range'][-1] - track['range'][0] + 1
            if track['range'][-1] > 0 and  track['range'][-1] in [k_, k_-1, k_-2]:
                pts = np.array([(track['trajectory'][0] + 10) * self.radar_scale[0], 
                                 track['trajectory'][1] * self.radar_scale[1]])\
                                .transpose().reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(self.ax_radar, [pts], False, (0, 255, 0), 1)
                cov_radar_scale = [ele * cov_scale for ele in self.radar_scale]
                plot_covariance_ellipse_cv(self.ax_radar, pts[-1, 0], track['cov'][:2, :2], scale=cov_radar_scale)
                
    def plot_map(self, k, merge_frame, ego_trace, 
                  landmark_trace, cov_trace, stable_landmarks, show_id=False):
        # TODO: implement plot_map funciton"
        pass
                
    def show(self, show=True, save_name=None, pause=1):
        stack_image = np.hstack((self.ax_image[:,:,::-1], self.ax_radar[::-1,:,::-1]))
        
        if show:
            cv2.imshow('fusion result', stack_image[:,:,::-1])
            cv2.waitKey(int(pause))
            
        if save_name is not None:
            cv2.imwrite(save_name, stack_image[:, :, ::-1])
        
    
class Visualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_name_len = len(next(Path(data_path).rglob('*.jpg')).stem)
        
    def visualize(self, data, show_id=False):
        # handy parameters
        left, bottom = 0.1, 0.1
        width, height = 0.7, 0.95
        rect_image = [left, bottom, width, height]
        rect_radar = [left + width, bottom, 0.29, height]
        image_height, image_width = 360, 640
        scope = [-10, 10, 0, 220]
    
        plt.figure(figsize=(12, 5))
        ax_image = plt.axes(rect_image)
        ax_image.tick_params(direction='in', top=True, right=True)
        ax_radar = plt.axes(rect_radar)
        ax_radar.tick_params(direction='in', top=True, right=True)
        ax_radar.yaxis.tick_right()
        ax_radar.yaxis.set_label_position('right')
    
        # draw image map
        format_str = '{:0' + str(self.img_name_len) + 'd}.jpg'
        image_path = opj(self.data_path, format_str.format(data.frame_id))
        image = plt.imread(image_path) if os.path.exists(image_path) else np.zeros((image_height, image_width, 3))
        image_height, image_width, _ = image.shape
        ax_image.imshow(image)
    
        # draw bounding box and radar detection points
        # draw vision bounding box
        for obj in data.vision_objects:
            rec = Rectangle((obj.ImgX, obj.ImgY), obj.ImgWidth, obj.ImgHeight, color='r', fill=False)
            ax_image.add_patch(rec)
    
        # draw radar detection points
        rc = data.radar_image_coords
        valid_idx = (rc[0] >=0) * (rc[1] >=0) * (rc[0] <=image_width) * (rc[1] <= image_height)
        rc = rc[:, valid_idx]
        ax_image.plot(rc[0], rc[1], 'y+')
    
        # draw ipm map
        ax_radar.plot(data.vision_world_coords[0], data.vision_world_coords[1], 'r*')
        ax_radar.plot(data.radar_world_coords[0], data.radar_world_coords[1], 'y+')
    
        if show_id:
            for obj in data.vision_objects:
                if 0 <= obj.ImgX < image_width and 0 <= obj.ImgY < image_height:
                    ax_image.text(obj.ImgX, obj.ImgY, int(obj.ID), fontsize=8, color='r')
    
                if scope[0] <= obj.WldX < scope[1] and scope[2] <= obj.WldY < scope[3]:
                    ax_radar.text(obj.WldX, obj.WldY, int(obj.ID), fontsize=8, color='r')
    
            for obj in data.radar_objects:
                if 0 <= obj.ImgX < image_width and 0 <= obj.ImgY < image_height:
                    ax_image.text(obj.ImgX, obj.ImgY, int(obj.ID), fontsize=8, color='y')
    
                if scope[0] <= obj.WldX < scope[1] and scope[2] <= obj.WldY < scope[3]:
                    ax_radar.text(obj.WldX, obj.WldY, int(obj.ID), fontsize=8, color='y')
    
        ax_image.axis([0, image_width, image_height, 0])
        ax_image.set_title('frame: {}'.format(data.frame_id))
    
        ax_radar.axis(scope)
        ax_radar.set_aspect(0.2)
        ax_radar.set_title('timestamp: {:.3f}ms'.format(data.milli_seconds))
        ax_radar.grid()
        ax_radar.set_xlabel('x (m)')
        ax_radar.set_ylabel('y (m)')
        ax_radar.legend(['camera', 'radar'])
        return ax_image, ax_radar

    def visualize_cv(self, data, show_id=False, verbose=False):
        # handy parameters
        image_height, image_width = 360, 640
        radar_image_height = image_height
        radar_image_width = radar_image_height // 2
        scope = [-10, 10, 0, 220]
        radar_scale = [radar_image_width / (scope[1] - scope[0]), radar_image_height / (scope[3] - scope[2])] 
        if verbose:
            print('frame id:{}, #vision meas: {}, #radar meas: {}'\
                  .format(data.frame_id, len(data.vision_objects), len(data.radar_objects)))
            
        def trans_radar_meas_to_scale_image(x, y):
            x -= scope[0]
            x *= radar_scale[0]
            y *= radar_scale[1]
            return x, y
        
        def radar_meas_wrapper(x, y):
            return tuple2int(*trans_radar_meas_to_scale_image(x, y))
        
        # draw image map
        format_str = '{:0' + str(self.img_name_len) + 'd}.jpg'
        image_path = opj(self.data_path, format_str.format(data.frame_id))
        if os.path.exists(image_path):
            vision_img = cv2.imread(image_path)
            self.prev_img = vision_img
        else:
            if hasattr(self, 'prev_img'):
                vision_img = self.prev_img
            else:
                vision_img = np.zeros((image_height, image_width, 3))
            
        image_height, image_width, _ = vision_img.shape
    
        # draw bounding box and radar detection points
        # draw vision bounding box
        for obj in data.vision_objects:
            cv2.rectangle(vision_img, tuple2int(obj.ImgX, obj.ImgY), 
                          tuple2int(obj.ImgX + obj.ImgWidth - 1, obj.ImgY + obj.ImgHeight - 1),
                          (0,0,255), 1)
    
        # draw radar detection points
        rc = data.radar_image_coords
        valid_idx = (rc[0] >=0) * (rc[1] >=0) * (rc[0] <=image_width) * (rc[1] <= image_height)
        rc = rc[:, valid_idx]
        radius = 2
        thickness = -1
        font_scale = 0.4
        font_thickness = 1
        for col in range(rc.shape[1]):
            cv2.circle(vision_img, tuple2int(rc[:, col][0], rc[:, col][1]), radius, (0, 255, 255), thickness)
    
        # draw ipm map
        radar_img = np.zeros((radar_image_height, radar_image_width, 3), dtype=np.uint8)
        for col in range(data.vision_obj_count):
            cv2.circle(radar_img, radar_meas_wrapper(data.vision_world_coords[:, col][0], data.vision_world_coords[:, col][1]),
                   radius, (0,0,255), thickness)
        for col in range(data.radar_obj_count):
            cv2.circle(radar_img, radar_meas_wrapper(data.radar_world_coords[:, col][0], data.radar_world_coords[:, col][1]),
                   radius, (0,255,255), thickness)
        
        if show_id:
            for obj in data.vision_objects:
                if 0 <= obj.ImgX < image_width and 0 <= obj.ImgY < image_height:
                    cv2.putText(vision_img, str(int(obj.ID)), tuple2int(obj.ImgX, obj.ImgY), FONT[0],
                                font_scale, (0,0,255), font_thickness)
    
                if scope[0] <= obj.WldX < scope[1] and scope[2] <= obj.WldY < scope[3]:
                    cv2.putText(radar_img, str(int(obj.ID)), radar_meas_wrapper(obj.WldX, obj.WldY), FONT[0],
                                font_scale, (0,0,255), font_thickness)
    
            for obj in data.radar_objects:
                if 0 <= obj.ImgX < image_width and 0 <= obj.ImgY < image_height:
                    cv2.putText(vision_img, str(int(obj.ID)), tuple2int(obj.ImgX, obj.ImgY), FONT[0],
                                font_scale, (0,255,255), font_thickness)
    
    
                if scope[0] <= obj.WldX < scope[1] and scope[2] <= obj.WldY < scope[3]:
                    cv2.putText(radar_img, str(int(obj.ID)), radar_meas_wrapper(obj.WldX, obj.WldY), FONT[0],
                                font_scale, (0,255,255), font_thickness)
        
        return vision_img, radar_img


def trans_to_world_coords(coords, T, theta):
    # coords: [2, #pts]
    R = np.array([[np.cos(theta), np.sin(-theta)],
                  [np.sin(theta), np.cos(theta)]])
    if coords.ndim == 2:
        T = T[:, None]
    return R.dot(coords) + T


def plot_ego_slam(ax, k, merge_frame, vision_coords, radar_coords, ego_trace, 
                  landmark_trace, cov_trace, stable_landmarks, show_id=False):
    plot_camera_meas = [None]
    plot_radar_meas = [None]
    plot_stable_lm = [None]
    
    # ego_trace [num_pts, 2]
    cur_vision_coords = []
    for col in range(merge_frame.vision_obj_count):
        coords = merge_frame.vision_world_coords[:2, col]
        coords = trans_to_world_coords(coords, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
        cur_vision_coords.append(coords)
        vision_coords.append(coords)
        
    cur_radar_coords = []
    for col in range(merge_frame.radar_obj_count):
        coords = merge_frame.radar_world_coords[:2, col]
        coords = trans_to_world_coords(coords, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
        cur_radar_coords.append(coords)
        radar_coords.append(coords)
        
    cur_vision_coords = np.array(cur_vision_coords).T    
    cur_radar_coords = np.array(cur_radar_coords).T    
    vc = np.array(vision_coords).T
    rc = np.array(radar_coords).T
    
    ax.cla()
    
    if len(vc) > 0:
        plot_camera_meas = ax.plot(cur_vision_coords[0], cur_vision_coords[1], 'r.', markersize=8, alpha=0.9)
        ax.plot(vc[0], vc[1], 'r.', markersize=1, alpha=0.5)
    if len(rc) > 0:
        plot_radar_meas = ax.plot(cur_radar_coords[0], cur_radar_coords[1], 'y.', markersize=8, alpha=0.9)
        ax.plot(rc[0], rc[1], 'y.', markersize=1, alpha=0.5)
        
    plot_ego = ax.plot(ego_trace[:k, 0], ego_trace[:k, 1], color=[0, 1, 1])
    plot_lm = ax.plot(landmark_trace[k][0], landmark_trace[k][1], '.', color=[0, 1, 1], markersize=5, alpha=0.5)
    
    for i, lm in enumerate(landmark_trace[k].T):
        # ax.text(*lm, lbl, verticalalignment='top', fontsize=8, color='r')
        cov_ = cov_trace[k][3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
        plot_covariance_ellipse(lm, cov_, percentile=0.999, ax=ax)
    plot_cov = plot_covariance_ellipse(ego_trace[k][:2], cov_trace[k][:2, :2], percentile=0.999, ax=ax)
    
    if k > 0:
        stable_lms = np.hstack(stable_landmarks[:k])
        plot_stable_lm = ax.plot(stable_lms[0], stable_lms[1], '.', color='chartreuse', markersize=3)
        
    return plot_camera_meas, plot_radar_meas, plot_ego, plot_lm, plot_cov, plot_stable_lm

def plot_tracks_cv(k, ax_image, ax_radar, estimates, radar_scale, proj_plane):
    k_ = k if k == 0 else k - 1
    for track in estimates[k_]:
        if proj_plane == 'camera':
            ax_draw = ax_image
        elif proj_plane == 'radar':
            ax_draw = ax_radar
        else:
            raise TypeError

        # range_t = track['range'][-1] - track['range'][0] + 1
        if track['range'][-1] > 0 and  track['range'][-1] in [k_, k_-1, k_-2]:
            pts = np.array([(track['trajectory'][0] + 10) * radar_scale[0], 
                             track['trajectory'][1] * radar_scale[1]])\
                            .transpose().reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(ax_draw, [pts], False, (0, 255, 0), 1)
            plot_covariance_ellipse_cv(ax_draw, pts[-1, 0], track['cov'][:2, :2], scale=radar_scale)
            
def plot_tracks(k, ax, ax_image, ax_radar, estimates, ego_trace, radar_scale, proj_plane):
    k_ = k if k == 0 else k - 1
    plot_track = [None]
    for track in estimates[k_]:
        if proj_plane == 'camera':
            ax_draw = ax_image
        elif proj_plane == 'radar':
            ax_draw = ax_radar
        else:
            raise TypeError

        range_t = track['range'][-1] - track['range'][0] + 1
        if track['range'][-1] > 0 and  track['range'][-1] in [k_, k_-1, k_-2]:
            pts = np.array([(track['trajectory'][0] + 10) * radar_scale[0], 
                             track['trajectory'][1] * radar_scale[1]])\
                            .transpose().reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(ax_draw, [pts], False, (0, 255, 0), 1)
            plot_covariance_ellipse_cv(ax_draw, pts[-1, 0], track['cov'][:2, :2], scale=5)
            
#            cv2.putText(ax_draw, str(int(meas_data.frame_id)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
#                        1, (0,0,255), 1)
            
            pts = np.array([track['trajectory'][0], track['trajectory'][1]])
            pts = trans_to_world_coords(pts, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
            plot_track = ax.plot(pts[0], pts[1], 'lightgreen')
    
#            if proj_plane == 'radar':
#            # project to image plane
#                coords = track['trajectory'][:2]
#                coords = np.vstack((coords, np.ones(coords.shape[1])))
#                coords = HOMOGRAPHY.dot(coords)
#                coords[:2] /= 2*coords[2]
#                ax_image.plot(coords[0], coords[1], '-')
    return plot_track


def plot_main_object(k, ax, merge_frame, ego_trace):
    plot_mainobj = [None]
    if hasattr(merge_frame, 'mainobj_info'):
        main_obj, main_obj_info, track_obj_info = merge_frame.mainobj_info
        if main_obj_info is not None:
            main_obj_coord = np.array([main_obj_info.WldX, main_obj_info.WldY])
            main_obj_coord = trans_to_world_coords(main_obj_coord, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
            plot_mainobj = ax.plot(main_obj_coord[0], main_obj_coord[1], 'wo', markersize=10, alpha=0.5)
    return plot_mainobj

def plot_lanes(k, ax, ego_trace, reg_bonds):
    for bond_coords in reg_bonds:
        bond_coords = trans_to_world_coords(bond_coords, ego_trace[k, :2], ego_trace[k, 2] - np.pi/2)
        ax.plot(bond_coords[0], bond_coords[1])   
        
def post_plot(ax, center_point, map_scope, plot_camera_meas, plot_radar_meas, plot_ego, plot_lm, plot_cov, plot_stable_lm,
              plot_track, plot_mainobj):
    end_point = center_point
    ax.plot(end_point[0], end_point[1], 'ws', color=[0, 1, 1], markersize=3)
    ax.set_xlim([end_point[0] - map_scope, end_point[0] + map_scope])
    ax.set_ylim([end_point[1] - map_scope, end_point[1] + map_scope])
    ax.legend((plot_camera_meas[0], plot_radar_meas[0], plot_ego[0], plot_lm[0], plot_stable_lm[0], 
               plot_track[0], plot_cov, plot_mainobj[0]),
          ['camera meas.', 'radar meas.', 'ego trace', 'landmarks', 'stable landmarks', 'tracks',
           'gating cov', 'main obj.'],
          loc='upper right')
    ax.set_aspect(1)
            

def plot_cardinality(estimates):
    plt.figure()
    # PMB_card_pred = [for ele in track for step, track in enumerate(PMBMEstimates)]
    PMB_card_pred = []
    for step, tracks in enumerate(estimates):
        valid_track_num = 0
        for track in tracks:
            range_t = track['range'][-1] - track['range'][0] + 1
            if track['range'][-1] == step:
                valid_track_num += 1
        PMB_card_pred.append(valid_track_num)
    plt.plot(PMB_card_pred, 'y+')
    plt.legend(['GT', 'PMBM'])
    plt.grid()