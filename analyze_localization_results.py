#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 15:43:00 2021

@author: mzins
"""

import numpy as np

import os
from ellcv.visu import generate_triaxe_pointcloud
from ellcv.io import write_ply
from ellcv.utils import pose_error
from seven_scenes_tools import SevenScenes_loader



a = 0.236649
a = (0.224995+0.227072)/2
b = 0.6611015

scale = a/b # from 7-Scene world scale to slam scale



loader1 = SevenScenes_loader("/media/mzins/DATA1/7-Scenes/chess/seq-01")
R0, t0 = loader1.get_Rt(0)
o0 = R0.T
p0 = -R0.T @ t0

#%%
pts = np.loadtxt("pointcloudd.txt", delimiter=",")
# pts /= scale

slam_poses_file = "build/out_poses_slam.txt"


data_poses_slam = np.loadtxt(slam_poses_file)
pose = data_poses_slam[0, 1:].reshape((3, 4))

# transform into the coord frame of the first camera (do nothing because it's the identity)
# o = pose[:3, :3]
# p = pose[:, 3].reshape((-1, 1)) 
# R = o.T
# t = -o.T @ p
# pts = (R @ pts.T + t).T

# rescale to real world
pts /= scale

# transform to seven-scenes world coord system
pts_w = pts
pts_w = (o0 @ pts.T + p0).T

write_ply("pointcloud_slam_world.ply", pts_w)

#%%


poses_file = "build/out_poses.txt"
data_poses_reloc = np.loadtxt(poses_file)

triaxes_pts = []
triaxes_cols = []
poses = [None] * data_poses_reloc.shape[0]
for i in range(data_poses_reloc.shape[0]):
    status = data_poses_reloc[i, 0]
    if np.sum(np.abs(data_poses_reloc[i, 1:])) < 1:
        print("skip")
        continue
    pose = data_poses_reloc[i, 1:].reshape((3, 4))
    o = pose[:3, :3]
    p = pose[:, 3].reshape((-1, 1))
    
    p /= scale
    
    o = o0 @ o
    p = o0 @ p + p0

    poses[i] = [o, p]
    
    pts, cols = generate_triaxe_pointcloud([o, p], 0.1, 50)
    triaxes_cols.append(cols)
    triaxes_pts.append(pts)

triaxes_cols = np.vstack(triaxes_cols)
triaxes_pts = np.vstack(triaxes_pts)

write_ply("poses.ply", triaxes_pts, triaxes_cols)



#%%
loader = SevenScenes_loader("/media/mzins/DATA1/7-Scenes/chess/seq-02")

triaxes_gt_pts = []
triaxes_gt_cols = []
gt_poses = [None] * loader.nb_images
for i in range(loader.nb_images):
    R, t = loader.get_Rt(i)
    o = R.T
    p = -R.T @ t

    gt_poses[i] = [o, p]

    pts, cols = generate_triaxe_pointcloud([o, p], 0.2, 50)
    triaxes_gt_cols.append(cols)
    triaxes_gt_pts.append(pts)

triaxes_gt_cols = np.vstack(triaxes_gt_cols)
triaxes_gt_pts = np.vstack(triaxes_gt_pts)

write_ply("poses_gt.ply", triaxes_gt_pts, triaxes_gt_cols)

#%%
pos_errors = [-1]*len(poses)
rot_errors = [-1]*len(poses)
for i, (p, gt_p) in enumerate(zip(poses, gt_poses)):
    if p is not None:
        rot_error, pos_error = pose_error(p, gt_p)
        pos_errors[i] = pos_error
        rot_errors[i] = np.rad2deg(rot_error)

print("median pos error = ", np.median(pos_errors[88:]))
print("median rots error = ", np.median(rot_errors[88:]))
print("mean pos error = ", np.mean(pos_errors[88:]))
print("mean rot error = ", np.mean(rot_errors[88:]))

np.savetxt("POS_ERRORS.txt", pos_errors)
np.savetxt("ROT_ERRORS.txt", rot_errors)

    
