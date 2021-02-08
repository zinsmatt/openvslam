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
from scipy.spatial.transform.rotation import Rotation as Rot


invalid_value_pos = -1.05
invalid_value_rot = -1.05

name = "_seq_02"
name = "_seq_02_rgbd"
name = "_seq_02_rgbd_01_04_06"
# name = "_seq_03"
# name = "_seq_05"
# name = "_seq_01_map_200"
input_pose_file = "/home/mzins/dev/openvslam/build/out_poses" + name + ".txt"
output_pos_errors_file = "POS_ERRORS" + name + ".txt"
output_rot_errors_file = "ROT_ERRORS" + name + ".txt"

sequence_folder = "/media/mzins/DATA1/7-Scenes/chess/seq-02"
# sequence_folder = "/media/mzins/DATA1/7-Scenes/chess/seq-03"
# sequence_folder = "/media/mzins/DATA1/7-Scenes/chess/seq-05"
# sequence_folder = "/media/mzins/DATA1/7-Scenes/chess/seq-01"

a = 0.236649
a = (0.224995+0.227072)/2
b = 0.6611015



# scale = 1.1*a/b # from 7-Scene world scale to slam scale (if keypoints 1000 and 1.2)
scale = a/b # from 7-Scene world scale to slam scale (if keypoints 2000 and 1.5)
# scale = 1.005*a/b # for seq-01 0-199
scale = 1

# Load the sequence corresponding to slam to transform the poses into real world coordinates
loader1 = SevenScenes_loader("/media/mzins/DATA1/7-Scenes/chess/seq-01")
R0, t0 = loader1.get_Rt(0)
o0 = R0.T
p0 = -R0.T @ t0

# #%% Transform the map point cloud into the 7-Scene world coordinates
# pts = np.loadtxt("pointcloud_slam_01.txt", delimiter=",")
# pts = np.loadtxt("pointcloud_slam_01_200.txt", delimiter=",")
# pts = np.loadtxt("pointcloud_slam_01_rgbd.txt", delimiter=",")
pts = np.loadtxt("pointcloud_slam_01_04_06_rgbd.txt", delimiter=",")

# pts = (rr @ pts.T + tt).T

# rescale to real world
pts /= scale
# transform to seven-scenes world coord system
pts_w = pts
pts_w = (o0 @ pts.T + p0).T
# write_ply("pointcloud_slam_01_200_world.ply", pts_w)
# write_ply("pointcloud_slam_01_rgbd.ply", pts_w)
write_ply("pointcloud_slam_01_04_06_rgbd.ply", pts_w)

#%%


# R fine was determined manually from the point cloud
# it helps to get a fine alignment of the slam map with the ground truth map
# of 7-Scenes
R_fine = Rot.from_euler("y", 2, degrees=True).as_matrix()
t_fine = np.array([0.0, 0.0, -0.04]).reshape((-1, 1))

data_poses_reloc = np.loadtxt(input_pose_file)
triaxes_pts = []
triaxes_cols = []
poses = [None] * data_poses_reloc.shape[0]
for i in range(data_poses_reloc.shape[0]):
    status = data_poses_reloc[i, 0]
    if status != 1 or np.sum(np.abs(data_poses_reloc[i, 1:])) < 1:
        print("Invalid pose: frame", i)
        continue
    pose = data_poses_reloc[i, 1:].reshape((3, 4))
    o = pose[:3, :3]
    p = pose[:, 3].reshape((-1, 1))
    
    p /= scale
    
    o = o0 @ o
    p = o0 @ p + p0

    o = R_fine @ o
    p = R_fine @ p + t_fine

    poses[i] = [o, p]

    pts, cols = generate_triaxe_pointcloud([o, p], 0.1, 50)
    triaxes_cols.append(cols)
    triaxes_pts.append(pts)

triaxes_cols = np.vstack(triaxes_cols)
triaxes_pts = np.vstack(triaxes_pts)
write_ply("poses.ply", triaxes_pts, triaxes_cols)



#%%
loader = SevenScenes_loader(sequence_folder)

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
pos_errors = np.array([invalid_value_pos]*len(poses))
rot_errors = np.array([invalid_value_rot]*len(poses))
for i, (p, gt_p) in enumerate(zip(poses, gt_poses)):
    if p is not None:
        rot_error, pos_error = pose_error(p, gt_p)
        print(pos_error)
        pos_errors[i] = pos_error
        rot_errors[i] = np.rad2deg(rot_error)

goods = np.where(pos_errors >= 0)[0]
print("median pos error = ", np.median(pos_errors[goods]))
print("median rots error = ", np.median(rot_errors[goods]))
print("mean pos error = ", np.mean(pos_errors[goods]))
print("mean rot error = ", np.mean(rot_errors[goods]))

np.savetxt(output_pos_errors_file, pos_errors)
np.savetxt(output_rot_errors_file, rot_errors)
