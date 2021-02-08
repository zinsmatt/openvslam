#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:55:10 2021

@author: mzins
"""
import os

sequences = ["seq-01", "seq-04", "seq-06"]


lines = []
lines_depth = []
count = 0
for seq in sequences:
    for i in range(1000):
        name = seq + "/frame-%06d.color.png" % i
        lines.append(str(count) + " " + name + "\n")
        name = seq + "/frame-%06d.depth.png" % i
        lines_depth.append(str(count) + " " + name + "\n")
        count += 1
        

with open("rgb.txt", "w") as fout:
    fout.writelines(lines)
with open("depth.txt", "w") as fout:
    fout.writelines(lines_depth)
    