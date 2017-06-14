#!/usr/bin/env sh
/home/maxiao/CoreLib/caffe_train/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 --weights ./model/hand_iter_$2.caffemodel 2>&1 | tee ./output.txt