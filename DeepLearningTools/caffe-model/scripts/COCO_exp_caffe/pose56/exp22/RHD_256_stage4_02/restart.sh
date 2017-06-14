#!/usr/bin/env sh
/home/maxiao/CoreLib/caffe_train/build/tools/caffe train --solver=pose_solver.prototxt --gpu=$1 --snapshot ./model/hand_iter_$2.solverstate 2>&1 | tee ./output.txt