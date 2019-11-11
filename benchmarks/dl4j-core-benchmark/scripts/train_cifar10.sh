#!/usr/bin/env sh
# change to -gpu all for multi-gpu

while true; do
    read -p "Quick, Batch, Full?" yn
    case $yn in
        'Quick' ) time $HOME/caffe/build/tools/caffe train --solver=dl4j-core-benchmark/src/main/java/org/deeplearning4j/Cifar10/caffe/caffe_cifar10_full_quick_solver.prototxt -gpu all;;
        'Batch' ) time $HOME/caffe/build/tools/caffe train --solver=dl4j-core-benchmark/src/main/java/org/deeplearning4j/Cifar10/caffe/caffe_cifar10_full_sigmoid_solver_bn.prototxt -gpu all;;
        'Full' ) time $HOME/caffe/build/tools/caffe train --solver=dl4j-core-benchmark/src/main/java/org/deeplearning4j/Cifar10/caffe/caffe_cifar10_full_solver_lr2.prototxt -gpu all;;
        *) echo "Invalid response";;
    esac
done

time python dl4j-core-benchmark/src/main/java/org/deeplearning4j/Cifar10/tensorflow/cifar10_multi_gpu_train.py --num_gpus 4
time th dl4j-core-benchmark/src/main/java/org/deeplearning4j/Cifar10/torch/torch-cifar10.lua


time java -cp dl4j-core-benchmark/dl4j-core-benchmark.jar org.deeplearning4j.Cifar10.dl4j.Dl4j_Cifar10
