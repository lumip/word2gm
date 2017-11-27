#!/bin/bash
## see https://www.tensorflow.org/extend/adding_an_op#build_the_op_library for instructions

set -x # be verbose; echo commands

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared word2vec_ops.cc word2vec_kernels.cc -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -o word2vec_ops.so -D_GLIBCXX_USE_CXX11_ABI=0
