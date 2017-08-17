# CMake generated Testfile for 
# Source directory: /home/jiguangshen/HPC_MachineLearning/MLcpp/src
# Build directory: /home/jiguangshen/HPC_MachineLearning/MLcpp/src
# 
# This file includes the relevent testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
ADD_TEST(neuron_network_tests "/home/jiguangshen/HPC_MachineLearning/MLcpp/bin/neural_net_unittest" "WORKING_DIRECTORY" "/home/jiguangshen/HPC_MachineLearning/MLcpp/src/../bin")
SUBDIRS(neural_network)
SUBDIRS(python_wrapper)
SUBDIRS(neural_network.unittest)
