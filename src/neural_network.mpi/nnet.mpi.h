// This file defines the neuron network using mpi
// unlike the single node bpnet we need to update gradients
// based on the results from all processors which requires a different
// work-flow

#include "pch.h"
#include "../neural_network/neuron.h"
#include "../neural_network/layer.h"
#include "../neural_network/nnet.h"


void mpi_bpnet(std::string path_to_data)
{
    // Main function to train a bpnet



}
