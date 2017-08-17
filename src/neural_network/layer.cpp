#include "pch.h"
#include "math.h"
#include "layer.h"
#include <iostream>


layer::layer():n_neuron(0),n_input(0)
{
}


void layer::create(int _n_input, int _n_neuron)
{
    neurons.resize(_n_neuron);
    n_input = _n_input;
    layerinput.resize(_n_input);
    n_neuron = _n_neuron;

    for(int i = 0; i< n_neuron; i++)
    {
        neurons[i].create(_n_input);
    }
}


void layer_sigmoid::calculate()
{
    for(int i = 0; i < n_neuron; ++i)
    {
       double sum = 0.0;
       for(int j = 0; j < n_input; ++j)
       {
          sum += neurons[i].weights[j] * layerinput[j];
       }
       sum += neurons[i].bias * neurons[i].w_bias;

       //sigmoid activation
       neurons[i].output = 1.0/(1.0 + exp(-sum));
    }
}


void layer_tanh::calculate()
{
    for(int i = 0; i < n_neuron; ++i)
    {
       double sum = 0.0;
       for(int j = 0; j < n_input; ++j)
       {
          sum += neurons[i].weights[j] * layerinput[j];
       }
       sum += neurons[i].bias * neurons[i].w_bias;

       //tanh activation
       neurons[i].output = tanh(sum);
    }

}

void layer_softmax::calculate()
{
    double tot = 0.0;
    double mmax = -10000;
    for(int i = 0; i < n_neuron; ++i)
    {
       double sum = 0.0;
       for(int j = 0; j < n_input; ++j)
       {
          sum += neurons[i].weights[j] * layerinput[j];
       }
       sum += neurons[i].bias * neurons[i].w_bias;

       //softmax activation
       if(sum > mmax) mmax = sum;
    }

    for(int i = 0; i < n_neuron; ++i)
    {
       double sum = 0.0;
       for(int j = 0; j < n_input; ++j)
       {
          sum += neurons[i].weights[j] * layerinput[j];
       }
       sum += neurons[i].bias * neurons[i].w_bias;

       //softmax activation
       neurons[i].output = exp(sum-mmax);
       tot += exp(sum-mmax);
    }

    for(int i = 0; i < n_neuron; ++i)
    {
       neurons[i].output /= tot;
    }
}

