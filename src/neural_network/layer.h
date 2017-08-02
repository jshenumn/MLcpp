#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
 
struct layer
{
     std::vector<neuron> neurons;
     std::vector<double> layerinput;
     int n_neuron;
     int n_input;
     
     layer();
     virtual ~layer(){};    
    
     void create(int _n_input, int _n_neuron);
     virtual void calculate() = 0;
};

struct layer_sigmoid : public layer
{
     layer_sigmoid():layer(){};
    ~layer_sigmoid(){};
      
     void calculate();
};

struct layer_tanh : public layer
{
     layer_tanh():layer(){};
    ~layer_tanh(){};
     
     void calculate();
};


#endif
