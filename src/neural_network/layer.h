#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
/*! 
  \brief the layer structure which consists of neurons and an activation function. 
*/
struct layer
{
     std::vector<neuron> neurons;
     std::vector<double> layerinput;
     int n_neuron;
     int n_input;
     /** constructor
     */
     layer();
     /**
         destructor
     */
     virtual ~layer(){};
     /** create a layer
       \param _n_input the number of inputs
       \param _n_neuron the number of neurons in the layer 
     */
     void create(int _n_input, int _n_neuron);
     /** pure virtual function to calculate layer output
         based on the activation function
     */
     virtual void calculate() = 0;
};

/*! 
  \brief The layer with a sigmoid activation
*/
struct layer_sigmoid : public layer
{
     layer_sigmoid():layer(){}
    ~layer_sigmoid(){}

     void calculate();
};
/*! 
  \brief The layer with a tanh activation
*/
struct layer_tanh : public layer
{
     layer_tanh():layer(){}
    ~layer_tanh(){}

     void calculate();
};
/*! 
  \brief The layer with a softmax activation, which is popular for multi-class cases. 
*/
struct layer_softmax : public layer
{
     layer_softmax() : layer(){}
    ~layer_softmax(){}

     void calculate();
};

#endif
