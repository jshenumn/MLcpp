#ifndef NNET_H
#define NNET_H

#include "layer.h"

enum layer_type{SIGMOID, HYPERTAN};

class nnet
{
   protected:
           layer*     input_layer;
           layer*     output_layer;
           layer**    hidden_layers;
           int        n_hidden_layers;
   
   public:
         nnet(int n_input,  int n_neurons_in,
              int n_output, int *_hidden_layers,
              int _n_hidden_layers, enum layer_type layT);
         virtual ~nnet();
         
         virtual int get_n_hidden_layers(){return n_hidden_layers;}
         virtual double train(const std::vector<double> train_data,
                              const std::vector<double> train_class,
                              double learning_rate, double momentum) = 0;
};

// child class, backpropagation foward feed nerual net
class bpnet_sst : public nnet
{
   public:
         bpnet_sst(int n_input,          int n_neurons_in,
                   int n_output,         int *_hidden_layers,
                   int _n_hidden_layers, enum layer_type layT)
       : nnet(n_input, n_neurons_in, n_output,
             _hidden_layers, _n_hidden_layers, layT){};
        ~bpnet_sst();

         void propagate(const std::vector<double> input);
         void update(int layer_index);  
         double train(const std::vector<double> train_data,
                      const std::vector<double> train_class,
                      double learning_rate, double momentum);
};


// TODO
// class cnn : public nnet    convolutional nerual network
// class rnn : public nnet    recurrent nerual network


#endif
