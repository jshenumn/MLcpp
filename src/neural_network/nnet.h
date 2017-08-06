#ifndef NNET_H
#define NNET_H

#include "layer.h"


class bpnet
{
protected:
    std::unique_ptr<layer>   input_layer;
    std::unique_ptr<layer>   output_layer;
    std::vector<std::unique_ptr<layer> >  hidden_layers;

    int n_hidden_layers;
    int n_input;
    int n_neurons_in;
    int n_output;

    std::vector<int> hidden_layer_layout;

public:
    bpnet(int _n_input, int _n_neurons_in, int _n_output, std::vector<int> _hidden_layers, int _n_hidden_layers);

    virtual ~bpnet(){};

    virtual void create(){}
    virtual int get_n_hidden_layers(){}
    virtual void propagate(const std::vector<double>& input);
    virtual void update(int layer_index);
    virtual double train(const std::vector<double>& train_data,
                         const std::vector<double>& train_class,
                         double learning_rate, double momentum){}
    virtual void getOutput(std::vector<double>& input, std::vector<double>& output);
};

// child class, backpropagation foward feed nerual net
class bpnet_MSE_sigmoid : public bpnet
{
public:
    bpnet_MSE_sigmoid(int n_input, int n_neurons_in,int n_output, std::vector<int> _hidden_layers,int _n_hidden_layers):bpnet(n_input, n_neurons_in, n_output,
                          _hidden_layers, _n_hidden_layers) {}

   ~bpnet_MSE_sigmoid(){}

    void create();
    void propagate(const std::vector<double>& input);
    void update(int layer_index);
    double train(const std::vector<double>& train_data,
                 const std::vector<double>& train_class,
                 double learning_rate, double momentum);
    int get_n_hidden_layers()
    {
        return n_hidden_layers;
    }
    void getOutput(std::vector<double>& input, std::vector<double>& output);

};

// child class, backpropagation foward feed nerual net
class bpnet_crossentropy_softmax : public bpnet
{
public:
    bpnet_crossentropy_softmax(int n_input, int n_neurons_in, int n_output, std::vector<int> _hidden_layers, int _n_hidden_layers):bpnet(n_input, n_neurons_in, n_output,
                               _hidden_layers, _n_hidden_layers){}

   ~bpnet_crossentropy_softmax(){}

    void create();
    void propagate(const std::vector<double>& input);
    void update(int layer_index);
    double train(const std::vector<double>& train_data,
                 const std::vector<double>& train_class,
                 double learning_rate, double momentum);
    int get_n_hidden_layers()
    {
        return n_hidden_layers;
    }
    void getOutput(std::vector<double>& input, std::vector<double>& output);

};




// TODO
// class cnn : public nnet    convolutional nerual network
// class rnn : public nnet    recurrent nerual network


#endif
