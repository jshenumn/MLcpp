#ifndef NNET_H
#define NNET_H

#include "layer.h"
#include <boost/log/trivial.hpp>


// Back propagation neural network
/*!
\brief A backpropagation neural network class.
   The back-propagation network has the architecture of
   three components propagate, update and train
*/
class bpnet
{
protected:
    std::unique_ptr<layer>   input_layer; /**< input layer */
    std::unique_ptr<layer>   output_layer;  /**< output layer */
    std::vector<std::unique_ptr<layer> >  hidden_layers; /**< hidden layers holder*/

    int n_hidden_layers;
    int n_input;
    int n_neurons_in;
    int n_output;

    std::vector<int> hidden_layer_layout;

public:
    /*! bpnet constructor
      \param _n_input number of input variables.
      \param _n_neurons_in number of neurons in input layer.
      \param _n_ouput number of output
      \param _hidden_layers hidden_layers size vector
      \param _n_hidden_layers number of hidden layers
    */
    bpnet(int _n_input, int _n_neurons_in, int _n_output, std::vector<int> _hidden_layers, int _n_hidden_layers);

    /*! bpnet destructor */
    virtual ~bpnet(){};

    /*! a virtual function to create neural network */
    virtual void create(){}
    /*! a function to get number of hidden layers
      \return number of hidden layers
    */
    virtual int get_n_hidden_layers(){}
    /*! a function to perform forward feeding of data
      \param input the input data vector
    */
    void propagate(const std::vector<double>& input);
    /*! a function to update
      \param layer_index the layer index
    */
    void update(int layer_index);
    /*! a virtual function to train data
      \param train_data training data
      \param train_class training data class label
      \param learning_rate learning rate
      \param momentum momentum or damping factor
      \return loss value
    */
    virtual double train(const std::vector<double>& train_data,
                         const std::vector<double>& train_class,
                         double learning_rate, double momentum){}
    /*! a virtual function to get output class labels
      \param input  input data
      \param output output class label
    */
    virtual void get_output(std::vector<double>& input, std::vector<double>& output);

    virtual void extract_weights(std::vector<double>& weights_flattened, int& dim);
    virtual void broadcast_weights(std::vector<double>& weights_flattened, int& dim);


};

/*!
  \brief child class, backpropagation foward feed nerual net using mean squared error loss
   and sigmoid activation. Usually a good choice for binary classification problems.
*/
class bpnet_MSE_sigmoid : public bpnet
{
public:
    /*! constructor initialize from base class
      \param _n_input number of input variables.
      \param _n_neurons_in number of neurons in input layer.
      \param _n_ouput number of output
      \param _hidden_layers hidden_layers size vector
      \param _n_hidden_layers number of hidden layers
    */
    bpnet_MSE_sigmoid(int n_input, int n_neurons_in,int n_output, std::vector<int> _hidden_layers,int _n_hidden_layers):bpnet(n_input, n_neurons_in, n_output,
                          _hidden_layers, _n_hidden_layers) {}
    /*! destructor */
   ~bpnet_MSE_sigmoid(){}

    /*! create network */
    void create();
    /*! forward feeding of data
      \param input the input data vector
    */
    void propagate(const std::vector<double>& input);
    /*! update a layer
       \param layer_index the layer index
    */
    void update(int layer_index);
    /*! training function
      \param train_data training data
      \param train_class training data class label
      \param learning_rate learning rate
      \param momentum momentum or damping factor
      \return loss value
    */
    double train(const std::vector<double>& train_data,
                 const std::vector<double>& train_class,
                 double learning_rate, double momentum);
    /*! get number of hidden layers
      \return number of hidden layers
    */
    int get_n_hidden_layers()
    {
        return n_hidden_layers;
    }
    /*! get output class labels
      \param input  input data
      \param output output class label
    */
    void get_output(std::vector<double>& input, std::vector<double>& output);

    void extract_weights(std::vector<double>& weights_flattened, int& dim);
    void broadcast_weights(std::vector<double>& weights_flattened, int& dim);

};

/*!
  \brief child class, backpropagation foward feed nerual net using croos entropy loss
   and softmax activation. Usually a good choice for multi-classification problems.
*/
class bpnet_CrossEntropy_softmax : public bpnet
{
public:
    /*! constructor initialize from base class
      \param _n_input number of input variables.
      \param _n_neurons_in number of neurons in input layer.
      \param _n_ouput number of output
      \param _hidden_layers hidden_layers size vector
      \param _n_hidden_layers number of hidden layers
    */
    bpnet_CrossEntropy_softmax(int n_input, int n_neurons_in, int n_output, std::vector<int> _hidden_layers, int _n_hidden_layers):bpnet(n_input, n_neurons_in, n_output,
                               _hidden_layers, _n_hidden_layers){}
    /*!
        default destructor
    */
   ~bpnet_CrossEntropy_softmax(){}

    /*! create network */
    void create();
    /*! forward feeding of data
      \param input the input data vector
    */
    void propagate(const std::vector<double>& input);
    /*! update a layer
       \param layer_index the layer index
    */
    void update(int layer_index);
    /*! training function
      \param train_data training data
      \param train_class training data class label
      \param learning_rate learning rate
      \param momentum momentum or damping factor
      \return loss value
    */
    double train(const std::vector<double>& train_data,
                 const std::vector<double>& train_class,
                 double learning_rate, double momentum);
    /*! get number of hidden layers
      \return number of hidden layers
    */
    int get_n_hidden_layers()
    {
        return n_hidden_layers;
    }
     /*! get output class labels
      \param input  input data
      \param output output class label
    */
    void get_output(std::vector<double>& input, std::vector<double>& output);

    void extract_weights(std::vector<double>& weights_flattened, int& dim);
    void broadcast_weights(std::vector<double>& weights_flattened, int& dim);

};



// TODO
// class cnn : public nnet    convolutional nerual network
// class rnn : public nnet    recurrent nerual network


#endif
