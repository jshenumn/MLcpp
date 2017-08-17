#include "pch.h"
#include "nnet.h"



//============ BASE CLASS METHODS =======================
bpnet::bpnet(int _n_input, int _n_neurons_in, int _n_output, std::vector<int> _hidden_layers, int _n_hidden_layers)
{
    hidden_layer_layout = _hidden_layers;

    n_input = _n_input;
    n_neurons_in = _n_neurons_in;
    n_output = _n_output;
    n_hidden_layers = _n_hidden_layers;
}

void bpnet::getOutput(std::vector<double>& input,std::vector<double>& opt)
{
    opt.resize(output_layer->n_neuron);

    propagate(input);

    for(int i = 0; i < output_layer->n_neuron; i++)
    {
        opt[i] = output_layer->neurons[i].output;
    }
}




void bpnet::propagate(const std::vector<double>& input)
{
    input_layer->layerinput = input;
    input_layer->calculate();

    update(-1);//propagate the inputlayer out values to the next layer

    if(!hidden_layers.empty())
    {
        //Calculating hidden layers if any
        for(int i=0; i < n_hidden_layers; i++)
        {
            hidden_layers[i]->calculate();
            update(i);
        }
    }

    //calculating the final statge: the output layer
    output_layer->calculate();
}

void bpnet::update(int layer_index)
{
    int i;
    if(layer_index == -1)
    {
        //dealing with the inputlayer here and propagating to the next layer
        for(i =0 ; i < input_layer->n_neuron; i++)
        {
            if(!hidden_layers.empty())//propagate to the first hidden layer
            {
                hidden_layers[0]->layerinput[i] = input_layer->neurons[i].output;
            }
            else //propagate directly to the output layer
            {
                output_layer->layerinput[i] = input_layer->neurons[i].output;
            }
        }

    }
    else
    {
        for(i = 0; i < hidden_layers[layer_index]->n_neuron; i++)
        {
            //not the last hidden layer
            if(layer_index < n_hidden_layers - 1)
            {
                hidden_layers[layer_index + 1] -> layerinput[i]
                    = hidden_layers[layer_index]->neurons[i].output;
            }
            else
            {
                output_layer->layerinput[i] = hidden_layers[layer_index]->neurons[i].output;
            }
        }
    }
}










// =========   MSE loss function with sigmoid activation =========//
void bpnet_MSE_sigmoid::create()
{
    std::cout << "================= The architecture for the network =================\n";
    std::cout << "Input  layer size:" <<  n_neurons_in << "\n";
    std::cout << "Hidden layer size:" <<  n_hidden_layers << "\n";
    std::cout << "Output layer size:" <<  n_output << "\n";

    input_layer  = std::make_unique<layer_sigmoid>();
    output_layer = std::make_unique<layer_sigmoid>();

    if(!hidden_layer_layout.empty() && n_hidden_layers > 0)
    {
        hidden_layers.resize(n_hidden_layers);
        for(int i = 0; i < n_hidden_layers; ++i)
        {
            hidden_layers[i] = std::make_unique<layer_sigmoid>();
        }
    }

    //input layer
    input_layer->create(n_input, n_neurons_in);

    // hidden layer
    if(!hidden_layer_layout.empty() && n_hidden_layers > 0)
    {
        for(int i = 0; i < n_hidden_layers; ++i)
        {
            if(i == 0)
            {
                //first hidden layer take inputs from input layer
                hidden_layers[i]->create(n_neurons_in, hidden_layer_layout[i]);
            }
            else
            {
                hidden_layers[i]->create(hidden_layer_layout[i-1], hidden_layer_layout[i]);
            }
        }

        //last hidden layer dumps outputs to the output layer
        output_layer->create(hidden_layer_layout[n_hidden_layers-1], n_output);
    }
    else
    {
        output_layer->create(n_neurons_in,n_output);
    }

}


void bpnet_MSE_sigmoid::propagate(const std::vector<double>& input)
{
    bpnet::propagate(input);
}


void bpnet_MSE_sigmoid::update(int layer_index)
{
    bpnet::update(layer_index);
}

double bpnet_MSE_sigmoid::train(const std::vector<double>& train_data, const std::vector<double>& train_class, double learning_rate, double momentum)
{
    double loss = 0.0;
    double loc_err = 0.0;
    double sum = 0.0, csum = 0.0;
    double delta = 0.0, udelta = 0.0;
    double output = 0.0;

    // propagate forward the input data
    propagate(train_data);

    // back propagation from the output layer

    for(int i = 0; i < output_layer->n_neuron; i++)
    {
        output = output_layer->neurons[i].output;

        //local error terms w.r.t sst loss function
        loc_err = (train_class[i] - output) * output * (1.0 - output); // not the update
        //SST
        loss += (train_class[i] - output) * (train_class[i] - output);

        //update weights
        for(int j = 0; j < output_layer->n_input; j++)
        {
            //get current delta value
            delta = output_layer->neurons[i].deltas[j]; //d_ij    j-th neuron from previous layer to the i-th neuron of current layer
            udelta = learning_rate * loc_err * output_layer->layerinput[j]
                     + delta * momentum;  //update rule with momentum
            //update weights
            output_layer->neurons[i].weights[j] += udelta;
            //record deltas
            output_layer->neurons[i].deltas[j]  = udelta;

            //also need to track sum of weights dot errors
            sum += output_layer->neurons[i].weights[j] * loc_err;
        }
        // calculate the weight bias
        output_layer->neurons[i].w_bias += learning_rate * loc_err * output_layer->neurons[i].bias;

    }
    // now proceed to the hidden layers
    for(int i = n_hidden_layers - 1; i >= 0; i--)
    {
        for(int j = 0; j < hidden_layers[i]->n_neuron; j++ )
        {
            output  = hidden_layers[i]->neurons[j].output;
            // local error by formula where sum is from the next layer
            loc_err = output * (1.0 - output) * sum;
            // update neuron weights
            for(int k = 0; k < hidden_layers[i]->n_input; k++)
            {
                delta = hidden_layers[i]->neurons[j].deltas[k];
                udelta = learning_rate * loc_err * hidden_layers[i]->layerinput[k] + delta * momentum;
                hidden_layers[i]->neurons[j].weights[k] += udelta;
                hidden_layers[i]->neurons[j].deltas[k]  =  udelta;

                csum += hidden_layers[i]->neurons[j].weights[k] * loc_err;
            }

            hidden_layers[i]->neurons[j].w_bias += learning_rate * loc_err * hidden_layers[i]->neurons[j].bias;
        }
        sum = csum;
        csum = 0.0;
    }

    // go to the first layer
    for(int i = 0; i < input_layer->n_neuron; i++)
    {
        output = input_layer->neurons[i].output;

        //local error terms w.r.t sst loss function
        loc_err = sum * output * (1.0 - output);


        //update weights
        for(int j = 0; j < input_layer->n_input; j++)
        {
            //get current delta value
            delta = input_layer->neurons[i].deltas[j]; //d_ij    j-th neuron from previous layer to the i-th neuron of current layer
            udelta = learning_rate * loc_err * input_layer->layerinput[j]
                     + delta * momentum;  //update rule with momentum
            //update weights
            input_layer->neurons[i].weights[j] += udelta;
            //record deltas
            input_layer->neurons[i].deltas[j]  = udelta;

        }
        // calculate the weight bias
        input_layer->neurons[i].w_bias += learning_rate * loc_err * input_layer->neurons[i].bias;

    }

    return loss/2.0;
}

void bpnet_MSE_sigmoid::getOutput(std::vector<double>& input,std::vector<double>& opt)
{
    bpnet::getOutput(input, opt);
}


//==========  Cross entropy loss function with softmax activation ========//
void bpnet_CrossEntropy_softmax::create()
{
    std::cout << "================= The architecture for the network =================\n";
    std::cout << "Input  layer size:" <<  n_neurons_in << "\n";
    std::cout << "Hidden layer size:" <<  n_hidden_layers << "\n";
    std::cout << "Output layer size:" <<  n_output << "\n";

    input_layer  = std::make_unique<layer_sigmoid>();
    output_layer = std::make_unique<layer_softmax>();

    if(!hidden_layer_layout.empty() && n_hidden_layers > 0)
    {
        hidden_layers.resize(n_hidden_layers);
        for(int i = 0; i < n_hidden_layers; ++i)
        {
            hidden_layers[i] = std::make_unique<layer_sigmoid>();
        }
    }

    //input layer
    input_layer->create(n_input, n_neurons_in);

    // hidden layer
    if(!hidden_layer_layout.empty() && n_hidden_layers > 0)
    {
        for(int i = 0; i < n_hidden_layers; ++i)
        {
            if(i == 0)
            {
                //first hidden layer take inputs from input layer
                hidden_layers[i]->create(n_neurons_in, hidden_layer_layout[i]);
            }
            else
            {
                hidden_layers[i]->create(hidden_layer_layout[i-1], hidden_layer_layout[i]);
            }
        }

        //last hidden layer dumps outputs to the output layer
        output_layer->create(hidden_layer_layout[n_hidden_layers-1], n_output);
    }
    else
    {
        output_layer->create(n_neurons_in,n_output);
    }

}

void bpnet_CrossEntropy_softmax::propagate(const std::vector<double>& input)
{
    bpnet::propagate(input);
}

void bpnet_CrossEntropy_softmax::update(int layer_index)
{
    bpnet::update(layer_index);
}

double bpnet_CrossEntropy_softmax::train(const std::vector<double>& train_data, const std::vector<double>& train_class, double learning_rate, double momentum)
{
    double loss = 0.0;
    double loc_err = 0.0;
    double sum = 0.0, csum = 0.0;
    double delta = 0.0, udelta = 0.0;
    double output = 0.0;

    // propagate forward the input data
    propagate(train_data);

    // back propagation from the output layer
    for(int i = 0; i < output_layer->n_neuron; i++)
    {
        output = output_layer->neurons[i].output;

        //local error terms w.r.t cross entropy loss function
        loc_err = (output - train_class[i]); // not the update
        //std::cout << i << "-th neuron" << output << "," << train_class[i] << std::endl;
        //Cross entropy
        loss -= train_class[i] * log(output);

        //update weights
        for(int j = 0; j < output_layer->n_input; j++)
        {
            //get current delta value
            delta = output_layer->neurons[i].deltas[j]; //d_ij    j-th neuron from previous layer to the i-th neuron of current layer
            udelta = learning_rate * loc_err * output_layer->layerinput[j] + delta * momentum;  //update rule with momentum
            //update weights
            output_layer->neurons[i].weights[j] -= udelta;
            //record deltas
            output_layer->neurons[i].deltas[j]  = udelta;
            //also need to track sum of weights dot errors
            sum += output_layer->neurons[i].weights[j] * loc_err;
        }
        // calculate the weight bias
        output_layer->neurons[i].w_bias -= learning_rate * loc_err * output_layer->neurons[i].bias;

    }
    // now proceed to the hidden layers
    for(int i = n_hidden_layers - 1; i >= 0; i--)
    {
        for(int j = 0; j < hidden_layers[i]->n_neuron; j++ )
        {
            output  = hidden_layers[i]->neurons[j].output;
            // local error by formula where sum is from the next layer
            loc_err = output * (1.0 - output) * sum;
            // update neuron weights
            for(int k = 0; k < hidden_layers[i]->n_input; k++)
            {
                delta = hidden_layers[i]->neurons[j].deltas[k];
                udelta = learning_rate * loc_err * hidden_layers[i]->layerinput[k] + delta * momentum;
                hidden_layers[i]->neurons[j].weights[k] -= udelta;
                hidden_layers[i]->neurons[j].deltas[k]  =  udelta;

                csum += hidden_layers[i]->neurons[j].weights[k] * loc_err;
            }

            hidden_layers[i]->neurons[j].w_bias -= learning_rate * loc_err * hidden_layers[i]->neurons[j].bias;
        }
        sum = csum;
        csum = 0.0;
    }

    // go to the first layer
    for(int i = 0; i < input_layer->n_neuron; i++)
    {
        output = input_layer->neurons[i].output;

        //local error terms w.r.t sst loss function
        loc_err = sum * output * (1.0 - output);

        //update weights
        for(int j = 0; j < input_layer->n_input; j++)
        {
            //get current delta value
            delta = input_layer->neurons[i].deltas[j]; //d_ij    j-th neuron from previous layer to the i-th neuron of current layer
            udelta = learning_rate * loc_err * input_layer->layerinput[j] + delta * momentum;  //update rule with momentum
            //update weights
            input_layer->neurons[i].weights[j] -= udelta;
            //record deltas
            input_layer->neurons[i].deltas[j]  = udelta;

        }
        // calculate the weight bias
        input_layer->neurons[i].w_bias -= learning_rate * loc_err * input_layer->neurons[i].bias;

    }

    return loss;
}

void bpnet_CrossEntropy_softmax::getOutput(std::vector<double>& input,std::vector<double>& opt)
{
    bpnet::getOutput(input, opt);
}



