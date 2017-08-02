#include "pch.h"
#include "nnet.h"

nnet::nnet(int n_input, int n_neurons_in, int n_output, int *_hidden_layers, int _n_hidden_layers, enum layer_type layT)
{
    n_hidden_layers = _n_hidden_layers;
   
    switch(layT) //initialize different layer type
    {
       case SIGMOID: 
            input_layer  = new layer_sigmoid();
	    output_layer = new layer_sigmoid();
            // hidden layer
    	    if(_hidden_layers && _n_hidden_layers)
   	    {
      	       hidden_layers = new layer*[_n_hidden_layers];
	       for(int i = 0; i < n_hidden_layers; ++i)
       	       {
                  hidden_layers[i] = new layer_sigmoid;
	       }
	    }
       case HYPERTAN: 
            input_layer  = new layer_tanh();
	    output_layer = new layer_tanh();
            // hidden layer
    	    if(_hidden_layers && _n_hidden_layers)
   	    {
      	       hidden_layers = new layer*[_n_hidden_layers];
	       for(int i = 0; i < n_hidden_layers; ++i)
       	       {
                  hidden_layers[i] = new layer_tanh;
	       }
	    }          
    }
   

    //input layer
    input_layer->create(n_input, n_neurons_in);
     
    // hidden layer
    if(_hidden_layers && _n_hidden_layers)
    {
       for(int i = 0; i < n_hidden_layers; ++i)
       {
          if(i == 0)
 	  {
            //first hidden layer take inputs from input layer       
       	    hidden_layers[i]->create(n_neurons_in, _hidden_layers[i]);
          }
          else
          {
            hidden_layers[i]->create(_hidden_layers[i-1], _hidden_layers[i]);
	  }
       }
       
       //last hidden layer dumps outputs to the output layer
       output_layer->create(_hidden_layers[n_hidden_layers-1], n_output);
    }

}


nnet::~nnet()
{
    if(hidden_layers)
    {
        for(int i=0; i < n_hidden_layers; i++)
        {
            delete hidden_layers[i];
        }

        delete [] hidden_layers;
    }

    delete input_layer;
    delete output_layer;
}


void bpnet_sst::propagate(const std::vector<double> input)
{
     input_layer->layerinput = input;
     input_layer->calculate();
     
     update(-1);//propagate the inputlayer out values to the next layer
     if(hidden_layers)
     {
        //Calculating hidden layers if any
        for(int i=0;i < n_hidden_layers; i++)
        {
            hidden_layers[i]->calculate();
            update(i);
        }
     }

    //calculating the final statge: the output layer
    output_layer->calculate(); 
}


void bpnet_sst::update(int layer_index)
{
    int i;
    if(layer_index==-1)
    {
        //dealing with the inputlayer here and propagating to the next layer
        for(i =0 ;i < input_layer->n_neuron; i++)
        {
            if(hidden_layers)//propagate to the first hidden layer
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
        for(i = 0;i < hidden_layers[layer_index]->n_neuron; i++)
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

double bpnet_sst::train(const std::vector<double> train_data, const std::vector<double> train_class, double learning_rate, double momentum)
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

     

     
     
  







