#ifndef ARCHITECTURE_H_INCLUDED
#define ARCHITECTURE_H_INCLUDED

enum layer_type {SIGMOID, HYPERTAN, SOFTMAX};

struct architect
{
       int n_input;
       int n_neurons_in;
       int n_output;
       int n_hidden_layers;



       std::vector<int> hidden_layers;

       architect(int _n_input, int _n_neurons_in, int _n_output, int _n_hidden_layers);
      ~architect(){}



};


#endif // ARCHITECTURE_H_INCLUDED
