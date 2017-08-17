#ifndef NEURON_H
#define NEURON_H


struct neuron
{
   std::vector<double> weights;
   std::vector<double> deltas;
   double output;
   double bias;
   double w_bias;
   bool active;

   neuron();
  ~neuron(){};
   void create(int n_input);
   void activate();
   void deactivate();
};

#endif
