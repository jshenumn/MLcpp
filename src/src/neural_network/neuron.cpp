#include "pch.h"
#include "neuron.h"
#include "boost/random.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/variate_generator.hpp"
typedef boost::minstd_rand base_generator_type;


neuron::neuron():weights(0),deltas(0),output(0),bias(0),w_bias(0)
{
}


neuron::neuron(int n_input)
{
   weights.resize(n_input);
   deltas.resize(n_input);
   output = 0.0;
   bias   = 1.0;
   active = TRUE;
   
   //random gen a vector
   base_generator_type generator(42u);
   boost::uniform_real<> uni_dist(-1,1);
   boost::variate_generator<base_generator_type&, boost::uniform_real<> > uni(generator, uni_dist);
   for(int i = 0; i < n_input - 1; ++i)
       weights[i] = uni()/2.0;  

   w_bias = uni()/2.0;
}


void neuron::activate()
{
   active = TRUE;
}

void neuron::deactivate()
{
   active = FALSE;
}
