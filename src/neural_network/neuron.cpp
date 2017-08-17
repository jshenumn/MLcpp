#include "pch.h"
#include "neuron.h"
#include "boost/random.hpp"
#include "boost/random/uniform_real.hpp"
#include "boost/random/variate_generator.hpp"
#include "boost/random/random_device.hpp"

typedef boost::minstd_rand base_generator_type;


neuron::neuron():weights(0),deltas(0),output(0),bias(0),w_bias(0)
{
}


void neuron::create(int n_input)
{
   weights.resize(n_input);
   deltas.resize(n_input);
   output = 0.0;
   bias   = 1.0;
   active = TRUE;
   std::fill(deltas.begin(), deltas.end(), 0.0);
   std::fill(weights.begin(), weights.end(), 0.0);


   //random gen a vector
   boost::random_device dev;
   boost::mt19937 gen(dev);
   boost::uniform_real<> dist(-2.0, 2.0);
   boost::variate_generator<boost::mt19937&, boost::uniform_real<> > uni(gen, dist);
   for(int i = 0; i < n_input; ++i)
   {
      weights[i] = uni()/2.0;
   }
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
