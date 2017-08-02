#include "pch.h"
#include "../neural_network/neuron.h"

#define TRUE 1
#define FALSE 0

TEST(neuron, neuron_ctr)
{
  neuron nr(10);
  
  EXPECT_EQ(10, nr.weights.size());
  EXPECT_EQ(10, nr.deltas.size());
  EXPECT_EQ(TRUE, nr.active);
  
  nr.deactivate();
  EXPECT_EQ(FALSE, nr.active);
}


TEST(neuron, neuron_rand)
{
   neuron nr(10);
   auto it = std::max_element(std::begin(nr.weights), std::end(nr.weights));   
   EXPECT_GE(0.5, *it);
   it = std::min_element(std::begin(nr.weights), std::end(nr.weights));   
   EXPECT_LE(-0.5, *it);
}
