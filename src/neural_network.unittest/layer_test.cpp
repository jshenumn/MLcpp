#include "pch.h"
#include "../neural_network/layer.h"


TEST(layer, layer_ctr)
{
   layer* Lsig  = new layer_sigmoid;
   layer* Ltanh = new layer_tanh;
   
   Lsig->create(3,3);
   Ltanh->create(3,3);

   EXPECT_EQ(3, Lsig->n_input);
   EXPECT_EQ(3, Lsig->n_neuron);
   EXPECT_EQ(3, Ltanh->n_input);
   EXPECT_EQ(3, Ltanh->n_neuron);

   EXPECT_EQ(3, Lsig->neurons.size());
   EXPECT_EQ(3, Lsig->layerinput.size());
   EXPECT_EQ(3, Ltanh->neurons.size());
   EXPECT_EQ(3, Ltanh->layerinput.size());


   delete Lsig;
   delete Ltanh;
}


TEST(layer, layer_cal)
{
   layer* Lsig  = new layer_sigmoid;
   layer* Ltanh = new layer_tanh;
   
   Lsig->create(3,3);
   Ltanh->create(3,3);

   Lsig->layerinput[0]  = 1.0;
   Ltanh->layerinput[0] = 1.0; 

   Lsig->calculate();
   Ltanh->calculate();

   EXPECT_GE(1.0, Lsig->neurons[0].output);
   EXPECT_GE(1.0, Ltanh->neurons[0].output);

   delete Lsig;
   delete Ltanh;

}
