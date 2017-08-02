#include "pch.h"
#include "../neural_network/nnet.h"

TEST(nnet, nnet_ctr)
{
   int* hidden_layers = new int[3];
   hidden_layers[0] = 3;
   hidden_layers[1] = 3;
   hidden_layers[2] = 3;
   
   nnet* nt = new bpnet_sst(3,3,2,hidden_layers,3, SIGMOID); 
   
   int num_hl = nt->get_n_hidden_layers();     
   
   EXPECT_EQ(num_hl, 3);

   delete [] hidden_layers;
   delete nt;
}
