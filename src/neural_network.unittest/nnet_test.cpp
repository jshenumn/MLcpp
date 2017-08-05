#include "pch.h"
#include <cmath>
#include "../neural_network/nnet.h"
#define MAX_ITER 50000

TEST(bpnet, bpnet_ctr)
{
    std::vector<int> hidden_layers(3,3);

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_sst>(3,3,2,hidden_layers,3, SIGMOID);

    int num_hl = nt->get_n_hidden_layers();

    EXPECT_EQ(num_hl, 3);

}

// TEST FOR 0 HIDDEN LAYERS AND 3 INPUT LAYERS
TEST(bpnet, bpnet_train_XOR)
{
    //create training data
    std::vector<std::vector<double> > train_data(4, std::vector<double>(2,0.0));
    train_data[0][0] = 0.0;
    train_data[0][1] = 0.0;
    train_data[1][0] = 0.0;
    train_data[1][1] = 1.0;
    train_data[2][0] = 0.0;
    train_data[2][1] = 1.0;
    train_data[3][0] = 1.0;
    train_data[3][1] = 1.0;

    std::vector<std::vector<double> > train_class(4, std::vector<double>(1,0.0));
    train_class[0][0] = 0.0;
    train_class[1][0] = 1.0;
    train_class[2][0] = 1.0;
    train_class[3][0] = 0.0;

    std::vector<std::vector<double> > output_class(4, std::vector<double>(1,0.0));

    std::vector<int> hidden_layers;
    double err;
    double err_old = 1.0;

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_sst>(2,3,1,hidden_layers,0, SIGMOID);

    for(int ep = 0; ep < MAX_ITER; ep++)
    {
        err = 0.0;
        for(int j = 0; j < 4; j++)
        {
            err += nt->train(train_data[j], train_class[j], 0.2, 0.1);
        }
        err /= 4;
        if(fabs(err - err_old)  < 1e-6)
           break;
        err_old = err;
    }

    for(int i=0; i<4; i++)
    {
        nt->getOutput(train_data[i], output_class[i]);
    }

    EXPECT_GE(0.1, output_class[0][0]);
    EXPECT_LE(0.9, output_class[1][0]);
    EXPECT_LE(0.9, output_class[2][0]);
    EXPECT_GE(0.1, output_class[3][0]);

}

// TEST FOR 1 HIDDEN LAYERS AND 2 INPUT LAYERS
TEST(bpnet, bpnet_train_XOR_MLP)
{
    //create training data
    std::vector<std::vector<double> > train_data(4, std::vector<double>(2,0.0));
    train_data[0][0] = 0.0;
    train_data[0][1] = 0.0;
    train_data[1][0] = 0.0;
    train_data[1][1] = 1.0;
    train_data[2][0] = 0.0;
    train_data[2][1] = 1.0;
    train_data[3][0] = 1.0;
    train_data[3][1] = 1.0;

    std::vector<std::vector<double> > train_class(4, std::vector<double>(1,0.0));
    train_class[0][0] = 0.0;
    train_class[1][0] = 1.0;
    train_class[2][0] = 1.0;
    train_class[3][0] = 0.0;

    std::vector<std::vector<double> > output_class(4, std::vector<double>(1,0.0));

    std::vector<int> hidden_layers(1,2);
    double err;
    double err_old = 1.0;

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_sst>(2,2,1,hidden_layers,1, SIGMOID);

    for(int ep = 0; ep < MAX_ITER; ep++)
    {
        err = 0.0;
        for(int j = 0; j < 4; j++)
        {
            err += nt->train(train_data[j], train_class[j], 0.2, 0.1);
        }
        err /= 4;
        if(fabs(err - err_old)  < 1e-6)
           break;
        err_old = err;
    }

    for(int i=0; i<4; i++)
    {
        nt->getOutput(train_data[i], output_class[i]);
    }

    EXPECT_GE(0.1, output_class[0][0]);
    EXPECT_LE(0.9, output_class[1][0]);
    EXPECT_LE(0.9, output_class[2][0]);
    EXPECT_GE(0.1, output_class[3][0]);

}

// TEST FOR BULL EYE DATA
TEST(bpnet, bpnet_train_BULL_EYE)
{

}
