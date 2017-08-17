#include "pch.h"
#include "../neural_network/nnet.h"
#define MAX_ITER 50000

typedef boost::minstd_rand base_gen_type;


TEST(bpnet, bpnet_ctr)
{
    std::vector<int> hidden_layers(3,3);

    std::unique_ptr<bpnet> nt  = std::make_unique<bpnet_MSE_sigmoid>(3,3,2,hidden_layers,3);
    std::unique_ptr<bpnet> nt2 = std::make_unique<bpnet_CrossEntropy_softmax>(3,3,2,hidden_layers,3);

    nt->create();
    nt2->create();

    int num_hl  = nt->get_n_hidden_layers();
    int num_hl2 = nt2->get_n_hidden_layers();

    EXPECT_EQ(num_hl, 3);
    EXPECT_EQ(num_hl2, 3);

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

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_MSE_sigmoid>(2,3,1,hidden_layers,0);

    nt->create();

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

    std::vector<int> hidden_layers;
    hidden_layers.push_back(2);

    double err;
    double err_old = 1.0;

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_MSE_sigmoid>(2,2,1,hidden_layers,1);

    nt->create();

    for(int ep = 0; ep < MAX_ITER; ep++)
    {
        err = 0.0;
        for(int j = 0; j < 4; j++)
        {
            err += nt->train(train_data[j], train_class[j], 0.2, 0.1);
        }
        err /= 4;
        if(fabs(err - err_old)  < 1e-12)
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


TEST(bpnet, bpnet_train_ABALONE)
{
    std::cout << "Loading the Abalone data....\n";
    //=============== Load data ==================
    std::vector<std::vector<double> > class_label;
    std::vector<std::vector<double> > class_data;

    //=============== Read data ==================
    std::ifstream file("../data/abalone.data");

    if(file.is_open())  //check open or not
    {
        std::vector<double> label;
        std::vector<double> data;
        while(file.peek() != EOF)
        {
            label.clear();
            data.clear();

            std::string line;
            std::getline(file, line);

            std::stringstream thisline(line);
            std::string       token;


            while(getline(thisline, token, ','))
            {
                if(token == "F")
                {
                    label.push_back(1.0);
                    label.push_back(0.0);
                    label.push_back(0.0);
                }
                else if(token == "M")
                {
                    label.push_back(0.0);
                    label.push_back(1.0);
                    label.push_back(0.0);
                }
                else if(token == "I")
                {
                    label.push_back(0.0);
                    label.push_back(0.0);
                    label.push_back(1.0);
                }
                else if(token != "")
                {
                    double dat = std::stod(token.c_str());
                    data.push_back(dat);
                }
            }
            class_data.push_back(data);
            class_label.push_back(label);
        }
    }
    else
    {
        std::cout << "error opening the file. \n";
        exit(0);
    }

    //===============  Train and test split ==============
    std::cout << "  Data loading done, data size is " << class_data.size() << "\n";
    std::cout << "  Feature number is " << class_data[0].size() << "\n";

    std::cout << "  Training set and testing set split....\n";

    std::vector<int> train_index;
    std::vector<int> test_index;

    base_gen_type generator(42u);
    boost::uniform_real<> uni_dist(0.0,1.0);
    boost::variate_generator<base_gen_type&, boost::uniform_real<> > uni(generator, uni_dist);

    for(unsigned int i=0; i < class_data.size(); i++)
    {
       double num = uni();
       if (uni() > 0.67)
       {
           test_index.push_back(int(i));
       }
       else
       {
           train_index.push_back(int(i));
       }
    }
    std::cout << "  Training data is: " << train_index.size() << "\n";
    std::cout << "  Test data is: " << test_index.size() << "\n";

    //==============  Training ========================
    std::cout << "Training Start....\n";

    //neural architecture should be 3 layers, each layer with 8 neurons, 3 output layers
    std::vector<int> hidden_layers;
    hidden_layers.push_back(6);


    double err;
    double err_old = 1.0;

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_CrossEntropy_softmax>(8,8,3,hidden_layers,1);

    nt->create();

    int N_T = 50;

    for(int ep = 0; ep < 30000; ep++)
    {
        err = 0.0;
        for(int j = 0; j < N_T; j++)
        {
            err += nt->train(class_data[train_index[j]], class_label[train_index[j]], 0.002, 0.001);
        }
        if(fabs(err - err_old)  < 1e-8)
            break;
        err_old = err;
        //std::cout << "Epoch " << ep << " done, error is " << err/N_T << "\n";
    }


    std::vector<std::vector<double> > output_class;
    for(int i=0; i<N_T; i++)
    {
        std::vector<double> vec(3,0.0);
        output_class.push_back(vec);
    }

    int corr = 0;
    for(int i=0; i<N_T; i++)
    {
        nt->getOutput(class_data[test_index[i]], output_class[i]);
        double temp,a,b,c;
        temp = (output_class[i][0] > output_class[i][1])? output_class[i][0]:output_class[i][1];
        temp = (output_class[i][2] > temp)? output_class[i][2]:temp;
        a =  (output_class[i][0] < temp)? 0.0:1.0;
        b =  (output_class[i][1] < temp)? 0.0:1.0;
        c =  (output_class[i][2] < temp)? 0.0:1.0;

        double ter = abs(class_label[test_index[i]][0] - a)
                                    + abs(class_label[test_index[i]][1] - b)
                                    + abs(class_label[test_index[i]][2] - c) ;
        if(ter < 0.1) corr++;
    }
    std::cout << "Training done!  Accuracy is " << (double)corr/N_T * 100 << " % \n";
    EXPECT_GE((double)corr/N_T, 0.50);

}
