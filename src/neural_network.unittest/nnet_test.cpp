#include "pch.h"
#include "../neural_network/nnet.h"
#include "../examples/iris.h"
#include "../examples/banknote.h"

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
        nt->get_output(train_data[i], output_class[i]);
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
        nt->get_output(train_data[i], output_class[i]);
    }

    EXPECT_GE(0.1, output_class[0][0]);
    EXPECT_LE(0.9, output_class[1][0]);
    EXPECT_LE(0.9, output_class[2][0]);
    EXPECT_GE(0.1, output_class[3][0]);

}


TEST(bpnet, DISABLED_bpnet_train_iris)
{
    //===============  Load and preprocessing data ===========
    std::unique_ptr<dataframe> iris = std::make_unique<iris_data>();
    std::string path = "../data/iris.data";
    iris->load(path);

    //===============  Train and test split ==================
    std::cout << "  Training set and testing set split....\n";

    std::vector<int> train_index;
    std::vector<int> test_index;

    base_gen_type generator(42u);
    boost::uniform_real<> uni_dist(0.0,1.0);
    boost::variate_generator<base_gen_type&, boost::uniform_real<> > uni(generator, uni_dist);

    for(unsigned int i=0; i < iris->n_instance; i++)
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

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_CrossEntropy_softmax>(4,4,3,hidden_layers,1);

    nt->create();

    int N_T = train_index.size();

    for(int ep = 0; ep < 10000; ep++)
    {
        err = 0.0;
        for(int j = 0; j < N_T; j++)
        {
            err += nt->train(iris->all_feature[train_index[j]], iris->all_label[train_index[j]], 0.005, 0.001);
        }
        //std::cout << "Training epoch " << ep << " done, error is " << err/N_T << "\n";
    }

    std::vector<std::vector<double> > output_class;
    for(int i=0; i<N_T; i++)
    {
        std::vector<double> vec(3,0.0);
        output_class.push_back(vec);
    }

    int corr = 0;
    for(int i=0; i<test_index.size(); i++)
    {
        nt->get_output(iris->all_feature[test_index[i]], output_class[i]);
        double temp,a,b,c;
        temp = (output_class[i][0] > output_class[i][1])? output_class[i][0]:output_class[i][1];
        temp = (output_class[i][2] > temp)? output_class[i][2]:temp;
        a =  (output_class[i][0] < temp)? 0.0:1.0;
        b =  (output_class[i][1] < temp)? 0.0:1.0;
        c =  (output_class[i][2] < temp)? 0.0:1.0;

        double ter = abs(iris->all_label[test_index[i]][0] - a)
                     + abs(iris->all_label[test_index[i]][1] - b)
                     + abs(iris->all_label[test_index[i]][2] - c) ;
        if(ter < 0.1) corr++;
    }
    std::cout << "Training done!  Accuracy is " << (double)corr/test_index.size() * 100 << " % \n";
    EXPECT_GE((double)corr/test_index.size(), 0.90);

}

TEST(bpnet, DISABLED_bpnet_train_banknote)
{
    //===============  Load and preprocessing data ===========
    std::unique_ptr<dataframe> banknote = std::make_unique<banknote_data>();
    std::string path = "../data/banknote.data";
    banknote->load(path);

    //===============  Train and test split ==================
    std::cout << "  Training set and testing set split....\n";

    std::vector<int> train_index;
    std::vector<int> test_index;

    base_gen_type generator(42u);
    boost::uniform_real<> uni_dist(0.0,1.0);
    boost::variate_generator<base_gen_type&, boost::uniform_real<> > uni(generator, uni_dist);

    for(unsigned int i=0; i < banknote->n_instance; i++)
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

    //neural architecture should be 3 layers, each layer with 4 neurons, 2 output layers
    std::vector<int> hidden_layers;
    hidden_layers.push_back(3);

    double err;

    std::unique_ptr<bpnet> nt = std::make_unique<bpnet_CrossEntropy_softmax>(4,4,2,hidden_layers,1);

    nt->create();

    int N_T = train_index.size();

    for(int ep = 0; ep < 1000; ep++)
    {
        err = 0.0;
        for(int j = 0; j < N_T; j++)
        {
            err += nt->train(banknote->all_feature[train_index[j]], banknote->all_label[train_index[j]], 0.002, 0.001);
        }
        //std::cout << "Training epoch " << ep << " done, error is " << err/N_T << "\n";
    }

    std::vector<std::vector<double> > output_class;
    for(int i=0; i<N_T; i++)
    {
        std::vector<double> vec(3,0.0);
        output_class.push_back(vec);
    }

    int corr = 0;
    for(int i=0; i<test_index.size(); i++)
    {
        nt->get_output(banknote->all_feature[test_index[i]], output_class[i]);
        double temp, ter;
        temp = *max_element(output_class[i].begin(), output_class[i].end());


        int a[2];
        ter = 0.0;
        for(int j=0; j < 2; j++){
          a[j] =  (output_class[i][j] < temp)? 0.0:1.0;
          ter += abs(banknote->all_label[test_index[i]][j] - a[j]);
        }

        if(ter < 0.1) corr++;
    }
    std::cout << "Training done!  Accuracy is " << (double)corr/test_index.size() * 100 << " % \n";
    EXPECT_GE((double)corr/test_index.size(), 0.90);

}
