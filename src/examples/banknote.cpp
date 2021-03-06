/*
   do classification of abalone data sets

*/
#include "pch.h"
#include "banknote.h"

void banknote_data::data_info()
{
    std::cout << "1. variance of Wavelet Transformed image (continuous)" << std::endl;
    std::cout << "2. skewness of Wavelet Transformed image (continuous) " << std::endl;
    std::cout << "3. curtosis of Wavelet Transformed image (continuous)" << std::endl;
    std::cout << "4. entropy of image (continuous) " << std::endl;
    std::cout << "5. class (integer) " << std::endl;
}

void banknote_data::load(std::string& path)
{
    std::cout << "Loading the banknote data....\n";
//=============== Read data ==================
    std::ifstream file(path);

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
                label.resize(2,0.0);
                if(token == "0")
                {
                    label[0] = 1.0;
                }
                else if (token == "1")
                {
                    label[1] = 1.0;
                }
                else if(token != "")
                {
                    double dat = std::stod(token.c_str());
                    data.push_back(dat);
                }

            }
            all_feature.push_back(data);
            all_label.push_back(label);
        }
    }
    else
    {
        std::cout << "error opening the file. \n";
        exit(0);
    }

    n_instance   = all_feature.size();
    n_attributes = all_feature[0].size();

}
