/*
   do classification of abalone data sets

*/
#include "pch.h"
#include "iris.h"

void iris_data::data_info()
{
   std::cout << "1. sepal length in cm " << std::endl;
   std::cout << "2. sepal width in cm " << std::endl;
   std::cout << "3. petal length in cm " << std::endl;
   std::cout << "4. petal width in cm " << std::endl;
   std::cout << "5. class: -- Iris Setosa -- Iris Versicolour -- Iris Virginica " << std::endl;
}

void iris_data::load(std::string& path)
{
    std::cout << "Loading the iris data....\n";
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
                if(token == "Iris-setosa")
                {
                    label.push_back(1.0);
                    label.push_back(0.0);
                    label.push_back(0.0);
                }
                else if(token == "Iris-versicolor")
                {
                    label.push_back(0.0);
                    label.push_back(1.0);
                    label.push_back(0.0);
                }
                else if(token == "Iris-virginica")
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
