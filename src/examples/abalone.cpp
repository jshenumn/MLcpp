/*
   do classification of abalone data sets

*/
#include "pch.h"
#include "abalone.h"

void abalone_data::data_info()
{
    std::cout << "Abalone data is relatively small, we could do read and store\n";
    std::cout << "Number of Instances: 4177\n";
    std::cout << "Number of Attributes: 8 \n";
    std::cout << "Attribute information: \n";
    std::cout << "Given is the attribute name, attribute type, the measurement unit and a\n";
    std::cout << "brief description.The number of rings is the value to predict: either \n";
    std::cout << "as a continuous value or as a classification problem.\n";
    std::cout << "Name		Data Type	Meas.	Description \n";
    std::cout << "----		---------	-----	----------- \n";
    std::cout << "Sex		nominal			M, F, and I (infant) \n";
    std::cout << "Length		continuous	mm	Longest shell measurement \n";
    std::cout << "Diameter	continuous	mm	perpendicular to length \n";
    std::cout << "Height		continuous	mm	with meat in shell \n";
    std::cout << "Whole weight	continuous	grams	whole abalone \n";
    std::cout << "Shucked weight	continuous	grams	weight of meat \n";
    std::cout << "Viscera weight	continuous	grams	gut weight (after bleeding) \n";
    std::cout << "Shell weight	continuous	grams	after being dried \n";
    std::cout << "Rings		integer			+1.5 gives the age in years \n";
}

void abalone_data::load(std::string& path)
{
    std::cout << "Loading the Abalone data....\n";
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
