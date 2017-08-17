#include "pch.h"

void printvector(const std::vector<double>& vec)
{
    for(int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << ",";
    }
    std::cout << "\n";
}
