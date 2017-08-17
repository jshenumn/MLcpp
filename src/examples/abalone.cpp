/*
   do classification of abalone data sets

*/
#include "pch.h"

//







// data processing
ifstream file("../../data/abalone.data");

//read first row for column count, initialize
if(file.is_open())  //check open or not
{
    std::string line;
    std::getline(file, line);
    std::stringstream thisline(line);
    std::string       token;

    while(getline(thisline, token, ','))
    {
        CSVcolumn col;
        if(token != "")
        {
            double data = std::stod(token.c_str());
            col.push_back(data);  // each col gets head elements
        }
        CSVtable.push_back(col);
    }
}
else
{
    exit(0);
}

//continue to read
while(file)
{
    std::string line;
    std::getline(file, line);  //get rows

    std::stringstream thisline(line);
    std::string       token;

    int colIdx = 0;
    while(getline(thisline, token, ','))
    {
        if(token != "")
        {
            double data = std::stod(token.c_str());
            CSVtable[colIdx].push_back(data);
        }
        colIdx++;
    }
}
