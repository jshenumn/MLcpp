#ifndef DATAFRAME_H_INCLUDED
#define DATAFRAME_H_INCLUDED

#include "pch.h"

/*!
  \brief The data framework for classification problems.
         It defines the way to load data and handle categorical class variables
         and display data info 
*/
struct dataframe
{
    int n_instance;
    int n_attributes;

    std::vector<std::vector<double> > all_label;  /**< all class label which is transformed into numerical values */
    std::vector<std::vector<double> > all_feature;  /**< feature sets*/
    
    /** constructor */
    dataframe(){}  
    /** destructor */
    virtual ~dataframe(){}
    /** load data from file
       \param path:  string path to file
    */
    virtual void load(std::string& path){}
    /** print data info
    */
    virtual void data_info(){}

};


#endif // DATAFRAME_H_INCLUDED
