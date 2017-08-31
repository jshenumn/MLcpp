#ifndef ABALONE_H_INCLUDED
#define ABALONE_H_INCLUDED
/*
Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
*/

#include "dataframe.h"
#include "pch.h"
/*!
  \brief  The UCI data iris data set
*/
struct iris_data : public dataframe
{
      iris_data():dataframe(){}
     ~iris_data(){}

      void load(std::string& path);
      void data_info();
};







#endif
