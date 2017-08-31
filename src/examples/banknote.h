#ifndef BANKNOTE_H_INCLUDED
#define BANKNOTE_H_INCLUDED


#include "dataframe.h"
#include "pch.h"
/*!
  \brief  The UCI data banknote data set
*/
struct banknote_data : public dataframe
{
      banknote_data():dataframe(){}
     ~banknote_data(){}

      void load(std::string& path);
      void data_info();
};


#endif // YEAST_H_INCLUDED
