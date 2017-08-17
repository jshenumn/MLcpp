#ifndef DATAFRAME_H_INCLUDED
#define DATAFRAME_H_INCLUDED

#include "pch.h"

struct dataframe
{
    int n_instance;
    int n_attributes;

    vector<double> train_class;
    vector<double> train_data;
    vector<double>

    dataframe(int n_ins, int n_attrib);
    virtual ~dataframe(){}
    virtual int class_label(){}
};


#endif // DATAFRAME_H_INCLUDED
