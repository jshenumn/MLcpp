#ifndef ABALONE_H_INCLUDED
#define ABALONE_H_INCLUDED
/*
Abalone data is relatively small, we could do read and store

----  Number of Instances: 4177


----- Number of Attributes: 8


----- Attribute information:

   Given is the attribute name, attribute type, the measurement unit and a
   brief description.  The number of rings is the value to predict: either
   as a continuous value or as a classification problem.

	Name		Data Type	Meas.	Description
	----		---------	-----	-----------
	Sex		nominal			M, F, and I (infant)
	Length		continuous	mm	Longest shell measurement
	Diameter	continuous	mm	perpendicular to length
	Height		continuous	mm	with meat in shell
	Whole weight	continuous	grams	whole abalone
	Shucked weight	continuous	grams	weight of meat
	Viscera weight	continuous	grams	gut weight (after bleeding)
	Shell weight	continuous	grams	after being dried
	Rings		integer			+1.5 gives the age in years

*/

#include "dataframe.h"
#include "pch.h"

struct abalone : public dataframe
{


      abalone(int n_ins, int n_att) : dataframe(n_ins, n_att){}
     ~abalone(){}




};







#endif
