#ifndef NEURON_H
#define NEURON_H

/*!
  \brief The neuron structure, which is the smallest unit of a neural network
*/
struct neuron
{
   std::vector<double> weights; /**< The weights (not include bias) */
   std::vector<double> deltas;  /**< The gradient increment */
   double output;
   double bias;
   double w_bias;
   bool active; /**< usuful when perform random shut down*/

   /*! constructor
   */
   neuron();
   /*! destructor 
   */
   ~neuron(){};
   /*! create the neuron 
     \param n_input the dimension of input
   */
   void create(int n_input);  
   /*! activate neuron if deactivated 
   */
   void activate();
   /*! deactivate neuron if activated
   */
   void deactivate();
};

#endif
