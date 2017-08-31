// This file defines the neuron network using mpi
// unlike the single node bpnet we need to update gradients
// based on the results from all processors which requires a different
// work-flow


/*! bpnet using mpi
*/
class bpnet_mpi
{
    protected:
    std::unique_ptr<layer>   input_layer; /**< input layer */
    std::unique_ptr<layer>   output_layer;  /**< output layer */
    std::vector<std::unique_ptr<layer> >  hidden_layers; /**< hidden layers holder*/

    int m_num_hidden_layers;
    int m_num_input;
    int m_num_neurons_in;
    int m_num_output;

    std::vector<int> m_hidden_layer_layout;

    public:
    /*! bpnet constructor
      \param i_num_input number of input variables.
      \param i_num_input_neurons number of neurons in input layer.
      \param i_num_output number of output
      \param i_hidden_layers hidden_layers size vector
      \param i_num_hidden_layers number of hidden layers
    */
    bpnet_mpi(int i_num_input, int i_num_input_neurons, int i_num_output,
              std::vector<int> i_hidden_layers, int i_num_hidden_layers);

    /*! bpnet_mpi destructor */
    virtual ~bpnet_mpi(){};

    /*! a virtual function to create neural network */
    virtual void create(){}
    /*! a function to perform forward feeding of data
      \param input the input data vector
    */
    void propagate(const std::vector<double>& input);
    /*! a function to update
      \param layer_index the layer index
    */
    void update(int i_layer_index);

};



class bpnet_mpi_MSE_sigmoid : public bpnet_mpi
{
public:
    /*! constructor initialize from base class
      \param _n_input number of input variables.
      \param _n_neurons_in number of neurons in input layer.
      \param _n_ouput number of output
      \param _hidden_layers hidden_layers size vector
      \param _n_hidden_layers number of hidden layers
    */
    bpnet_mpi_MSE_sigmoid(int i_num_input, int i_num_neurons_in, int i_num_output,
                          std::vector<int> i_hidden_layers, int i_num_hidden_layers): bpnet(i_num_input, i_num_neurons_in,
                          i_num_output,i_hidden_layers, i_num_hidden_layers) {}
    /*! destructor */
   ~bpnet_mpi_MSE_sigmoid(){}

    /*! create network */
    void create();
    /*! forward feeding of data
      \param input the input data vector
    */
    void propagate(const std::vector<double>& input);
    /*! update a layer
       \param layer_index the layer index
    */
    void update(int i_layer_index);
};
