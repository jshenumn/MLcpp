#include <boost/python.hpp>

char const* greet()
{
  return "HELLO WORLD FROM C++!";
}


BOOST_PYTHON_MODULE(libpyhello)
{
  using namespace boost::python;
  def("greet", greet);
}
