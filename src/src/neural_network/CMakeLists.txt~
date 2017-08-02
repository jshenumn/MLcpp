# cmake for nerual network project

cmake_minimum_required(VERSION 2.8)
find_library(RAND libboost_random.so PATHS "${BOOST_LIBRARY_DIRS}")

set(SOURCES
   pch.cpp
   neuron.cpp
   layer.cpp
   nnet.cpp
)

add_library(neural_net STATIC ${SOURCES})
add_library(neural_net_dy SHARED ${SOURCES})
target_link_libraries(neural_net ${RAND})
target_link_libraries(neural_net_dy ${RAND})
