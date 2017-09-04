# cmake for nerual network MPI project

cmake_minimum_required(VERSION 2.8)
find_library(RAND libboost_random.so PATHS "${BOOST_LIBRARY_DIRS}")
find_library(LOG libboost_log.so PATHS "${BOOST_LIBRARY_DIRS}")

set(SOURCES
   pch.cpp
   nnet_mpi.cpp
)

add_library(neural_net_mpi SHARED ${SOURCES})
add_dependencies(neural_net_mpi neural_net)
target_link_libraries(neural_net_mpi ${RAND} ${LOG})
