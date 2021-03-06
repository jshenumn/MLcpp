# cmake for nerual network project

cmake_minimum_required(VERSION 2.8)

set(NNET "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libneural_net.a")
set(DATA "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libexamples.so")

find_library(GTLIB libgtest.a PATHS "${GTEST_LIBRARY_DIRS}")
find_library(GTLIBMAIN libgtest_main.a PATHS "${GTEST_LIBRARY_DIRS}")
find_library(RAND libboost_random.so PATHS "${BOOST_LIBRARY_DIRS}")
find_library(LOG libboost_log.so PATHS "${BOOST_LIBRARY_DIRS}")

include_directories(
    "${CMAKE_CURRENT_SOURCE_DIR}/../neural_network/"
    "${CMAKE_CURRENT_SOURCE_DIR}/../examples"
    "${GTEST_INCLUDE_DIRS}"
)


set(SOURCES
   main.cpp
   neuron_test.cpp
   layer_test.cpp
   nnet_test.cpp
)

add_executable(neural_net_unittest ${SOURCES})
add_dependencies(neural_net_unittest neural_net)
target_link_libraries(neural_net_unittest ${NNET} ${GTLIBMAIN} ${GTLIB} ${RAND} ${LOG} ${DATA})
