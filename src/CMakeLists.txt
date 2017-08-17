# cmake for nerual network project

cmake_minimum_required(VERSION 2.8)

set(BOOST_VERSION 1.54)
set(GOOGLE_TEST_VERSION 1.8.0)
set(BOOST_ROOT $ENV{THIRDPARTY}/boost/${BOOST_VERSION})
set(GOOGLE_TEST_ROOT $ENV{THIRDPARTY}/googletest/${GOOGLE_TEST_VERSION})

set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")

set(GTEST_INCLUDE_DIRS "${GOOGLE_TEST_ROOT}/include")
set(GTEST_LIBRARY_DIRS "${GOOGLE_TEST_ROOT}/lib")

set(PYTHON_INCLUDE_DIRS "/usr/include/python2.7")

include_directories(${Boost_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

link_directories(${BOOST_LIBRARY_DIRS} ${GTEST_LIBRARY_DIRS})

set(CMAKE_CXX_COMPILER g++-5)
set(CMAKE_CXX_FLAGS "-std=c++14 -lstdc++ -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib")

add_subdirectory(neural_network)
add_subdirectory(python_wrapper)

add_subdirectory(neural_network.unittest)

enable_testing()
add_test(NAME neuron_network_tests COMMAND neural_net_unittest WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

