# cmake for nerual network project

option(BUILD_DOC "Build documentation" ON)
option(USE_OPENMP "Use OpenMP" OFF)

cmake_minimum_required(VERSION 2.8)


#===========================  PACKAGE MANAGEMENT =======================
set(BOOST_VERSION 1.54)
set(GOOGLE_TEST_VERSION 1.8.0)
set(BOOST_ROOT $ENV{THIRDPARTY}/boost/${BOOST_VERSION})
set(GOOGLE_TEST_ROOT $ENV{THIRDPARTY}/googletest/${GOOGLE_TEST_VERSION})

set(BOOST_LIBRARY_DIRS "${BOOST_ROOT}/lib")
set(BOOST_INCLUDE_DIRS "${BOOST_ROOT}/include")

set(GTEST_INCLUDE_DIRS "${GOOGLE_TEST_ROOT}/include")
set(GTEST_LIBRARY_DIRS "${GOOGLE_TEST_ROOT}/lib")

set(PYTHON_INCLUDE_DIRS "/usr/include/python2.7")

#==========================  INCLUDE DIRECTORY ==========================
include_directories(${Boost_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

link_directories(${BOOST_LIBRARY_DIRS} ${GTEST_LIBRARY_DIRS})


#=========================== COMPILER OPTIONS ===========================
set(CMAKE_CXX_COMPILER g++-5)
set(CMAKE_CXX_FLAGS "-std=c++14 -lstdc++ -pthread -DBOOST_LOG_DYN_LINK")
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")


#============================  OPTIONS ==================================
if(USE_OPENMP)
   find_package(OpenMP)
   if (OPENMP_FOUND)
     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   endif(OPENMP_FOUND)
endif(USE_OPENMP)

if(BUILD_DOC)
   find_package(Doxygen)
   if(DOXYGEN_FOUND)
     set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/../docs/Doxygen.in)
     set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/../docs/Doxygen)

     configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
     message(STATUS "Doxygen build started")

     add_custom_target( doc_doxygen ALL
         COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_IN}
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../docs
         COMMENT "Generating API documentation with Doxygen"
         VERBATIM )
   else(DOXYGEN_FOUND)
       message("Doxygen need to be installed to generate the doxygen documentation")
   endif(DOXYGEN_FOUND)
endif(BUILD_DOC)

#=======================================================================


#========================  OUTPUT OPTIONS ==============================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../lib")


#========================  SUBDIRECTORIES ==============================
add_subdirectory(neural_network)
add_subdirectory(examples)
add_subdirectory(python_wrapper)
add_subdirectory(neural_network.mpi)

add_subdirectory(neural_network.unittest)


#==========================   TESTING ===================================
enable_testing()
add_test(NAME neuron_network_tests COMMAND neural_net_unittest WORKING_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

