# cmake for nerual network project

cmake_minimum_required(VERSION 2.8)
find_library(PYTHON libboost_python.so PATHS "${BOOST_LIBRARY_DIRS}")

set(SOURCES
   hello_boost_python.cpp
)

add_library(pyhello SHARED ${SOURCES})
target_link_libraries(pyhello ${PYTHON})

