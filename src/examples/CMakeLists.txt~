# cmake for nerual network project

cmake_minimum_required(VERSION 2.8)

find_library(RAND libboost_random.so PATHS "${BOOST_LIBRARY_DIRS}")
find_library(LOG libboost_log.so PATHS "${BOOST_LIBRARY_DIRS}")


set(SOURCES
   iris.cpp
   banknote.cpp
)

add_library(examples SHARED ${SOURCES})
target_link_libraries(examples ${RAND} ${LOG})
