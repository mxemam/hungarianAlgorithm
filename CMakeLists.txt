cmake_minimum_required(VERSION 3.14)

set(This HungAlgoProj) # Set project name
project(${This} LANGUAGES C CXX) # Set project languages
include(ExternalProject) # Include module to download content

# Define code and compiler settings
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Prepare eigen library files
set(EIGEN_BUILD_DIR   ${CMAKE_BINARY_DIR}/eigen)
set(EIGEN_INSTALL_DIR ${CMAKE_SOURCE_DIR}/include/eigen)

ExternalProject_Add(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
    PREFIX ${EIGEN_BUILD_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${EIGEN_BUILD_DIR}/src/eigen/Eigen ${EIGEN_INSTALL_DIR}
    TEST_COMMAND ""
)
include_directories(SYSTEM ${EIGEN_INSTALL_DIR})
message(STATUS, " Eigen library files set!")

set(Headers
    ${CMAKE_SOURCE_DIR}/include/HungarianAlgorithm.h
) # Header files
set(Sources
    ${CMAKE_SOURCE_DIR}/src/HungarianAlgorithm.cpp
) # Source files

include_directories(${CMAKE_SOURCE_DIR}/include) # Include directories for compilation
add_executable(${This} ${CMAKE_SOURCE_DIR}/main.cpp ${Sources} ${Headers}) # Create and add library for source files
add_dependencies(${This} eigen)
