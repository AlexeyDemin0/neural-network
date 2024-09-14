cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME "NeuralNetwork")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_DEBUG "Enable debug information" OFF)
option(ENABLE_EXAMPLES "Enable examples compilation" OFF)

if(${ENABLE_DEBUG})
	set(CMAKE_BUILD_TYPE "Debug")
else()
	set(CMAKE_BUILD_TYPE "Release")
endif()

project(${PROJECT_NAME})

add_subdirectory(src)

if(${ENABLE_EXAMPLES})
	add_subdirectory(examples)
endif()