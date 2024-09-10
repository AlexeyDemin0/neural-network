cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME "neural-network")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(${PROJECT_NAME})

add_library(${PROJECT_NAME} STATIC
	"src/math/matrix.h"
	"src/math/matrix.cpp"
	"src/perceptron.h"
	"src/perceptron.cpp"
	"src/math/functions.h"
	"src/math/functions.cpp"
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)