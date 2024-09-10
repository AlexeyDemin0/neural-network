add_library(${PROJECT_NAME} STATIC
	"math/matrix.h"
	"math/matrix.cpp"
	"perceptron.h"
	"perceptron.cpp"
	"math/functions.h"
	"math/functions.cpp"
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/include)