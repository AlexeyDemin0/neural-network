add_executable(${PROJECT_NAME}Example1
	"main.cpp"
)

target_link_libraries(${PROJECT_NAME}Example1 PRIVATE ${PROJECT_NAME})