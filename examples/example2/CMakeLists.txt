add_executable(${PROJECT_NAME}Example2
	"main.cpp"
)

target_link_libraries(${PROJECT_NAME}Example2 PRIVATE ${PROJECT_NAME})