add_executable(test_ui test_ui.cpp)
target_link_libraries(test_ui ui ${OpenCV_LIBS} glog gflags)