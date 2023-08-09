add_executable(test_ui test_ui.cpp)
target_link_libraries(test_ui ui ${OpenCV_LIBS} glog gflags)
add_executable(test_system test_system.cpp)
target_link_libraries(test_system ui ${OpenCV_LIBS} glog gflags ssvio ui)