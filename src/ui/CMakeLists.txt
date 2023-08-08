add_library(ui SHARED
    pangolin_window_impl.cpp
    pangolin_window.cpp
)

target_link_libraries(ui ${pangolin_libs})