add_library(ssvio SHARED
        system.cpp
        frontend.cpp
        camera.cpp
        orbextractor.cpp
        frame.cpp
        feature.cpp
        keyframe.cpp
        map.cpp
        mappoint.cpp
        setting.cpp
        orbpattern.cpp
        backend.cpp
        loopclosing.cpp
)

target_link_libraries(ssvio glog)