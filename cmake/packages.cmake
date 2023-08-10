list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# third party include path
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/eigen)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/sophus)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/g2o)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty)

# OpenCV
find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# csparse
find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

# cholmod
find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDE_DIRS})

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
message(STATUS "Pangolin_INCLUDE_DIRS ${Pangolin_INCLUDE_DIRS}")
message(STATUS "Pangolin Version ${Pangolin_VERSION}")
set(g2o_libs
        ${CSPARSE_LIBRARY} ${CHOLMOD_LIBRARY}
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_stuff.so
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_core.so
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_solver_dense.so
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_solver_csparse.so
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_csparse_extension.so
        ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o_types_sba.so
)
set(dbow_libs ${PROJECT_SOURCE_DIR}/thirdparty/DBoW2/lib/libDBoW2.so)
set(pangolin_libs ${Pangolin_LIBRARIES} GL GLU GLEW glut)
