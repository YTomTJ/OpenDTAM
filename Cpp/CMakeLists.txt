cmake_minimum_required(VERSION 3.0)
set(CMAKE_BUILD_TYPE Debug)

project(dtam)
cmake_policy(SET CMP0012 NEW)

# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/local_store/Qt/5.3/gcc_64/lib/cmake")# "/local_store/opencv-2.4.9/debug/unix-install/")
# message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb")

find_package(OpenCV 4 REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(YAML-CPP REQUIRED)

# add_definitions( -D__OPENCV_BUILD=1 )
# message(STATUS ${OpenCV_CONSIDERED_CONFIGS})
macro(add_sources)
    file(RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")

    foreach(_src ${ARGN})
        if(_relPath)
            list(APPEND DTAM_SOURCES "${_relPath}/${_src}")
        else()
            list(APPEND DTAM_SOURCES "${_src}")
        endif()
    endforeach()

    if(_relPath)
        # propagate SRCS to parent directory
        set(DTAM_SOURCES ${DTAM_SOURCES} PARENT_SCOPE)
    endif()
endmacro(add_sources)

set(BASEPATH "${CMAKE_SOURCE_DIR}")

# add_library(GraphicsLib SHARED graphics.cpp)#must be in own library so we can link against it.
add_subdirectory(CostVolume)
add_subdirectory(Track)
add_subdirectory(Optimizer)
add_subdirectory(DepthmapDenoiseWeightedHuber)
add_subdirectory(utils)

include_directories("${BASEPATH}" ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -g
    # -gencode arch=compute_30,code=sm_30;
    # -gencode arch=compute_35,code=sm_35;
    # -gencode arch=compute_50,code=sm_50;
    -gencode arch=compute_70,code=sm_70;
    -gencode=arch=compute_75,code=compute_75;
    --use_fast_math;
    --restrict;
    -lineinfo;
    -O3
)
cuda_add_library(OpenDTAM SHARED ${DTAM_SOURCES})
target_link_libraries(OpenDTAM pthread opencv_cudaimgproc opencv_cudastereo ${Boost_LIBRARIES})

add_executable(testprog testprog.cpp graphics.cpp)
target_link_libraries(testprog OpenDTAM ${OpenCV_LIBS} ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES})

add_executable(dtam main.cpp graphics.cpp)
target_link_libraries(dtam OpenDTAM ${OpenCV_LIBS} ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES})
