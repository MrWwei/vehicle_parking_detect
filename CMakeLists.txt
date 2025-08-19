cmake_minimum_required(VERSION 3.5)
project(Detect_SDK_TRT_V1.2 LANGUAGES CXX CUDA)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 启用CUDA分离编译
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fPIC -O0 -Wfatal-errors -pthread -w -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w")

set(ThirdParty /home/ubuntu/ThirdParty)

# 查找CUDA
# find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# opencv
include_directories(${ThirdParty}/opencv-4.5.4/build/install/include/opencv4)
link_directories(${ThirdParty}/opencv-4.5.4/build/install/lib)
# cryptopp
include_directories(${ThirdParty}/cryptopp-CRYPTOPP_8_7_0/install/include)
# aes getkey
add_library(aes_key SHARED IMPORTED)
set_target_properties( aes_key PROPERTIES IMPORTED_LOCATION ${ThirdParty}/aes_key/lib/libEncry.so )
include_directories(${ThirdParty}/aes_key/include)

# object_detection
add_library(detect_lib SHARED IMPORTED)
set_target_properties( detect_lib PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/object_detect/build/libdetect_yl_nc11_trt_X86_V6.1.5_u.so )
include_directories(/home/ubuntu/wtwei/object_detect/include)

# object_track
add_library(track_lib SHARED IMPORTED)
set_target_properties( track_lib PROPERTIES IMPORTED_LOCATION /home/ubuntu/wtwei/object_track/build/libtrack_btk_n_x86_V6.3.so )
include_directories(/home/ubuntu/wtwei/object_track/include)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(vehicle_parking_detect_V1.0 SHARED ${CMAKE_SOURCE_DIR}/src/vehicle_parking_detect.cpp ${CMAKE_SOURCE_DIR}/src/world_coord.cu ${CMAKE_SOURCE_DIR}/src/cuda_features.cu)
target_link_libraries(vehicle_parking_detect_V1.0 
    opencv_img_hash 
    opencv_world
    cudart
)


# 添加CUDA可执行文件
add_executable(main main.cpp) 

# 设置CUDA属性
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 链接库，包括CUDA运行时
target_link_libraries(main 
    vehicle_parking_detect_V1.0
    detect_lib 
    track_lib 
    opencv_img_hash 
    opencv_world
    # ${CUDA_LIBRARIES}
    cudart
)



#################### jni #################
# include_directories(${ThirdParty}/jdk1.8.0_381/include)
# include_directories(${ThirdParty}/jdk1.8.0_381/include/linux)

# # V2.0
# set(jni_target_name detect_yl_nc11.4_trt_X86_SDK6.1.5_JNI_V6.1.5)
# include_directories(${CMAKE_SOURCE_DIR}/jni_detect)
# add_library(${jni_target_name} SHARED ${CMAKE_SOURCE_DIR}/jni_detect/cn_xtkj_jni_algor_YoloDetectAlgor.cpp )
# target_link_libraries(${jni_target_name} ${sdk_target_name}) 

