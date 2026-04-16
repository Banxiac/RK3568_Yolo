cmake_minimum_required(VERSION 3.4.1)
project(edge_ai_pipeline)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "lib")

if(CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH aarch64)
else()
  set(LIB_ARCH armhf)
endif()

# Auto-detect rknpu2 path
foreach(_candidate
    /home/cat/rknpu2
    /home/lubancat/lubancat_ai_manual_code/dev_env/rknpu2
    $ENV{HOME}/rknpu2)
  if(EXISTS "${_candidate}")
    set(RKNPU2_PATH ${_candidate})
    break()
  endif()
endforeach()
if(NOT RKNPU2_PATH)
  message(FATAL_ERROR "Cannot find rknpu2 directory, set RKNPU2_PATH manually")
endif()
set(RKNN_DEMO_PATH ${RKNPU2_PATH}/rknn_yolov5_demo)

# RKNN
set(RKNN_API_PATH ${RKNPU2_PATH}/runtime/RK356X/Linux/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)

# RGA
set(RGA_PATH ${RKNPU2_PATH}/rknn_yolov5_demo/../3rdparty/rga/RK356X)
set(RGA_LIB ${RGA_PATH}/lib/Linux/${LIB_ARCH}/librga.so)
include_directories(${RGA_PATH}/include)

# OpenCV
set(OpenCV_DIR ${RKNPU2_PATH}/rknn_yolov5_demo/../3rdparty/opencv/opencv-linux-aarch64/share/OpenCV)
find_package(OpenCV REQUIRED)

# mosquitto: link by name, resolved at runtime on target board
set(MOSQUITTO_LIB mosquitto)

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${RKNN_DEMO_PATH}/include
  ${RKNPU2_PATH}/rknn_yolov5_demo/../3rdparty
)

add_executable(edge_ai_pipeline
  src/main.cpp
  src/capture.cpp
  src/inference.cpp
  src/alarm.cpp
  ${RKNN_DEMO_PATH}/src/postprocess.cc
)

target_link_libraries(edge_ai_pipeline
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  ${OpenCV_LIBS}
  ${MOSQUITTO_LIB}
  pthread
)

install(TARGETS edge_ai_pipeline DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
install(PROGRAMS ${RGA_LIB} DESTINATION lib)

# web_viewer: HTTP server with MJPEG stream + SSE
add_executable(web_viewer
  src/web_viewer.cpp
)

target_link_libraries(web_viewer
  ${OpenCV_LIBS}
  ${MOSQUITTO_LIB}
  pthread
  rt
)

install(TARGETS web_viewer DESTINATION ./)
