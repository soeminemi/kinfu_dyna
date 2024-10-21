cmake_minimum_required(VERSION 3.2.1)
set(CMAKE_CXX_STANDARD 17)
# ---[ Configurations types
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "Possible configurations" FORCE)
if (DEFINED CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
endif()

set(CMAKE_BUILD_TYPE RELEASE)
# ---[ Solution name
project(kfusion C CXX)

# ---[ utility
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
include(cmake/Utils.cmake)
include(cmake/Targets.cmake)


# ---[ find dependencies
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)
find_package(CUDA REQUIRED)
find_package(OpenNI REQUIRED)
find_package(Ceres REQUIRED)
set(DEPENDENCIES realsense2 )
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OPENNI_INCLUDE_DIR} ${CERES_INCLUDE_DIRS} /usr/include/eigen3)
# ---[ misc settings
if(USE_PROJECT_FOLDERS)
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)
  set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMakeTargets")
endif()

# ---[ cuda settings
set(HAVE_CUDA 1)
if (CUDA_VERSION_MAJOR GREATER 8)
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_80,code=sm_80")
else()
  list(APPEND CUDA_NVCC_FLAGS "-gencode;arch=compute_50,code=sm_50")
endif()

if(UNIX OR APPLE)
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler;-fPIC;")
endif()

warnings_disable(CMAKE_CXX_FLAGS /wd4985)

add_subdirectory(kfusion)
add_subdirectory(apps)
