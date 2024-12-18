list(APPEND CUDA_NVCC_FLAGS "--ftz=true;--prec-div=false;--prec-sqrt=false")

# Find Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_module_library(kfusion)
message(STATUS "G2O_LIBRARIES: ${G2O_LIBRARY}")
target_link_libraries(kfusion ${CUDA_LIBRARIES} ${PCL_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${OpenCV_LIBS} ${OPENNI_LIBRARY} -lpthread ${g2o_LIBRARIES} ${CERES_LIBRARIES} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
