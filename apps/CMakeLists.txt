find_package(PCL REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenCV REQUIRED)
find_package(ultralytics REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/kfusion/include
${PCL_INCLUDE_DIRS}
${Ceres_INCLUDE_DIRS}
/usr/include/jsoncpp
${CMAKE_SOURCE_DIR}/apps
)

file(GLOB srcs *.cpp *.hpp)
add_executable(demo ${srcs})
#message(${srcs} )
target_link_libraries(demo ${OpenCV_LIBS} ${PCL_LIBRARIES} kfusion -lceres -lglog -lbody3D -lcurl -lopenjp2 -ljsoncpp -lssl -lcrypto)

set_target_properties(demo PROPERTIES
    DEBUG_POSTFIX "d"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

install(TARGETS demo RUNTIME DESTINATION bin COMPONENT main)
install(FILES ${srcs} DESTINATION app COMPONENT main)
