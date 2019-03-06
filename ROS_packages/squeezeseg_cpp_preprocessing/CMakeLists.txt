cmake_minimum_required(VERSION 2.8.3)
project(squeezeseg_cpp_preprocessing)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES squeezeseg_cpp_preprocessing
#  CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(cpp_preprocessing src/cpp_preprocessing.cpp)
add_dependencies(cpp_preprocessing ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(cpp_preprocessing
  ${catkin_LIBRARIES}
)
