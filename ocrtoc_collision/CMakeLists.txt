cmake_minimum_required(VERSION 3.0.2)
project(ocrtoc_collision)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  std_msgs
  octomap REQUIRED
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ocrtoc_collision
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY
  # config
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  # scripts/perception_entrance.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
link_directories(${OCTOMAP_LIBRARY_DIRS})

include_directories(${OCTOMAP_INCLUDE_DIRS})
# target_link_libraries(${OCTOMAP_LIBRARIES})