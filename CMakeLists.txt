cmake_minimum_required(VERSION 3.8)
project(position_velocity)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectTracking.msg"
  "msg/ObjectTrackingArray.msg"
)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Add dependency for Python package installation
ament_export_dependencies(rosidl_default_runtime)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install resource files (package.xml, etc.) if needed
install(DIRECTORY resource
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
