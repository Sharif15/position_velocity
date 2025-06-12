import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch.event_handlers.on_process_exit import OnProcessExit
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Declare a launch argument to toggle camera launching
    launch_camera_arg = DeclareLaunchArgument(
        'launch_camera', default_value='false',
        description='Set to true to launch the Axis camera'
    )

    launch_camera = LaunchConfiguration('launch_camera')


    # Flag used for simulation argument

    launch_sim_arg = DeclareLaunchArgument(
        'sim', default_value='false',
        description='Set to true if running in simulation'
    )
    
    sim = LaunchConfiguration('sim')
 


    # Launch Axis drivers with parameters
    axis_camera_1 = GroupAction([
        #PushRosNamespace('axis_1'),
        IncludeLaunchDescription(
            XMLLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('axis_camera'),
                    'launch',
                    'axis_camera.launch'
                )
            ),
            launch_arguments={
                'hostname': '192.168.0.100',
                'password': 'NAPPLab1',
                'frame_width': '1920',
                'frame_height': '1080',
                'fps': '60',
                'ptz_config': os.path.join(
                    get_package_share_directory('ptz_videography'),
                    'config',
                    'axis_v5925.yaml'
                ),
                'enable_ptz': 'true'
            }.items()
        )
    ], condition=IfCondition(launch_camera))

    # Launch camera calibration node
    calibration_node = Node(
        package='position_velocity',
        executable='calibration',
        name='calibration_node',
        output='screen'
    )

    # Launch object detection node
    object_detection_node = Node(
        package='position_velocity',
        executable='object_detection',
        name='object_detection_node',
        output='screen'
    )

    # Launch position calculation node
    calculate_position_node = Node(
        package='position_velocity',
        executable='calculate_position',
        name='calculate_position_node',
        output='screen'
    )

    # Launch the Velocity node 

    calculate_velocity_node = Node(
        package='position_velocity',
        executable= 'kalman_velocity',
        output='screen'

    )


    # Register event AFTER node definitions
    delayed_launch = RegisterEventHandler(
        OnProcessExit(
            target_action=calibration_node,
            on_exit=[
                object_detection_node,
                calculate_position_node,
                calculate_velocity_node
            ]
        )
    )

    # Define the launch description
    return LaunchDescription([
        launch_camera_arg,
        axis_camera_1,
        calibration_node,
        delayed_launch
    ])
