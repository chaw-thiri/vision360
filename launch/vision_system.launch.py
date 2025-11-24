#!/usr/bin/env python3
"""
ROS 2 Launch file for Autonomous Driving Vision System.
Launches the vision processing node on the MacBook.
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get the package directory
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(pkg_dir, 'config', 'config.yaml')

    # Declare launch arguments
    config_arg = DeclareLaunchArgument(
        'config_path',
        default_value=config_path,
        description='Path to configuration file'
    )

    show_viz_arg = DeclareLaunchArgument(
        'show_visualization',
        default_value='true',
        description='Show visualization windows'
    )

    # Set ROS_DOMAIN_ID
    set_domain_id = SetEnvironmentVariable(
        name='ROS_DOMAIN_ID',
        value='17'
    )

    # Set config path environment variable
    set_config_path = SetEnvironmentVariable(
        name='VISION_CONFIG_PATH',
        value=LaunchConfiguration('config_path')
    )

    # Vision node
    vision_node = Node(
        package='capstone',
        executable='vision_node',
        name='autonomous_vision_node',
        output='screen',
        parameters=[{
            'show_visualization': LaunchConfiguration('show_visualization')
        }],
        remappings=[
            ('/camera/image_raw/compressed', '/camera/image_raw/compressed'),
            ('/cmd_vel', '/cmd_vel')
        ]
    )

    return LaunchDescription([
        config_arg,
        show_viz_arg,
        set_domain_id,
        set_config_path,
        vision_node
    ])
