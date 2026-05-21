from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # 1. Pioneer base driver
        Node(
            package='ariaNode',
            executable='ariaNode',
            name='ariaNode',
            output='screen',
            arguments=['-rp', '/dev/ttyUSB0'],
        ),

        # 2. Lakibeam LiDAR
        Node(
            package='lakibeam1',
            executable='lakibeam1_scan_node',
            name='lakibeam_lidar',
            # output='screen',
            arguments=['--ros-args', '--log-level', 'warn'],
            parameters=[{
                'frame_id':         'laser',
                'output_topic':     'scan',
                'sensorip':         '192.168.198.2',  
                'hostip':           '0.0.0.0',
                'port':             '2368',
                'inverted':         False,
                'angle_offset':     0,
                'scanfreq':         '30',
                'filter':           '3',
                'laser_enable':     'true',
                'scan_range_start': '0',
                'scan_range_stop':  '360',
            }],
        ),

        # # Sick Scan Lidar
        # Node(
        #     package='sick_scan_xd',
        #     executable='sick_generic_caller',
        #     name='sick_lidar',
        #     output='screen',
        #     arguments=[
        #         '/opt/ros/jazzy/share/sick_scan_xd/launch/sick_tim_7xx.launch',
        #         'hostname:=192.168.0.1',
        #         'use_binary_protocol:=false',
        #         'frame_id:=laser',
        #     ]
        # ),

        # 3. Joy node
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            output='screen',
            parameters=[{'dev': '/dev/input/js0'}]
        ),

        # 4. Foxglove bridge
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            name='foxglove_bridge',
            output='screen',
            parameters=[{
                'port': 8765,
                'address': '0.0.0.0',
            }]
        ),

        # # 5. Odom publisher — publishes odom->base_link on /tf at 20Hz
        # Node(
        #     package='autobot',
        #     executable='odom_publisher',
        #     name='odom_publisher',
        #     output='screen',
        # ),

        # 6. Static transform: base_link -> laser
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_laser',
            output='screen',
            arguments=['0', '0', '0.2', '0', '0', '0', 'base_link', 'laser']
        ),

        # 7. SLAM toolbox — mapping mode (delayed 3s)
        TimerAction(
            period=3.0,
            actions=[
                Node(
                    package='slam_toolbox',
                    executable='sync_slam_toolbox_node',
                    name='slam_toolbox',
                    output='screen',
                    parameters=[{
                        'use_sim_time':                     False,
                        'odom_frame':                       'odom',
                        'map_frame':                        'map',
                        'base_frame':                       'base_link',
                        'scan_topic':                       '/scan',
                        'mode':                             'mapping',
                        'debug_logging':                    False,
                        'throttle_scans':                   1,
                        'transform_publish_period':         0.02,
                        'map_update_interval':              5.0,
                        'resolution':                       0.05,
                        'max_laser_range':                  20.0,
                        'minimum_time_interval':            0.5,
                        'transform_timeout':                0.2,
                        'tf_buffer_duration':               30.0,
                        'stack_size_to_use':                40000000,
                        'use_scan_matching':                True,
                        'use_scan_barycenter':              True,
                        'minimum_travel_distance':          0.1,
                        'minimum_travel_heading':           0.1,
                        'scan_buffer_size':                 10,
                        'scan_buffer_maximum_scan_distance': 10.0,
                        'link_match_minimum_response_fine': 0.1,
                        'link_scan_maximum_distance':       1.5,
                        'loop_search_maximum_distance':     3.0,
                        'do_loop_closing':                  True,
                        'loop_match_minimum_chain_size':    10,
                        'loop_match_maximum_variance_covariance': 3.0,
                        'loop_match_minimum_response_coarse': 0.35,
                        'loop_match_minimum_response_fine': 0.45,
                        'correlation_search_space_dimension': 0.5,
                        'correlation_search_space_resolution': 0.01,
                        'correlation_search_space_smear_deviation': 0.1,
                        'loop_search_space_dimension':      8.0,
                        'loop_search_space_resolution':     0.05,
                        'loop_search_space_smear_deviation': 0.03,
                        'distance_variance_penalty':        0.5,
                        'angle_variance_penalty':           1.0,
                        'fine_search_angle_offset':         0.00349,
                        'coarse_search_angle_offset':       0.349,
                        'coarse_angle_resolution':          0.0349,
                        'minimum_angle_penalty':            0.9,
                        'minimum_distance_penalty':         0.5,
                        'use_response_expansion':           True,
                    }],
                ),
            ]
        ),

        # 8. DualShock teleop + LiDAR auto + OAK camera (delayed 2s)
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package='autobot',
                    executable='autobot_roll',
                    name='dualshock_mode_teleop',
                    output='screen',
                ),
            ]
        ),

        ComposableNodeContainer(
            name='phidgets_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[],
            output='screen',
        ),

        LoadComposableNodes(
            target_container='phidgets_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='phidgets_spatial',
                    plugin='phidgets::SpatialRosI',
                    name='phidgets_spatial',
                    parameters=['/root/ros2_autobot2/config/phidget_imu.yaml'],
                ),
            ],
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_imu',
            output='screen',
            arguments=['0', '0', '0.1', '0', '0', '0', 'base_link', 'imu_link']
        ),

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            output='screen',
            parameters=['/root/ros2_autobot/config/ekf.yaml'],
            remappings=[('odometry/filtered', '/odom_fused')]
        ),

    ])