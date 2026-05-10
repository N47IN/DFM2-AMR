#!/usr/bin/env python3
"""
VDB Semantic Mapping System Launch File

Launches the VDB-based semantic mapping system with three main components:
1. Main resilience node (main.py) - handles drift detection, NARadio processing, and semantic mapping
2. VDB semantic mapping node (vdb_semantic_mapping_node.py) - creates VDB-based semantic voxel maps
3. Narration display node (vlm_node.py) - displays narration and queries VLM

This launch file provides a complete system for resilience monitoring with efficient VDB-based semantic mapping.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    # Declare launch arguments for main resilience node
    declare_flip_y_axis = DeclareLaunchArgument(
        'flip_y_axis',
        default_value='false',
        description='Whether to flip Y axis for coordinate system'
    )
    
    declare_use_tf = DeclareLaunchArgument(
        'use_tf',
        default_value='false',
        description='Whether to use TF for coordinate transforms'
    )

    declare_image_rotation_deg = DeclareLaunchArgument(
        'image_rotation_deg',
        default_value='0',
        description='Rotate incoming camera images by 0 or 180 degrees'
    )
    
    declare_radio_model_version = DeclareLaunchArgument(
        'radio_model_version',
        default_value='radio_v2.5-b',
        description='NARadio model version'
    )
    
    declare_radio_lang_model = DeclareLaunchArgument(
        'radio_lang_model',
        default_value='siglip',
        description='NARadio language model'
    )
    
    declare_radio_input_resolution = DeclareLaunchArgument(
        'radio_input_resolution',
        default_value='512',
        description='NARadio input resolution'
    )
    
    declare_enable_naradio_visualization = DeclareLaunchArgument(
        'enable_naradio_visualization',
        default_value='true',
        description='Whether to enable NARadio visualization'
    )
    
    declare_enable_combined_segmentation = DeclareLaunchArgument(
        'enable_combined_segmentation',
        default_value='true',
        description='Whether to enable combined segmentation'
    )
    
    declare_enable_voxel_mapping = DeclareLaunchArgument(
        'enable_voxel_mapping',
        default_value='true',
        description='Whether to enable voxel mapping'
    )
    
    declare_pose_is_base_link = DeclareLaunchArgument(
        'pose_is_base_link',
        default_value='true',
        description='Whether pose is base link'
    )
    
    # Declare launch arguments for VDB semantic mapping node
    declare_max_range = DeclareLaunchArgument(
        'max_range',
        default_value='5.0',
        description='Maximum depth range for VDB mapping'
    )
    
    declare_voxel_resolution = DeclareLaunchArgument(
        'voxel_resolution',
        default_value='0.1',
        description='Voxel resolution in meters'
    )
    
    declare_min_range = DeclareLaunchArgument(
        'min_range',
        default_value='0.1',
        description='Minimum depth range'
    )
    
    declare_enable_semantic_mapping = DeclareLaunchArgument(
        'enable_semantic_mapping',
        default_value='true',
        description='Enable semantic voxel mapping with VLM similarity'
    )
    
    declare_semantic_similarity_threshold = DeclareLaunchArgument(
        'semantic_similarity_threshold',
        default_value='0.6',
        description='Cosine similarity threshold for semantic labeling'
    )
    
    declare_buffers_directory = DeclareLaunchArgument(
        'buffers_directory',
        default_value='/root/AirStack/robot/ros_ws/src/autonomy/4_global/a_world_models/resilience/buffers',
        description='Directory containing saved buffer data with VLM embeddings'
    )
    
    declare_publish_occupancy_cloud = DeclareLaunchArgument(
        'publish_occupancy_cloud',
        default_value='true',
        description='Whether to publish occupancy point cloud'
    )
    
    declare_publish_semantic_cloud = DeclareLaunchArgument(
        'publish_semantic_cloud',
        default_value='true',
        description='Whether to publish semantic point cloud'
    )
    
    declare_max_cloud_points = DeclareLaunchArgument(
        'max_cloud_points',
        default_value='50000',
        description='Maximum number of points in published clouds'
    )
    
    declare_cloud_publish_rate = DeclareLaunchArgument(
        'cloud_publish_rate',
        default_value='1.0',
        description='Rate for publishing point clouds (Hz)'
    )
    
    declare_stats_publish_rate = DeclareLaunchArgument(
        'stats_publish_rate',
        default_value='1.0',
        description='Rate for publishing statistics (Hz)'
    )
    
    # Declare launch arguments for narration display node
    declare_vlm_api_key = DeclareLaunchArgument(
        'vlm_api_key',
        default_value='',
        description='API key for VLM service (optional)'
    )
    
    declare_vlm_base_url = DeclareLaunchArgument(
        'vlm_base_url',
        default_value='http://localhost:8000/v1',
        description='Base URL for VLM API'
    )
    
    declare_vlm_model = DeclareLaunchArgument(
        'vlm_model',
        default_value='llava-v1.6-mistral-7b',
        description='VLM model to use'
    )
    
    declare_save_images = DeclareLaunchArgument(
        'save_images',
        default_value='true',
        description='Whether to save images with narration text'
    )
    
    # Main Resilience Node
    main_resilience_node = Node(
        package='resilience',
        executable='main.py',
        name='resilience_node',
        output='screen',
        parameters=[{
            'flip_y_axis': LaunchConfiguration('flip_y_axis'),
            'use_tf': LaunchConfiguration('use_tf'),
            'image_rotation_deg': LaunchConfiguration('image_rotation_deg'),
            'radio_model_version': LaunchConfiguration('radio_model_version'),
            'radio_lang_model': LaunchConfiguration('radio_lang_model'),
            'radio_input_resolution': LaunchConfiguration('radio_input_resolution'),
            'enable_naradio_visualization': LaunchConfiguration('enable_naradio_visualization'),
            'enable_combined_segmentation': LaunchConfiguration('enable_combined_segmentation'),
            'enable_voxel_mapping': LaunchConfiguration('enable_voxel_mapping'),
            'pose_is_base_link': LaunchConfiguration('pose_is_base_link'),
            'main_config_path': '',
            'mapping_config_path': ''
        }],
        remappings=[
            ('/robot_1/sensors/front_stereo/right/image', '/robot_1/sensors/front_stereo/right/image'),
            ('/robot_1/sensors/front_stereo/depth/depth_registered', '/robot_1/sensors/front_stereo/depth/depth_registered'),
            ('/robot_1/sensors/front_stereo/pose', '/robot_1/sensors/front_stereo/pose'),
            ('/robot_1/sensors/front_stereo/right/camera_info', '/robot_1/sensors/front_stereo/right/camera_info'),
            ('/vlm_answer', '/vlm_answer'),
        ]
    )
    
    # VDB Semantic Mapping Node
    vdb_semantic_mapping_node = Node(
        package='resilience',
        executable='frontier_mapping_node.py',
        name='vdb_semantic_mapping_node',
        output='screen',
        parameters=[{
            'depth_topic': '/robot_1/sensors/front_stereo/depth/depth_registered',
            'camera_info_topic': '/robot_1/sensors/front_stereo/left/camera_info',
            'pose_topic': '/robot_1/sensors/front_stereo/pose',
            'rgb_topic': '/robot_1/sensors/front_stereo/right/image',
            'image_rotation_deg': LaunchConfiguration('image_rotation_deg'),
            'map_frame': 'map',
            'voxel_resolution': LaunchConfiguration('voxel_resolution'),
            'max_range': LaunchConfiguration('max_range'),
            'min_range': LaunchConfiguration('min_range'),
            'probability_hit': 0.7,
            'probability_miss': 0.4,
            'occupancy_threshold': 0.5,
            'publish_occupancy_cloud': LaunchConfiguration('publish_occupancy_cloud'),
            'publish_semantic_cloud': LaunchConfiguration('publish_semantic_cloud'),
            'publish_stats': True,
            'max_cloud_points': LaunchConfiguration('max_cloud_points'),
            'cloud_publish_rate': LaunchConfiguration('cloud_publish_rate'),
            'stats_publish_rate': LaunchConfiguration('stats_publish_rate'),
            'pose_is_base_link': LaunchConfiguration('pose_is_base_link'),
            'apply_optical_frame_rotation': True,
            'cam_to_base_rpy_deg': [0.0, 0.0, 0.0],
            'cam_to_base_xyz': [0.0, 0.0, 0.0],
            'enable_semantic_mapping': LaunchConfiguration('enable_semantic_mapping'),
            'semantic_similarity_threshold': LaunchConfiguration('semantic_similarity_threshold'),
            'buffers_directory': LaunchConfiguration('buffers_directory'),
            'bridge_queue_max_size': 100,
            'bridge_queue_process_interval': 0.1,
            'sync_buffer_seconds': 2.0,
            'inactivity_threshold_seconds': 2.5,
            'semantic_export_directory': LaunchConfiguration('buffers_directory'),
            'mapping_config_path': '',
            'nominal_path': '/root/AirStack/robot/ros_ws/src/autonomy/4_global/a_world_models/resilience/assets/adjusted_nominal_spline.json',
            'main_config_path': ''
        }],
        remappings=[
            ('/semantic_hotspots', '/semantic_hotspots'),
            ('/semantic_hotspot_mask', '/semantic_hotspot_mask'),
            ('/vdb_occupancy_cloud', '/vdb_occupancy_cloud'),
            ('/vdb_semantic_cloud', '/vdb_semantic_cloud'),
            ('/vdb_stats', '/vdb_stats')
        ]
    )
    
    # Narration Display Node
    vlm_node = Node(
        package='resilience',
        executable='vlm_node.py',
        name='vlm_node',
        output='screen',
        parameters=[{
            'vlm_api_key': LaunchConfiguration('vlm_api_key'),
            'vlm_base_url': LaunchConfiguration('vlm_base_url'),
            'vlm_model': LaunchConfiguration('vlm_model'),
            'save_images': LaunchConfiguration('save_images')
        }],
        remappings=[
            ('/narration_image', '/narration_image'),
            ('/narration_text', '/narration_text'),
            ('/vlm_answer', '/vlm_answer')
        ]
    )

    
    
    return LaunchDescription([
        # Main resilience node arguments
        declare_flip_y_axis,
        declare_use_tf,
        declare_image_rotation_deg,
        declare_radio_model_version,
        declare_radio_lang_model,
        declare_radio_input_resolution,
        declare_enable_naradio_visualization,
        declare_enable_combined_segmentation,
        declare_enable_voxel_mapping,
        declare_pose_is_base_link,
        
        # VDB semantic mapping node arguments
        declare_max_range,
        declare_voxel_resolution,
        declare_min_range,
        declare_enable_semantic_mapping,
        declare_semantic_similarity_threshold,
        declare_buffers_directory,
        declare_publish_occupancy_cloud,
        declare_publish_semantic_cloud,
        declare_max_cloud_points,
        declare_cloud_publish_rate,
        declare_stats_publish_rate,
        
        # Narration display node arguments
        declare_vlm_api_key,
        declare_vlm_base_url,
        declare_vlm_model,
        declare_save_images,
        
        # Launch all nodes
        main_resilience_node,
        vdb_semantic_mapping_node,
        vlm_node
    ])
