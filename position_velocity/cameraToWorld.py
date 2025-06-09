import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ptz_videography_interfaces.msg import Detections
from ptz_videography_interfaces.msg import Coordinates

from ament_index_python.packages import get_package_share_directory

# importing the velocity calculation function 

from position_velocity.velocityCalculation import VelocityTracker


import cv2
import numpy as np
import yaml
import os

import time

class PixelConverter(Node):

    def __init__(self,config_path):
        
        super().__init__('pixel_converter')

        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        # Load configuration
        self.config_path = config_path
        self.load_config(config_path)

        # load the camera parameters 
        self.K, self.D = self.load_camera_parameters(self.config['paths']['intrinsics'])
        self.R, self.t = self.load_extrinsics(self.config['paths']['extrinsics'])

        self.scaled_cam = self.K
        self.K_inv = np.linalg.inv(self.K)
        self.R_inv = np.linalg.inv(self.R)
        # Calculate camera position in real world camera_position = -R_inverse * t

        self.cam_position = -self.R.T @ self.t.reshape(3, 1)
        
        self.get_logger().info(f"Camera position in world coords: {self.cam_position.flatten()}")
        self.get_logger().info("Camera parameters loaded successfully")

        # Make a publisher for the world coordinates 
        self.position_publisher = self.create_publisher(Coordinates, 'world_coordinates', 10)

        #Subscribe to the Detections topic for the pixel (x,y,id)
        self.coords_subscriber = self.create_subscription(
            Detections,
            'tracking_coords',
            self.position_calculation,
            10
        )

        # initialize the velocity calculation function 

        # self.velocity_tracker = VelocityTracker()


    # Used to calculate the realworld position from pixel data 

    def position_calculation(self, msg):

        try:
            print("=== Starting position_calculation ===")
            
            u_array = msg.x
            v_array = msg.y

            str_ids = msg.str_id if hasattr(msg, 'str_id') else []
            
            print(f"Number of detections: {len(u_array)}")

            # Build the message array to publish the world coords of objects 

            world_coords_msg = Coordinates()

            world_coords_msg.str_id = []

            world_coords_msg.x = []

            world_coords_msg.y = []

            world_coords_msg.z = []
                        
            for i in range(len(u_array)):
                try:
                    u = float(u_array[i])
                    v = float(v_array[i])

                    str_id = str_ids[i] if i < len(str_ids) else f"item_{i}"

                    print(f"\n--- Processing detection {i}: for object {str_id} at ({u:.1f}, {v:.1f}) ---")

                    result = np.array(self.pixel_to_world(u,v))

                    print(f"World coords: {result}")

                    # Add the calculated position to the messege 

                    world_coords_msg.str.append(str_id)
                    world_coords_msg.x.append(result[0])
                    world_coords_msg.y.append(result[1])
                    world_coords_msg.z.append(result[2])

                    # Do velocity calculation for the given position 

                    # object_velocity = self.velocity_tracker.update_position(result[:2])  # Only x, y
                    # print(f"Smoothed velocity: {object_velocity}")

                    if result is None:
                        continue

                    # delete the reprojection code for the final product 

                    reprojection = np.array(self.project_world_to_pixel(result))

                    print(f"Reprojection: {reprojection}")
                      
                except Exception as detection_error:
                    print(f"ERROR processing detection {i}: {detection_error}")
                    continue

            # publish the messege 
            self.position_publisher.publish(world_coords_msg)


        except Exception as e:
            print(f"ERROR in position_calculation: {e}")
            import traceback
            traceback.print_exc()

    # Helper function for position_calculation

    def load_camera_parameters(self, yaml_path):
        with open(yaml_path, 'r') as file:
            cam_data = yaml.safe_load(file)
        K = np.array(cam_data['camera_matrix']['data']).reshape(3, 3)
        D = np.array(cam_data['distortion_coefficients'])
        return K, D

    def load_extrinsics(self, yaml_path, max_wait_seconds=10, retry_interval=0.5):
       
        """Retry loading extrinsics until valid or timeout."""

        start_time = time.time()

        while True:
            try:
                if not os.path.exists(yaml_path):
                    raise FileNotFoundError(f"{yaml_path} does not exist yet.")

                with open(yaml_path, 'r') as file:
                    extrinsics_data = yaml.safe_load(file)

                if not extrinsics_data or 'pose_data' not in extrinsics_data:
                    raise ValueError(f"Invalid or incomplete extrinsics data in {yaml_path}")

                r_raw = extrinsics_data['pose_data']['rotation_matrix']
                t_raw = extrinsics_data['pose_data']['translation_vector']

                R = np.array(r_raw, dtype=np.float64)
                if R.shape != (3, 3):
                    raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")

                t = np.array(t_raw, dtype=np.float64).reshape(3, 1)

                self.get_logger().info("Extrinsics loaded successfully.")
                return R, t

            except Exception as e:
                elapsed = time.time() - start_time
                if elapsed >= max_wait_seconds:
                    self.get_logger().error(f"Timeout while waiting for valid extrinsics: {e}")
                    raise  # Let the node crash if it's never valid

                self.get_logger().warn(f"Waiting for valid extrinsics file... retrying in {retry_interval}s")
                time.sleep(retry_interval)


    # function that does the calculation for camera pixel to world conversion 
    def pixel_to_world(self, u, v, Z_world=0):
            # Build pixel homogeneous coordinate
            pixel = np.array([u, v, 1.0]).reshape(3, 1)

            # Inverse camera intrinsics
            inv_K = np.linalg.inv(self.K)

            # Get direction vector in camera frame
            cam_ray = inv_K @ pixel  # now a 3x1 vector

            # Convert to world coordinates
            ray_world = -self.R.T @ cam_ray
            cam_center_world = -self.R.T @ self.t

            print("Camera center in world coordinates:", self.cam_position.flatten())


            # Solve for scale factor s such that point lies on plane Z=Z_world
            s = (Z_world - self.cam_position[2, 0]) / ray_world[2, 0]

            world_point = self.cam_position + s * ray_world

            return world_point.flatten()

    '''
    Function used for verifying the accuracy of back projection. 
    
    Using the 3D world points we calculate as an input to go back to the original 2D pixel coordinates 
    
    Delete for final product 

    '''
    def project_world_to_pixel(self, point_3d_world):

        try:
            # Ensure shape (3,1)
            if isinstance(point_3d_world, list):
                point_3d_world = np.array(point_3d_world)
            if point_3d_world.shape == (3,):
                point_3d_world = point_3d_world.reshape((3, 1))

            # Transform world to camera frame. Point = ( R * [X,Y,Z] )+ t = a 

            point_camera = self.R @ point_3d_world + self.t

            # Project to image plane. ray_in_cam = K * point_cam . Which gives [u*z,v*z,z]

            point_proj = self.K @ point_camera

            # Normalize to get pixel coordinates. devide by z to get the 2D coords [u,v,0]
            u = point_proj[0, 0] / point_proj[2, 0]
            v = point_proj[1, 0] / point_proj[2, 0]

            return u, v

        except Exception as e:
            self.get_logger().error(f"Error in project_world_to_pixel: {e}")
            return None

    # Handels configuration loading 

    def load_config(self, config_path=None):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    
        if not self.config:
            raise ValueError(f"Failed to load configuration from {self.config_path}: File is empty or invalid YAML.")
        
        # Debug log to see what was loaded
        self.get_logger().info(f"Loaded config: {self.config}")
        
        # Ensure required keys exist
        if 'paths' not in self.config:
            raise KeyError("Missing 'paths' section in config.")
        if 'intrinsics' not in self.config['paths'] or 'extrinsics' not in self.config['paths']:
            raise KeyError("Missing 'intrinsics' or 'extrinsics' in config['paths'].")

        self.resolve_paths()
        self.get_logger().info(f"Configuration loaded from: {self.config_path}")

    # Get the desired path from the config file the intrensics and extrensics 
    def resolve_paths(self):
        """Resolve relative paths in configuration"""
        try:
            pkg_share = get_package_share_directory('position_velocity')
            
            # Resolve intrinsics path
            if not os.path.isabs(self.config['paths']['intrinsics']):
                self.config['paths']['intrinsics'] = os.path.join(pkg_share, 'data', self.config['paths']['intrinsics'])
            
            # Resolve extrinsics path
            if not os.path.isabs(self.config['paths']['extrinsics']):
                self.config['paths']['extrinsics'] = os.path.join(pkg_share, 'data', self.config['paths']['extrinsics'])
                
        except Exception as e:
            self.get_logger().warn(f"Could not resolve package paths: {e}")


def main(args=None):
    rclpy.init(args=args)

    config_path = os.path.join(get_package_share_directory('position_velocity'), 'config', 'position_calibration_config.yaml')

    try:
        pixel_converter_node = PixelConverter(config_path=config_path)
        rclpy.spin(pixel_converter_node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'pixel_converter_node' in locals():
            pixel_converter_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()