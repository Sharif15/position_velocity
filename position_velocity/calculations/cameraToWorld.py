import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ptz_videography_interfaces.msg import Detections
from ament_index_python.packages import get_package_share_directory


import cv2
import numpy as np
import yaml
import os

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


        #Subscribe to the Detections topic for the pixel (x,y,id)

        self.coords_subscriber = self.create_subscription(
            Detections,
            'tracking_coords',
            self.position_calculation,
            10
        )

    # Used to calculate the realworld position from pixel data 

    def position_calculation(self, msg):

        try:
            print("=== Starting position_calculation ===")
            
            u_array = msg.x
            v_array = msg.y

            str_ids = msg.str_id if hasattr(msg, 'str_id') else []
            
            print(f"Number of detections: {len(u_array)}")
                        
            for i in range(len(u_array)):
                try:
                    u = float(u_array[i])
                    v = float(v_array[i])

                    str_id = str_ids[i] if i < len(str_ids) else f"item_{i}"

                    print(f"\n--- Processing detection {i}: for object {str_id} at ({u:.1f}, {v:.1f}) ---")

                    result = np.array(self.pixel_to_world(u,v))

                    print(f"World coords: {result}")

                    if result is None:
                        continue

                    reprojection = np.array(self.project_world_to_pixel(result))

                    print(f"Reprojection: {reprojection}")
                      
                except Exception as detection_error:
                    print(f"ERROR processing detection {i}: {detection_error}")
                    continue

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

    def load_extrinsics(self, yaml_path):

        try:
            print(f"Loading extrinsics from: {yaml_path}")
            with open(yaml_path, 'r') as file:
                extrinsics_data = yaml.safe_load(file)

            # Get rotation matrix and translation vector from pose_data
            r_raw = extrinsics_data['pose_data']['rotation_matrix']
            t_raw = extrinsics_data['pose_data']['translation_vector']

            print(f"Raw Rotation matrix: {r_raw}")
            print(f"Raw Translation vector: {t_raw}")
            print(f"Type of r_raw: {type(r_raw)}")
            print(f"Type of t_raw: {type(t_raw)}")

            # Convert rotation matrix to numpy array
            print("Converting rotation matrix to numpy array...")
            R = np.array(r_raw, dtype=np.float64)
            print(f"R shape after conversion: {R.shape}")
            
            if R.shape != (3, 3):
                raise ValueError(f"Expected 3x3 rotation matrix, got shape {R.shape}")

            # Convert translation vector to numpy array
            print("Converting translation vector to numpy array...")
            t = np.array(t_raw, dtype=np.float64).reshape(3, 1)
            print(f"t shape after conversion: {t.shape}")

            return R, t
            
        except Exception as e:
            print(f"ERROR in load_extrinsics: {e}")
            import traceback
            traceback.print_exc()
            raise
 
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