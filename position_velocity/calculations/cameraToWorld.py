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

                    result_1 = self.pixel_to_world_1(u,v)

                    result_2 = self.pixel_to_world_2(u,v)
                    result_3 = self.pixel_to_world_coordinates(u, v)
                    print(f"Rotation matrix : {self.R}")

                    print(f"translation matrix : {self.t}")

                    print(f"World coords: {result}")

                    print(f"World Coords_1 : {result_1}")

                    print(f"World Coords_2 : {result_2}")
                    print(f"World Coords_3 : {result_3}")
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

    def compute_depth_to_plane(self, u, v):
        """
        Compute depth scale 's' for pixel (u, v) assuming the point lies on the plane Z = 0.
        """
        pixel_homog = np.array([[u], [v], [1]])
        ray_cam = self.K_inv @ pixel_homog
        ray_world = self.R_inv @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        cam_world = self.cam_position.flatten()
        ray = ray_world.flatten()

        if ray[2] == 0:
            raise ValueError("Ray is parallel to the ground plane")

        s = -cam_world[2] / ray[2]  # intersection with Z = 0
        return s, cam_world, ray

    def pixel_to_world_coordinates(self, u, v):
        """
        Convert pixel (u, v) to world coordinates assuming the point lies on the ground plane Z = 0.
        """
        s, cam_origin, ray = self.compute_depth_to_plane(u, v)
        point_world = cam_origin + s * ray
        x, y = point_world[0], point_world[1]  # return x, y only
        return abs(x) - abs(self.cam_position[0]), abs(y) - abs(self.cam_position[1])

    
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
    def pixel_to_world_1(self, u, v):
        
        image_mat = [u, v, 1]

        X_cp = (u - self.scaled_cam[0][2])/self.scaled_cam[0][0] # what is the scaled_cam 
        Y_cp = (v - self.scaled_cam[1][2])/self.scaled_cam[1][1]

        X_c = X_cp
        Y_c = Y_cp
        r = self.R
        #X_w, Y_w
        eqs = np.array([
            [r[0][0], r[0][1]], #r00*X_w + #r01Y_w + tx = X_c
            [r[1][0], r[1][1]], #r10*X_w + #r11Y_w + ty = Y_c 
        ])
        sols = np.array([X_c-self.t[0][0], Y_c-self.t[1][0]])

        result = np.linalg.solve(eqs,sols)

        print(result)

        return result


    # Third attempt at this function 

    def pixel_to_world_2(self,u,v):

        pixel_vec = np.array([u, v, 1])  # homogeneous pixel vector

        Rt = np.hstack((self.R, self.t))   # 3x4
        A = self.K @ Rt                    # 3x4

        # Extract projection matrix columns
        a1 = A[:, 0].reshape(3, 1)
        a2 = A[:, 1].reshape(3, 1)
        a3 = A[:, 2].reshape(3, 1)
        a4 = A[:, 3].reshape(3, 1)

        B = np.hstack((a1, a2))            # 3x2
        C = pixel_vec.reshape(3, 1) - a4 -a3  # 3x1

        # Least-squares solution
        world_coords = np.linalg.inv(B.T @ B) @ (B.T @ C)  # 2x1

        x, y = float(world_coords[0])/100, float(world_coords[1])/100
        return x, y

    
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