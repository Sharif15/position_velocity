import cv2
import numpy as np
import logging
import yaml
import os
import matplotlib
import rclpy
from rclpy.node import Node
from ptz_videography_interfaces.msg import Detections

class message():
    def __init__(self, x, y):
        self.x = x  
        self.y = y

class ObjectDetectorNode(Node):
    def __init__(self, config_path=None):
        super().__init__('object_detector_node')
        self.running = True
        
        # Set default config path if not provided
        if config_path is None:
            # Try to get package share directory, fallback to relative path
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg_share = get_package_share_directory('position_velocity')
                config_path = os.path.join(pkg_share, 'config', 'position_calibration_config.yaml')
            except:
                # Fallback to relative path if ROS package not found
                config_path = 'config/position_calibration_config.yaml'
        
        # Load configuration and resolve paths
        self.config_path = config_path
        self.load_config()
        
        # Load camera parameters using resolved paths
        self.load_camera_parameters()
        
        print(f"Camera parameters loaded successfully")
        print(f"Intrinsic matrix shape: {self.intrinsic_matrix.shape}")
        print(f"Rotation matrix shape: {self.rotation.shape}")
        print(f"Translation vector shape: {self.translation.shape}")
    
    def load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if not self.config:
            raise ValueError(f"Failed to load configuration from {self.config_path}: File is empty or invalid YAML.")
        
        # Ensure required keys exist
        if 'paths' not in self.config:
            raise KeyError("Missing 'paths' section in config.")
        if 'intrinsics' not in self.config['paths'] or 'extrinsics' not in self.config['paths']:
            raise KeyError("Missing 'intrinsics' or 'extrinsics' in config['paths'].")

        self.resolve_paths()
        print(f"Configuration loaded from: {self.config_path}")
    
    def resolve_paths(self):
        """Resolve relative paths in configuration"""
        try:
            # Try to get ROS package share directory
            from ament_index_python.packages import get_package_share_directory
            pkg_share = get_package_share_directory('position_velocity')
            
            # Resolve intrinsics path
            if not os.path.isabs(self.config['paths']['intrinsics']):
                self.config['paths']['intrinsics'] = os.path.join(pkg_share, 'data', self.config['paths']['intrinsics'])
            
            # Resolve extrinsics path
            if not os.path.isabs(self.config['paths']['extrinsics']):
                self.config['paths']['extrinsics'] = os.path.join(pkg_share, 'data', self.config['paths']['extrinsics'])
                
        except Exception as e:
            print(f"Could not resolve package paths: {e}")
            # Fallback to relative paths from current directory
            if not os.path.isabs(self.config['paths']['intrinsics']):
                self.config['paths']['intrinsics'] = os.path.join('data', self.config['paths']['intrinsics'])
            if not os.path.isabs(self.config['paths']['extrinsics']):
                self.config['paths']['extrinsics'] = os.path.join('data', self.config['paths']['extrinsics'])
    
    def load_camera_parameters(self):
        """Load camera intrinsics and extrinsics from resolved paths"""
        # Load intrinsics
        intrinsics_path = self.config['paths']['intrinsics']
        if not os.path.exists(intrinsics_path):
            raise FileNotFoundError(f"Intrinsics file not found at: {intrinsics_path}")
            
        with open(intrinsics_path, 'r') as f:
            intrin = yaml.safe_load(f)
        self.intrinsic_matrix = np.array(intrin['camera_matrix']['data']).reshape(3, 3)
        
        # Load extrinsics
        extrinsics_path = self.config['paths']['extrinsics']
        if not os.path.exists(extrinsics_path):
            raise FileNotFoundError(f"Extrinsics file not found at: {extrinsics_path}")
            
        with open(extrinsics_path, 'r') as f:
            extrin = yaml.safe_load(f)
        
        # Extract rotation and translation
        self.rotation = np.array(extrin['pose_data']['rotation_matrix'], dtype=np.float64)
        if self.rotation.shape != (3, 3):
            self.rotation = self.rotation.reshape(3, 3)
            
        self.translation = np.array(extrin['pose_data']['translation_vector'], dtype=np.float64).reshape(3, 1)
        
        # Handle scaled intrinsics if available
        if 'scaled_intrinsics' in extrin['pose_data']:
            self.scaled_cam = np.array(extrin['pose_data']['scaled_intrinsics'], dtype=np.float64)
            if self.scaled_cam.shape != (3, 3):
                self.scaled_cam = self.scaled_cam.reshape(3, 3)
        else:
            # Use regular intrinsics if scaled not available
            self.scaled_cam = self.intrinsic_matrix.copy()
        
        # Calculate camera position
        RC = np.matmul(np.negative(self.rotation), self.translation)      
        self.extrinsic_matrix = np.hstack((self.rotation, RC))
        
        print(f"Loaded intrinsics from: {intrinsics_path}")
        print(f"Loaded extrinsics from: {extrinsics_path}")
        
        # Subscribe to the Detections topic for pixel coordinates
        self.coords_subscriber = self.create_subscription(
            Detections,
            'tracking_coords',
            self.position_calculation_callback,
            10
        )
        self.get_logger().info("Subscribed to 'tracking_coords' topic")

    def position_calculation_callback(self, msg):
        """Callback function that processes incoming detection messages"""
        try:
            u_array = msg.x
            v_array = msg.y
            str_ids = msg.str_id if hasattr(msg, 'str_id') else []
            
            self.get_logger().info(f"Received {len(u_array)} detections")

            # RC = -R * T  (camera center in world coords, negated because of how projection matrices work)
            RC = -self.rotation @ self.translation
            self.extrinsic_matrix = np.hstack((self.rotation, RC))  # [R | -RC]


            # P = np.matmul(self.scaled_cam, self.extrinsic_matrix)

            
            for i in range(len(u_array)):
                u = float(u_array[i])
                v = float(v_array[i])
                str_id = str_ids[i] if i < len(str_ids) else f"item_{i}"
                
                # Create a message object for compatibility with existing calculate_position method
                
                detection_msg = message(u, v)
                self.calculate_position(detection_msg, str_id)
                
                world_coords = self.pixel_to_world(u, v, Z_world=0)
                print(f"Detection {str_id}: Pixel ({u:.1f}, {v:.1f}) -> World {world_coords}")

                homog_world = np.append(world_coords[:3], 1)

                P = self.scaled_cam @ np.hstack((self.rotation, self.translation))  # K[R|t]

                pixel_proj = P @ homog_world
                pixel_proj /= pixel_proj[2]
                print("Reprojection:", pixel_proj[:2])

                
        except Exception as e:
            self.get_logger().error(f"Error in position_calculation_callback: {e}")

    def calculate_position(self, msg, detection_id="unknown"):

        x_c = msg.x # x_c is x in camera pixel
        y_c = msg.y # y_c is y in camera pixel
        
        # The pixel coordinates for the detected object 
        image_pixel = [x_c, y_c, 1]

        # Normalize the [x_c,y_c,1] using the distortion matrix

        #P = np.matmul(self.intrinsic_matrix, self.extrinsic_matrix)

        P = self.scaled_cam @ np.hstack((self.rotation, self.translation))  # K[R|t]

        Ya = ((x_c*P[2][3] - P[0][3])/(P[0][0] - P[2][0]*x_c))*(P[1][0] - (y_c*P[2][0])) + P[1][3] - (y_c * P[2][3])
        
        Yb = ((x_c*P[2][1] - P[0][1])/(P[0][0] - P[2][0]*x_c))*((y_c*P[2][0]) - P[1][0]) + (y_c * P[2][1]) - P[1][1]
        
        Y = Ya/Yb # solved the world y 
        
        # Using world y to solve world x 

        X = ((P[2][1] * Y) + (P[2][3]*x_c) - (P[0][1] * Y) - P[0][3])/(P[0][0] - P[2][0]*x_c)
        
        Z = 0

        world_coords = [X, Y, Z, 1]

        print(f"Detection {detection_id}: World coordinates = {world_coords}")
       
        self.get_logger().info(f"Detection {detection_id}: Pixel ({msg.x:.1f}, {msg.y:.1f}) -> World ({X:.2f}, {Y:.2f}, {Z:.2f})")

        # Reversing the calculation 

        pixel_coords = np.matmul(P, np.asarray(world_coords))

        pixel_coords /= pixel_coords[2] # deviding by Z to get the x_c and y_c back 

        print("Reversal: ", pixel_coords)
        
        leftSideMat  = np.matmul(np.matmul(np.linalg.inv(self.rotation), np.linalg.inv(self.scaled_cam)), np.asarray(image_pixel).reshape(3,1))
        
        rightSideMat = np.matmul(np.linalg.inv(self.rotation), self.translation)
        
        print(leftSideMat.shape)
        print(rightSideMat.shape)
        
        s = (0 + rightSideMat[2][0])/leftSideMat[2][0]
        
        real_points =  np.matmul(np.linalg.inv(self.rotation), np.subtract((np.matmul(s*np.linalg.inv(self.scaled_cam), np.asarray(image_pixel))).reshape(3,1) ,self.translation).reshape(3,1))
        
        print("P = ", real_points)
        
        P_prime = (self.intrinsic_matrix @ np.hstack((self.rotation[:, :2], self.translation)))  # 3x3 matrix
        
        XY1 = np.linalg.lstsq(P_prime, np.asarray(image_pixel), rcond=None)[0]
        
        # Extract X and Y
        X_world, Y_world = XY1[0], XY1[1]
        Z_world = 0.0
        print(f"3D world coordinates: X={X_world}, Y={Y_world}, Z={Z_world}")
        K = np.matmul(self.scaled_cam, np.column_stack((self.rotation, self.translation)))
        
        print(K.shape)
        Y_right = (K[2][3]*msg.y) - K[1][3] - ((((K[2][3]*msg.x) - K[0][3])/(K[0][0] - (K[2][0]*msg.x))) * (K[1][0] - (msg.y*K[2][0])))
        Y_left = K[1][1] - (K[2][1]*msg.y) + ((((K[2][1]*msg.x) - K[0][1])/(K[0][0]- (K[2][0]*msg.x)))*(K[1][0] - (K[2][0]*msg.y)))
        new_Y = Y_right/Y_left
        new_X = (new_Y*((K[2][1]*msg.x) - K[0][1]) + (K[2][3]*msg.x) - K[0][3])/(K[0][0] - (K[2][0]*msg.x))
        s = new_X*K[2][0] + new_Y*K[2][1] + K[2][3]
        print(new_X, new_Y, s)
        newer_Y = (msg.y - K[1][3] + (K[1][0] * K[0][3]) - (K[1][0] * msg.x))/(K[1][1] - (K[0][1]*K[1][0]))
        newer_X = (msg.x - K[0][3] - (newer_Y*K[0][1]))
        print(newer_X, newer_Y)
    

    def pixel_to_world(self, u, v, Z_world=0):
        # Build pixel homogeneous coordinate
        pixel = np.array([u, v, 1.0]).reshape(3, 1)

        # Inverse camera intrinsics
        inv_K = np.linalg.inv(self.scaled_cam)

        # Get direction vector in camera frame
        cam_ray = inv_K @ pixel  # now a 3x1 vector

        # Convert to world coordinates
        ray_world = self.rotation.T @ cam_ray
        cam_center_world = -self.rotation.T @ self.translation

        print("Camera center in world coordinates:", cam_center_world.flatten())


        # Solve for scale factor s such that point lies on plane Z=Z_world
        s = (Z_world - cam_center_world[2, 0]) / ray_world[2, 0]

        world_point = cam_center_world + s * ray_world

        return world_point.flatten()


    def stop(self):
        self.running = False

def main(args=None):
    rclpy.init(args=args)
    
    try:
        # Try to use ROS package path first, fallback to relative path
        try:
            from ament_index_python.packages import get_package_share_directory
            config_path = os.path.join(get_package_share_directory('position_velocity'), 'config', 'position_calibration_config.yaml')
        except:
            config_path = 'config/position_calibration_config.yaml'
        
        node = ObjectDetectorNode(config_path=config_path)
        
        # For testing with a single message (comment out when using ROS topic)
        # msg = message(572, 495)
        # node.calculate_position(msg, "test_detection")
        
        # Spin to keep the node alive and process incoming messages
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()