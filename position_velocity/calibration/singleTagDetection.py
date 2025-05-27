## File used to generate extrinsics for the camera using a single apriltag at origin (0,0,0)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import yaml
import os
import apriltag


from std_msgs.msg import Float64MultiArray



class SingleAprilTagCalibrator(Node):
    def __init__(self, target_tag_id=0, tag_size=0.1):
        super().__init__('single_apriltag_calibrator')
        
        self.target_tag_id = target_tag_id
        self.tag_size = tag_size  # Physical size of the tag in meters
        self.calibration_complete = False
        
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.tag_detection,
            10
        )
        
        # Get absolute paths to data files
        pkg_share = get_package_share_directory('position_velocity')
        intrinsics_path = os.path.join(pkg_share, 'data', 'intrinsics.yaml')
        self.extrinsics_path = os.path.join(pkg_share, 'data', 'extrinsics.yaml')

        # Load intrinsics from YAML
        with open(intrinsics_path, 'r') as f:
            calib = yaml.safe_load(f)
        self.camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(calib['distortion_coefficients'])

        # Define 3D world coordinates of the single tag at origin
        # Tag corners in 3D world coordinates (tag at origin, lying flat on XY plane)
        half_size = self.tag_size / 2.0
        self.tag_3d_points = np.array([
            [-half_size, -half_size, 0.0],  # Bottom-left corner
            [ half_size, -half_size, 0.0],  # Bottom-right corner  
            [ half_size,  half_size, 0.0],  # Top-right corner
            [-half_size,  half_size, 0.0]   # Top-left corner
        ], dtype=np.float32)

        # Use apriltag detector with more conservative options
        options = apriltag.DetectorOptions(
            families="tag36h11",
            border=1,
            nthreads=1,
            quad_decimate=2.0,
            quad_blur=0.0,
            refine_edges=True,
            refine_decode=False,
            refine_pose=False,
            debug=False,
            quad_contours=True
        )
        self.at_detector = apriltag.Detector(options)
        
        self.get_logger().info(f"Looking for AprilTag ID: {self.target_tag_id}")
        self.get_logger().info(f"Tag size: {self.tag_size} meters")
        self.get_logger().info("Press 'q' to quit, 's' to save current detection")

    def tag_detection(self, msg):

        self.pose_pub = self.create_publisher(Float64MultiArray, '/apriltag/pose_tvec', 10)

        if self.calibration_complete:
            return
        
        # Add frame counter to reduce processing load
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        
        # Process every 3rd frame to reduce load
        if self.frame_count % 3 != 0:
            return
            
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode image.")
            return

        if frame.size == 0:
            self.get_logger().warn("Empty image received.")
            return

        # Resize frame if too large to reduce processing
        height, width = frame.shape[:2]
        if width > 640:
            scale = 640.0 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            # Scale camera matrix accordingly
            scaled_camera_matrix = self.camera_matrix.copy()
            scaled_camera_matrix[0, :] *= scale  # fx and cx
            scaled_camera_matrix[1, :] *= scale  # fy and cy
        else:
            scaled_camera_matrix = self.camera_matrix

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply some preprocessing to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Add try-catch for detection to handle segfaults
        try:
            tags = self.at_detector.detect(gray)
        except Exception as e:
            self.get_logger().error(f"AprilTag detection failed: {e}")
            cv2.putText(frame, "Detection Error - Check lighting/focus", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Single AprilTag Calibration", frame)
            cv2.waitKey(1)
            return
        
        # Limit number of tags to prevent overflow
        if len(tags) > 10:
            self.get_logger().warn(f"Too many tags detected: {len(tags)}. Using first 10.")
            tags = tags[:10]

        target_tag = None
        for tag in tags:
            if tag.tag_id == self.target_tag_id:
                target_tag = tag
                break

        if target_tag is not None:
            # Get the corner points of the detected tag
            corners = target_tag.corners.astype(np.float32)
            
            # Draw the tag detection
            cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)
            
            # Draw corner points
            for i, corner in enumerate(corners):
                cv2.circle(frame, (int(corner[0]), int(corner[1])), 5, (255, 0, 0), -1)
                cv2.putText(frame, str(i), (int(corner[0])+10, int(corner[1])+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw center and ID
            center = target_tag.center
            cv2.circle(frame, (int(center[0]), int(center[1])), 3, (0, 0, 255), -1)
            cv2.putText(frame, f'ID:{target_tag.tag_id}', 
                       (int(center[0]), int(center[1]) - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate pose using solvePnP
            success, rvec, tvec = cv2.solvePnP(
                self.tag_3d_points,
                corners,
                scaled_camera_matrix,
                self.dist_coeffs
            )
            
            if success:
                # Draw coordinate axes on the tag
                axis_points = np.array([
                    [0, 0, 0],           # Origin
                    [0.05, 0, 0],        # X-axis (red)
                    [0, 0.05, 0],        # Y-axis (green)  
                    [0, 0, -0.05]        # Z-axis (blue, pointing up from tag)
                ], dtype=np.float32)
                
                axis_img_points, _ = cv2.projectPoints(
                    axis_points, rvec, tvec, scaled_camera_matrix, self.dist_coeffs
                )
                
                axis_img_points = axis_img_points.reshape(-1, 2).astype(int)
                
                # Draw axes
                cv2.arrowedLine(frame, tuple(axis_img_points[0]), tuple(axis_img_points[1]), (0, 0, 255), 3)  # X-axis (red)
                cv2.arrowedLine(frame, tuple(axis_img_points[0]), tuple(axis_img_points[2]), (0, 255, 0), 3)  # Y-axis (green)
                cv2.arrowedLine(frame, tuple(axis_img_points[0]), tuple(axis_img_points[3]), (255, 0, 0), 3)  # Z-axis (blue)
                
                # Display pose information
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Convert translation to more readable format
                translation = tvec.flatten()
                distance = np.linalg.norm(translation)
                
                pose_text = [
                    f"Distance: {distance:.3f}m",
                    f"X: {translation[0]:.3f}m",
                    f"Y: {translation[1]:.3f}m", 
                    f"Z: {translation[2]:.3f}m"
                ]

                            # Define colors for each measurement (BGR format)
                colors = [
                    (0, 0, 255),    # Distance: Yellow
                    (0, 0, 255),      # X: Red
                    (0, 255, 0),      # Y: Green
                    (255, 0, 0)       # Z: Blue
                ]
                
                for i, (text,color) in enumerate(zip(pose_text,colors)):
                    cv2.putText(frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6,color, 2)
                
                cv2.putText(frame, "Press 's' to save extrinsics", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Store the current pose for potential saving
                self.current_rvec = rvec
                self.current_tvec = tvec
                self.current_rotation_matrix = rotation_matrix
                
        else:
            cv2.putText(frame, f"Looking for tag ID: {self.target_tag_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Single AprilTag Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quitting calibration")
            self.calibration_complete = True
            cv2.destroyAllWindows()
        elif key == ord('s') and target_tag is not None:
            self.save_extrinsics()


    def save_extrinsics(self):
        if not hasattr(self, 'current_rvec'):
            self.get_logger().warn("No valid pose detected to save")
            return
            
        # Create transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = self.current_rotation_matrix
        transformation[:3, 3] = self.current_tvec.flatten()
        
        # Save extrinsics data
        data = {
            'target_tag_id': self.target_tag_id,
            'tag_size': self.tag_size,
            'rotation_vector': self.current_rvec.flatten().tolist(),
            'translation_vector': self.current_tvec.flatten().tolist(),
            'rotation_matrix': self.current_rotation_matrix.tolist(),
            'transformation_matrix': transformation.tolist(),
            'camera_to_world_transform': transformation.tolist()
        }

        with open(self.extrinsics_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        self.get_logger().info("Extrinsics saved to extrinsics.yaml")
        self.get_logger().info(f"Translation: {self.current_tvec.flatten()}")
        self.get_logger().info(f"Distance to tag: {np.linalg.norm(self.current_tvec):.3f}m")


def main(args=None):
    rclpy.init(args=args)
    
    # You can modify these parameters:
    target_tag_id = 0      # ID of the AprilTag to use as origin
    tag_size = 0.2         # Physical size of the tag in meters
    
    node = SingleAprilTagCalibrator(target_tag_id, tag_size)
    
    try:
        # Process messages until calibration is complete or interrupted
        while rclpy.ok() and not node.calibration_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Calibration interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()