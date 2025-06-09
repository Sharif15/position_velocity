## File used to generate extrinsics for the camera using a single apriltag at origin (0,0,0)
## Auto-saves extrinsics when tag is detected - Now with YAML configuration

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import yaml
import os
import apriltag
import time

from std_msgs.msg import Float64MultiArray


class ConfigurableAprilTagCalibrator(Node):
    def __init__(self, config_path):
        super().__init__('configurable_apriltag_calibrator')
        
        if not config_path or not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
        # Load configuration
        self.config_path = config_path
        self.load_config(config_path)
        
        self.calibration_complete = False
        self.extrinsics_saved = False
        self.detection_count = 0
        self.pose_samples = []  # Store multiple pose samples for averaging
        
        self.subscription = self.create_subscription(
            CompressedImage,
            self.config['camera']['topic'],
            self.tag_detection,
            10
        )
        
        # Load intrinsics from YAML
        try:
            with open(self.config['paths']['intrinsics'], 'r') as f:
                calib = yaml.safe_load(f)
            self.camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
            self.dist_coeffs = np.array(calib['distortion_coefficients'])
            self.get_logger().info(f"Loaded intrinsics from: {self.config['paths']['intrinsics']}")
        except Exception as e:
            self.get_logger().error(f"Failed to load intrinsics: {e}")
            raise

        # Define 3D world coordinates of the single tag at origin
        half_size = self.config['apriltag']['size'] / 2.0
        self.tag_3d_points = np.array([
            [-half_size, half_size, 0.0],  # top-left corner , 0
            [ half_size, half_size, 0.0],  # top-right corner, 1
            [ half_size,  -half_size, 0.0],  # bottom-right corner , 2
            [-half_size,  -half_size, 0.0]   # bottom-left corner, 3
        ], dtype=np.float32)

        # Setup AprilTag detector with configurable options
        options = apriltag.DetectorOptions(
            families=self.config['apriltag']['family'],
            border=self.config['detection']['border'],
            nthreads=self.config['detection']['nthreads'],
            quad_decimate=self.config['detection']['quad_decimate'],
            quad_blur=self.config['detection']['quad_blur'],
            refine_edges=self.config['detection']['refine_edges'],
            refine_decode=self.config['detection']['refine_decode'],
            refine_pose=self.config['detection']['refine_pose'],
            debug=self.config['detection']['debug'],
            quad_contours=self.config['detection']['quad_contours']
        )
        self.at_detector = apriltag.Detector(options)
        
        self.print_config_summary()

    def load_config(self, config_path=None):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
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

    def print_config_summary(self):
        """Print configuration summary"""
        self.get_logger().info("=== CONFIGURATION SUMMARY ===")
        self.get_logger().info(f"Target AprilTag ID: {self.config['apriltag']['target_id']}")
        self.get_logger().info(f"Tag size: {self.config['apriltag']['size']} meters")
        self.get_logger().info(f"Tag family: {self.config['apriltag']['family']}")
        self.get_logger().info(f"Camera topic: {self.config['camera']['topic']}")
        self.get_logger().info(f"Target resolution: {self.config['camera']['target_width']}x{self.config['camera']['target_height']}")
        self.get_logger().info(f"Required stable detections: {self.config['detection']['required_stable_detections']}")
        self.get_logger().info(f"Intrinsics path: {self.config['paths']['intrinsics']}")
        self.get_logger().info(f"Extrinsics path: {self.config['paths']['extrinsics']}")
        self.get_logger().info("Searching for tag... (Press 'q' to quit manually)")

    def tag_detection(self, msg):
        if self.calibration_complete:
            return
        
        # Add frame counter to reduce processing load
        if not hasattr(self, 'frame_count'):
            self.frame_count = 0
        self.frame_count += 1
        
        # Process every Nth frame based on config
        if self.frame_count % self.config['camera']['process_every_n_frames'] != 0:
            return
            
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Failed to decode image.")
            return

        if frame.size == 0:
            self.get_logger().warn("Empty image received.")
            return

        # Resize frame based on config
        height, width = frame.shape[:2]
        target_width = self.config['camera']['target_width']
        target_height = self.config['camera']['target_height']
        
        if width != target_width or height != target_height:
            # Calculate scale to maintain aspect ratio
            scale_w = target_width / width
            scale_h = target_height / height
            scale = min(scale_w, scale_h)  # Use smaller scale to fit within target
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
            
            # Scale camera matrix accordingly
            scaled_camera_matrix = self.camera_matrix.copy()
            scaled_camera_matrix[0, :] *= scale  # fx and cx
            scaled_camera_matrix[1, :] *= scale  # fy and cy
            print("scaled frame")
            self.camera_matrix = scaled_camera_matrix


        else:
            scaled_camera_matrix = self.camera_matrix


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing based on config
        if self.config['preprocessing']['enable_blur']:
            kernel_size = self.config['preprocessing']['gaussian_blur_kernel']
            gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # AprilTag detection
        try:
            tags = self.at_detector.detect(gray)
        except Exception as e:
            self.get_logger().error(f"AprilTag detection failed: {e}")
            cv2.putText(frame, "Detection Error - Check lighting/focus", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # display_fram = cv2.resize(frame, (960, 540))
            cv2.imshow("Configurable AprilTag Calibration", frame)
            cv2.waitKey(1)
            return
        
        # Find target tag
        target_tag = None
        for tag in tags:
            if tag.tag_id == self.config['apriltag']['target_id']:
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
                self.dist_coeffs,
                flags = cv2.SOLVEPNP_IPPE_SQUARE

            )
            
            if success:
                # Store pose sample for averaging
                self.pose_samples.append({
                    'rvec': rvec.copy(),
                    'tvec': tvec.copy(),
                    'timestamp': time.time()
                })
                
                # Keep only recent samples
                current_time = time.time()
                window_size = self.config['detection']['stability_window_seconds']
                self.pose_samples = [
                    sample for sample in self.pose_samples 
                    if current_time - sample['timestamp'] < window_size
                ]
                
                self.detection_count += 1
                
                # Draw coordinate axes
                axis_points = np.array([
                    [0, 0, 0],           # Origin
                    [0.05, 0, 0],        # X-axis (red)
                    [0, 0.05, 0],        # Y-axis (green)  
                    [0, 0, 0.05]        # Z-axis (blue, pointing up from tag)
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
                translation = tvec.flatten()
                distance = np.linalg.norm(translation)
                
                pose_text = [
                    f"Distance: {distance:.3f}m",
                    f"X: {translation[0]:.3f}m",
                    f"Y: {translation[1]:.3f}m", 
                    f"Z: {translation[2]:.3f}m",
                    f"Detections: {len(self.pose_samples)}/{self.config['detection']['required_stable_detections']}"
                ]

                colors = [
                    (0, 255, 255),    # Distance: Yellow
                    (0, 0, 255),      # X: Red
                    (0, 255, 0),      # Y: Green
                    (255, 0, 0),      # Z: Blue
                    (255, 255, 255)   # Counter: White
                ]
                
                for i, (text, color) in enumerate(zip(pose_text, colors)):
                    cv2.putText(frame, text, (10, 30 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Show progress bar for detection stability
                required_detections = self.config['detection']['required_stable_detections']
                progress = min(len(self.pose_samples), required_detections)
                bar_width = 200
                bar_height = 20
                cv2.rectangle(frame, (10, 160), (10 + bar_width, 160 + bar_height), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 160), (10 + int(bar_width * progress / required_detections), 160 + bar_height), (0, 255, 0), -1)
                cv2.putText(frame, f"Stability: {progress}/{required_detections}", (10, 155), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if self.extrinsics_saved:
                    cv2.putText(frame, "CALIBRATION COMPLETE - Press 'q' to quit", (10, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "Extrinsics saved successfully!", (10, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


                # Auto-save when we have enough stable detections

                if len(self.pose_samples) >= required_detections and not self.extrinsics_saved:
                    self.get_logger().info("Stable detection achieved! Auto-saving extrinsics...")
                    self.save_averaged_extrinsics()
                    self.extrinsics_saved = True  # Set flag instead of calibration_complete

                # key = cv2.waitKey(1) & 0xFF
                # if key == ord('q'):
                #     self.get_logger().info("Stable detection achieved! Auto-saving extrinsics...")
                #     self.save_averaged_extrinsics()
                #     self.calibration_complete = True
                    
        else:
            # Reset detection count if no tag found
            self.detection_count = 0
            self.pose_samples.clear()
            cv2.putText(frame, f"Searching for tag ID: {self.config['apriltag']['target_id']}...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Please ensure tag is visible and well-lit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)

        # display is only for testing purposes will not be needed in the final product 

        display_frame = frame.copy()
        display_frame = cv2.resize(display_frame, (960, 540))
        cv2.imshow("Configurable AprilTag Calibration", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if self.extrinsics_saved:
            time.sleep(2.5) # Wait for 2.5 seconds 
            self.get_logger().info("Quitting calibration")
            self.calibration_complete = True
            cv2.destroyAllWindows()

    def save_averaged_extrinsics(self):
        if len(self.pose_samples) == 0:
            self.get_logger().warn("No valid pose samples to save")
            return
        
        # Average the rotation vectors and translation vectors
        rvecs = np.array([sample['rvec'].flatten() for sample in self.pose_samples])
        tvecs = np.array([sample['tvec'].flatten() for sample in self.pose_samples])
        
        avg_rvec = np.mean(rvecs, axis=0).reshape(-1, 1)
        avg_tvec = np.mean(tvecs, axis=0).reshape(-1, 1)
        
        # Convert averaged rotation vector to rotation matrix
        avg_rotation_matrix, _ = cv2.Rodrigues(avg_rvec)
        
        # Create transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = avg_rotation_matrix
        transformation[:3, 3] = avg_tvec.flatten()
        
        # Calculate standard deviations for quality assessment
        rvec_std = np.std(rvecs, axis=0)
        tvec_std = np.std(tvecs, axis=0)
        
        # Save extrinsics data including configuration used
        data = {
            'calibration_info': {
                'target_tag_id': self.config['apriltag']['target_id'],
                'tag_size': self.config['apriltag']['size'],
                'tag_family': self.config['apriltag']['family'],
                'num_samples': len(self.pose_samples),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'pose_data': {
                'rotation_vector': avg_rvec.flatten().tolist(),
                'translation_vector': avg_tvec.flatten().tolist(),
                'rotation_matrix': avg_rotation_matrix.tolist(),
                'transformation_matrix': transformation.tolist(),
                # 'scaled_intrinsics' : self.scaled_camera_matrix.flatten().tolist(),
                'camera_to_world_transform': transformation.tolist()
            },
            'quality_metrics': {
                'rotation_std': rvec_std.tolist(),
                'translation_std': tvec_std.tolist(),
                'max_translation_std': float(np.max(tvec_std)),
                'max_rotation_std': float(np.max(rvec_std))
            },
            'configuration_used': self.config
        }

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config['paths']['extrinsics']), exist_ok=True)
            
            with open(self.config['paths']['extrinsics'], 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
            
            self.get_logger().info("=== EXTRINSICS CALIBRATION COMPLETE ===")
            self.get_logger().info(f"Extrinsics saved to: {self.config['paths']['extrinsics']}")
            self.get_logger().info(f"Based on {len(self.pose_samples)} stable detections")
            self.get_logger().info(f"Average translation: {avg_tvec.flatten()}")
            self.get_logger().info(f"Distance to tag: {np.linalg.norm(avg_tvec):.3f}m")
            self.get_logger().info(f"Translation stability (std): {tvec_std}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save extrinsics: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    # Optional: specify custom config path
    # Will use default locations if None
    #Currently set to get the config file from shared directory
    config_path = os.path.join(get_package_share_directory('position_velocity'), 'config', 'apriltag_calibration_config.yaml')
    
    try:
        node = ConfigurableAprilTagCalibrator(config_path)
        
        node.get_logger().info("Starting configurable auto-calibration...")
        node.get_logger().info("Position the AprilTag in view of the camera")
        
        # Process messages until calibration is complete or interrupted
        while rclpy.ok() and not node.calibration_complete:
            rclpy.spin_once(node, timeout_sec=0.1)
            
    except KeyboardInterrupt:
        node.get_logger().info("Calibration interrupted by user")
    except Exception as e:
        node.get_logger().error(f"Unhandled exception: {e}")
    finally:
        cv2.destroyAllWindows()
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()