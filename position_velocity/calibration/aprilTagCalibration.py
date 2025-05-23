## File used to generate extrensics for the camera using apriltags for calibraton

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from ament_index_python.packages import get_package_share_directory

import numpy as np
import cv2
import yaml
import os

class AprilTagCalibrator(Node):
    def __init__(self):
        super().__init__('apriltag_calibrator')

        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.tag_detection,
            10
        )
        
        # Get absolute paths to data files
        pkg_share = get_package_share_directory('position_velocity')
        intrinsics_path = os.path.join(pkg_share, 'data', 'intrinsics.yaml')
        self.extrinsics_path = os.path.join(pkg_share, 'data', 'camera_extrinsics.yaml')


        # Load intrinsics from YAML
        with open(intrinsics_path, 'r') as f:
            calib = yaml.safe_load(f)
        self.camera_matrix = np.array(calib['camera_matrix']['data']).reshape(3, 3)
        self.dist_coeffs = np.array(calib['distortion_coefficients'])

        # Define 3D world coordinates of tags (in meters)
        self.world_points = {
            0: np.array([0.0, 0.0, 0.0]),          # Bottom-left
            1: np.array([30.0, 0.0, 0.0]),         # Bottom-right
            2: np.array([30.0, 30.0, 0.0]),        # Top-right
            3: np.array([0.0, 30.0, 0.0]),         # Top-left
        }

        # Set up AprilTag detector
        # Currently using open Cv but could use Apriltag libray 
        self.at_detector = cv2.AprilTagDetector_create()
        self.at_detector.setAprilTagFamily("tag25h9") # can change the tag family depending on what tag was used 

    def tag_detection(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.AprilTagDetector_create()
        detector.setAprilTagFamily("tag36h11")
        tags = detector.detect(gray)

        img_points = []
        obj_points = []

        for tag in tags:
            tag_id = int(tag.getId())
            if tag_id in self.world_points:
                center = tag.getCenter()
                img_points.append(center)
                obj_points.append(self.world_points[tag_id])

                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'ID:{tag_id}', (int(center[0]), int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if len(img_points) >= 4:
            success, rvec, tvec = cv2.solvePnP(
                np.array(obj_points, dtype=np.float32),
                np.array(img_points, dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs
            )
            if success:
                self.get_logger().info("PnP solution found.")
                self.save_extrinsics(rvec, tvec)

        cv2.imshow("AprilTag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    def save_extrinsics(self, rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation = np.eye(4)
        transformation[:3, :3] = rotation_matrix
        transformation[:3, 3] = tvec.flatten()

        data = {
            'rotation_vector': rvec.flatten().tolist(),
            'translation_vector': tvec.flatten().tolist(),
            'rotation_matrix': rotation_matrix.tolist(),
            'transformation_matrix': transformation.tolist()
        }

        with open(self.extrinsics_path, 'w') as f:
            yaml.dump(data, f)
        self.get_logger().info("Extrinsics saved to camera_extrinsics.yaml")

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
