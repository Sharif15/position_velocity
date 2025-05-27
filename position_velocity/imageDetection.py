import rclpy
from rclpy.node import Node
from ptz_videography_interfaces.msg import ObjectTracking
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Suppress YOLO warnings
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load YOLO model (make sure the weights file exists at the path)
model = YOLO("yolov8x-worldv2.pt")
model.set_classes(["person"])  # Only detect "person"

class ObjectDetectorNode(Node):

    def __init__(self):
        super().__init__('object_detector_node')

        # Create publisher for object tracking coordinates
        self.coord_publisher = self.create_publisher(ObjectTracking, 'tracking_coords', 10)

        # Subscribe to camera image topic
        self.image_subscriber = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.get_logger().info("ObjectDetectorNode initialized and subscribed to /image_raw/compressed")

    def image_callback(self, msg):
        # Decode compressed image to OpenCV format
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("Received empty or corrupted image frame")
            return

        # Resize for faster processing (optional)
        frame = cv2.resize(frame, (960, 540))

        # Run YOLO detection
        results = model(frame)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return  # No objects detected

        # Calculate average object center (for all detections)
        object_x = []
        object_y = []

        for box in boxes:
            coords = box.xyxy[0]
            cx = (coords[0] + coords[2]) / 2.0
            cy = (coords[1] + coords[3]) / 2.0
            object_x.append(cx.item())
            object_y.append(cy.item())

        avg_x = sum(object_x) / len(object_x)
        avg_y = sum(object_y) / len(object_y)

        # Prepare and publish ROS2 message
        msg_out = ObjectTracking()
        msg_out.frame_height = frame.shape[0]
        msg_out.frame_width = frame.shape[1]
        msg_out.objx = avg_x
        msg_out.objy = avg_y

        self.coord_publisher.publish(msg_out)

        # Optional: Show image with detection (for debugging)
        detection_img = results[0].plot()
        cv2.circle(detection_img, (int(avg_x), int(avg_y)), 5, (255, 255, 255), -1)
        cv2.imshow("Detection", detection_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down object detector")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
