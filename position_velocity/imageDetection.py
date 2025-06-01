import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from position_velocity.msg import ObjectTracking, ObjectTrackingArray

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging




# Suppress YOLO warnings
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load YOLO model (make sure the weights file exists at the path)
model = YOLO("yolov8x-worldv2.pt")
model.set_classes(["person"])  # Only detect "person"

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

class ObjectDetectorNode(Node):

    def __init__(self):
        super().__init__('object_detector_node')

        # Create publisher for object tracking coordinates
        self.coord_publisher = self.create_publisher(ObjectTrackingArray, 'tracking_coords', 10)

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
            # Update tracker with empty detections to handle track aging
            tracks = tracker.update_tracks([], frame=frame)
            return  # No objects detected

        # Making the input for DeepSORT 

        detections = []

        for box in boxes:
            coords = box.xyxy[0] # [x1, y1, x2, y2]
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            # Convert from [x1, y1, x2, y2] to [left, top, width, height]
            left = coords[0].item()
            top = coords[1].item()
            width = (coords[2] - coords[0]).item()
            height = (coords[3] - coords[1]).item()
            
            detection = ([left, top, width, height], confidence, class_id)
            detections.append(detection)

        tracks = tracker.update_tracks(detections, frame=frame)

        msg_out = ObjectTrackingArray()
        msg_out.frame_width = frame.shape[1]
        msg_out.frame_height = frame.shape[0]

        # Draw and collect tracked object info
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x = (l + r) / 2.0
            y = (t + b) / 2.0
            width = r - l
            height = b - t
            class_id = track.det_class

            # Add to message
            tracked_msg = ObjectTracking()
            tracked_msg.id = track_id
            tracked_msg.x = x
            tracked_msg.y = y
            tracked_msg.width = width
            tracked_msg.height = height
            tracked_msg.class_id = class_id

            msg_out.objects.append(tracked_msg)

            # Visualize
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.coord_publisher.publish(msg_out)

        cv2.imshow("Tracked Objects", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        # Optional: Show image with detection (for debugging)
        detection_img = results[0].plot()
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
