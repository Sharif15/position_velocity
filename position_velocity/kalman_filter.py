import threading
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from py_kalman_filter.kalman_multi_ros_wrapper import KalmanService # type: ignore
from py_kalman_filter.kalman_update_multi_publisher import Kalman_Update_Multi_Publisher # type: ignore
from srv_interfaces.srv import CreateKalman # type: ignore
from srv_interfaces.msg import MultiPredictAll, DestroyKalman # type: ignore
from ptz_videography_interfaces.msg import Coordinates, States


class Tracker(Node):
    # Apply a Multi Object Kalman Filter to object's world coordinates and publishes their states.
    def __init__(self):
        super().__init__('tracker')

        # Define Kalman Filter parameters
        self.A = np.array([[1, 0, 0, 1, 0, 0],
                           [0, 1, 0, 0, 1, 0],
                           [0, 0, 1, 0, 0, 1],
                           [0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        
        self.Q = np.eye(6)*1e-3

        # Maximum trace of the covariance matrix
        self.max_P_trace = 100

        # Dictionary for storing Kalman publishers
        self.kalman_publishers = {}
        self.id_to_str = {}

        # Create a client to request new Kalman filters
        self.client = self.create_client(CreateKalman, 'kalman_create')
        if not self.client.wait_for_service(timeout_sec=4.):
            self.get_logger().error('Multi object tracker service not available.')

        # Publisher for object states
        self.objects_publisher = self.create_publisher(States, 'object_states', 1)
        
        # Publisher to destroy kalman topics
        self.destroy_publisher = self.create_publisher(DestroyKalman, 'kalman_destroy', 10)

        # Subscriber to object detections in world coordinates
        self.detection_subscriber = self.create_subscription(Coordinates, 'world_coordinates', self.objects_callback, 1)
        
        # Subscriber to the kalman filter predictions
        self.predict_subscriber = self.create_subscription(MultiPredictAll, 'kalman_predict', self.predict_callback, 1)


    def send_create_request(self, str_id):
            # Send a CreateKalman request and wait for a response
            request = CreateKalman.Request()
            request.a = [float(x) for x in self.A.flatten()]
            request.h = [float(x) for x in self.H.flatten()]
            request.q = [float(x) for x in self.Q.flatten()]
            response = self.client.call(request)

            # Store the model ID and Kalman publisher
            model_id = response.modelid
            self.id_to_str[model_id] = (str_id)
            self.kalman_publishers[str_id] = (Kalman_Update_Multi_Publisher(model_id))


    def objects_callback(self, msg):
        # If there are no detections, do nothing
        if len(msg.str_id) == 0:
            return


        # Loop through each detected object
        for str_id, x, y, z in zip(msg.str_id, msg.x, msg.y, msg.z):

            print(f"The input : {msg}")

            # Check if we already have a Kalman filter for this object
            if str_id not in self.kalman_publishers:
                threading.Thread(target=self.send_create_request, args=(str_id,)).start()
            else:
                # Send the world coordinates to the Kalman filter
                self.kalman_publishers[str_id].sendUpdate(np.array([x, y, z]))
            

    def predict_callback(self, msg):
        # Create the States message
        states_msg = States()
        states_msg.str_id = []
        states_msg.x = []
        states_msg.y = []
        states_msg.z = []
        states_msg.vx = []
        states_msg.vy = []
        states_msg.vz = []

        for pred in msg.predictions:
            # Read the id in string format
            str_id = self.id_to_str.get(pred.modelid)
            if str_id is None:
                continue

            # If the covariance is too large, destroy the Kalman filter
            if np.trace(np.array(pred.p).reshape(6, 6)) > self.max_P_trace:
                message = DestroyKalman()
                message.modelid = pred.modelid
                self.destroy_publisher.publish(message)

                # Remove the Kalman filter from the dictionaries
                self.kalman_publishers.pop(str_id, None)
                self.id_to_str.pop(pred.modelid, None)
                continue
            
            # Check if the prediction is valid
            if pred.x[0]*pred.x[1]*pred.x[2] == 0:
                continue

            # Append data to the states message
            states_msg.str_id.append(str_id)
            states_msg.x.append(float(pred.x[0]))
            states_msg.y.append(float(pred.x[1]))
            states_msg.z.append(float(pred.x[2]))
            states_msg.vx.append(float(pred.x[3]))
            states_msg.vy.append(float(pred.x[4]))
            states_msg.vz.append(float(pred.x[5]))

        # Publish the states message
        self.objects_publisher.publish(states_msg)
        
        print(f"The velocities : {states_msg}")


def main(args=None):
    rclpy.init(args=args)

    # Initialize the Kalman filter service create the tracker node
    kalman_service = KalmanService(time=1/30)
    tracker = Tracker()

    # Create a multi thread executor and add the nodes
    executor = MultiThreadedExecutor()
    executor.add_node(kalman_service)
    executor.add_node(tracker)

    # Spin the executor and handle shutdown
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        tracker.destroy_node()
        kalman_service.destroy_node()
        rclpy.shutdown()