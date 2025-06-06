## Setup 

First clone the repo to the src folder 

```sh
cd ros2_ws/src
git clone https://github.com/Sharif15/position_velocity.git

```

go back to the ros2_ws and activate the python envirenment 

```sh 
cd ../..
source venv/bin/activate 

```

Download the required libreries :

go into the position_velocity base folder and run 

```sh
pip install -r requirements.txt

```

once we have the required libraries we are ready to build the project 

```sh 
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH

cd ros2_ws/
colcon build
source install/setup.bash

```

## Running the package 

There are two ways the package can be started 

1. Launching with starting the camera 
2. Launching without starting the camera

### Launching with the camera 

To launch the product along with the camera run this : 

```sh

ros2 launch position_velocity position_velocity.launch.py launch_camera:=true

```

### Launching without the camera 

To run the product with rosbag or camera that is already launched use this : 

```sh

ros2 launch position_velocity position_velocity.launch.py

```

## Running the scripts Manually 

To run the script we need to activate our current camera first 

```sh

ros2 launch axis_camera axis_camera.launch hostname:=192.168.0.100 password:=NAPPLab1 frame_width:=1920 frame_height:=1080 fps:=60 enable_ptz:=true

```

Then run the desired script :

```sh

ros2 run position_velocity < entry_point >

```

### Entry points so far 

Calibration : 

```sh

ros2 run position_velocity calibration

```

Object detection and tracking with id : 

```sh

ros2 run position_velocity object_detection

```

Calculate world position : 

```sh

ros2 run position_velocity calculate_position

```

#### Still to come ... 

- Unified script to calculate world position and velocity of object and exporting the data in a ros topic using the states.msg format 
