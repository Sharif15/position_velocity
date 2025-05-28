## Setup 

First clone the repo to the src folder 

'''sh
cd ros2_ws/src
git clone https://github.com/Sharif15/position_velocity.git
'''

go back to the ros2_ws and activate the python envirenment 

'''sh 
cd ../..
source venv/bin/activate 
pip install apriltag

'''

once we have the required libraries we are ready to build the project 

'''sh 
export PYTHONPATH=$VIRTUAL_ENV/lib/python3.12/site-packages:$PYTHONPATH

cd ros2_ws/
colcon build
source install/setup.bash

'''

## Running the script 

To run the script we need to activate our current camera first 

'''sh

ros2 launch axis_camera axis_camera.launch hostname:=192.168.0.100 password:=NAPPLab1 frame_width:=1920 frame_height:=1080 fps:=60 enable_ptz:=true

'''

Then run the desired script :

'''sh

ros2 run position_velocity locate_tag

'''