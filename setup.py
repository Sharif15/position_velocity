from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'position_velocity'

setup(
    name=package_name,
    version='0.0.1',
   packages=find_packages(where='.'),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', 
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share',package_name,'data'), [
            os.path.join('data','intrinsics.yaml'),
            os.path.join('data','extrinsics.yaml')
            ]),
        (os.path.join('share',package_name,'config'), glob(os.path.join('config','*.yaml'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch','*.py')))
    ],
    install_requires=[
        'setuptools',
        'PyYAML',
        'opencv-python',
        'numpy',
        'apriltag',
        'rclpy',
        'deep_sort_realtime',
        'ultralytics'
        ],
    zip_safe=True,
    maintainer='sharif15',
    maintainer_email='sharifpial225@gmail.com',
    description='Calibration package using AprilTags to generate camera extrinsics',
    license='BSD',
    extras_require={
        'test': ['pytest']
    },
    entry_points={
        'console_scripts': [
            'object_detection = position_velocity.object_tracking.imageDetection:main',
            'calibration = position_velocity.calibration.singleTagDetection_config:main',
            'calculate_position = position_velocity.calculations.cameraToWorld:main',
            # test
            'test_position = position_velocity.calculations.position_test:main'
        ],
    },
)
