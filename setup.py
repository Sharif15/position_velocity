from setuptools import setup, find_packages
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
            (os.path.join('share',package_name,'config'),[
                os.path.join('config','apriltag_calibration_config.yaml')
                ])
    ],
    install_requires=[
        'setuptools',
        'PyYAML',
        'opencv-python',
        'numpy',
        'apriltag'
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
            'april_tag_calibration = position_velocity.calibration.aprilTagCalibration:main',
            'locate_tag = position_velocity.calibration.singleTagDetection:main',
            'object_detection = position_velocity.imageDetection:main',
            'calibration = position_velocity.calibration.singleTagDetection_config:main'
        ],
    },
)
