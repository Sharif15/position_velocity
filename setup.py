from setuptools import setup

package_name = 'position_velocity'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'opencv-python',
        'PyYAML'
    ],
    zip_safe=True,
    maintainer='Sharif15',
    maintainer_email='sharifpial225@gmail.com',
    description='Package for PTZ camera calibration using AprilTags.',
    license='BSD',
    tests_require=[],
    entry_points={
        'console_scripts': [
            'april_tag_calibration = position_velocity.calibration.aprilTagCalibration:main',
        ],
    },
)
