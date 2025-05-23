from setuptools import setup

package_name = 'position_velocity'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/data', [
            'position_velocity/data/intrinsics.yaml',
            'position_velocity/data/extrinsics.yaml'
            ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sharif15',
    maintainer_email='sharifpial225@gmail.com',
    description='Calibration package using AprilTags to generate camera extrinsics',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'april_tag_calibration = position_velocity.calibration.aprilTagCalibration:main'
        ],
    },
)
