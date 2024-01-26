import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'foresee_the_unseen'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'launch', 'slam_toolbox'), glob(os.path.join('launch', 'slam_toolbox', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config/*.yaml'))),
        (os.path.join('share', package_name, 'resource'), glob(os.path.join('resource/road_structure.xml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visualization_node = foresee_the_unseen.visualization_node:main'
        ],
    },
)
