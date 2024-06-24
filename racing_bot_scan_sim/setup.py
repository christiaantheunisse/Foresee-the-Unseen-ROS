from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'racing_bot_scan_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config/*.yaml"))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='christiaan',
    maintainer_email='christiaantheunisse@hotmail.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scan_simulate_node = racing_bot_scan_sim.scan_simulate_node:main'
        ],
    },
)
