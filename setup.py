from setuptools import find_packages, setup

package_name = 'waypoint_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','numpy','cvxpy','scipy'],

    zip_safe=True,
    maintainer='hslin',
    maintainer_email='hslin@todo.todo',
    description='UR5e waypoint force/torque controller (DH).',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'waypoint_force_controller_dh = waypoint_control.waypoint_force_controller_dh:main',
            'waypoint_cli = waypoint_control.waypoint_cli:main',
            'wmpc_node = waypoint_control.wmpc_node:main',   
        ],
    },
)
