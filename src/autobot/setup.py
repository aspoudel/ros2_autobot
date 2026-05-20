from setuptools import find_packages, setup

package_name = 'autobot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adarsh_sharma',
    maintainer_email='adarsh_sharma@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'autobot_roll = autobot.autobot_roll:main',
            'odom_publisher = autobot.odom_publisher:main',
        ],
    },
)
