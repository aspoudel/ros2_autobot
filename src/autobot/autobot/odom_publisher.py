#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')

        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to ariaNode's odometry output
        self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Re-publish with correct frame IDs for SLAM toolbox
        self.odom_pub = self.create_publisher(Odometry, '/odom_corrected', 10)

        self.get_logger().info('Odom publisher started — broadcasting odom->base_link TF')

    def odom_callback(self, msg: Odometry):
        # Ensure correct frame IDs
        msg.header.frame_id = 'odom'
        msg.child_frame_id  = 'base_link'

        # Broadcast the TF transform odom -> base_link
        t = TransformStamped()
        t.header.stamp    = msg.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id  = 'base_link'

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = 0.0
        t.transform.rotation      = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(t)

        # Also republish the corrected odom message
        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()