/*
*   A basic node for ros2 that runs with ariaCoda
*   To run use 'ros2 run ariaNode ariaNode -rp /dev/ttyUSB0'
*
*   Author: Kieran Quirke-Brown
*   Date: 12/01/2024
*   Modified: added odometry publishing
*/

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <signal.h>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>

# include "Aria/Aria.h"

//used with signal handler as signal handler function doesn't accept parameters
bool stopRunning = false;

using namespace std::chrono_literals;

class ariaNode : public rclcpp::Node {
    public:
        ariaNode(float* forwardSpeed, float* rotationSpeed) : Node("Aria_node") {
            currentForwardSpeed = forwardSpeed;
            currentRotationSpeed = rotationSpeed;

            cmdVelSub = create_subscription<geometry_msgs::msg::Twist>(
                "cmd_vel", 10, std::bind(&ariaNode::cmdVelCallback, this, std::placeholders::_1)
            );

            odomPub = create_publisher<nav_msgs::msg::Odometry>("odom", 10);
            tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        }

        void publishOdometry(double x, double y, double theta, double vx, double vth) {
            auto now = this->get_clock()->now();

            tf2::Quaternion q;
            q.setRPY(0, 0, theta);

            // Publish odom -> base_link TF
            geometry_msgs::msg::TransformStamped tf;
            tf.header.stamp = now;
            tf.header.frame_id = "odom";
            tf.child_frame_id = "base_link";
            tf.transform.translation.x = x;
            tf.transform.translation.y = y;
            tf.transform.translation.z = 0.0;
            tf.transform.rotation.x = q.x();
            tf.transform.rotation.y = q.y();
            tf.transform.rotation.z = q.z();
            tf.transform.rotation.w = q.w();
            tfBroadcaster->sendTransform(tf);

            // Publish /odom topic
            nav_msgs::msg::Odometry odom;
            odom.header.stamp = now;
            odom.header.frame_id = "odom";
            odom.child_frame_id = "base_link";
            odom.pose.pose.position.x = x;
            odom.pose.pose.position.y = y;
            odom.pose.pose.position.z = 0.0;
            odom.pose.pose.orientation.x = q.x();
            odom.pose.pose.orientation.y = q.y();
            odom.pose.pose.orientation.z = q.z();
            odom.pose.pose.orientation.w = q.w();
            odom.twist.twist.linear.x = vx;
            odom.twist.twist.angular.z = vth;
            odomPub->publish(odom);
        }

    private:
        void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
            *currentForwardSpeed = msg->linear.x;
            *currentRotationSpeed = msg->angular.z;
        }

        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmdVelSub;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub;
        std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
        float* currentForwardSpeed;
        float* currentRotationSpeed;
};

void my_handler(int s){
    printf("Caught signal %d\n",s);
    stopRunning = true;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    Aria::init();
    ArArgumentParser parser(&argc, argv);
    parser.loadDefaultArguments();
    ArRobot* robot;
    robot = new ArRobot();

    signal(SIGINT, my_handler);

    ArRobotConnector robotConnector(&parser, robot);
    if(!robotConnector.connectRobot()) {
        ArLog::log(ArLog::Terse, "simpleConnect: Could not connect to the robot.");
        if(parser.checkHelpAndWarnUnparsed()) {
            Aria::logOptions();
            Aria::exit(1);
        }
    }

    robot->setAbsoluteMaxTransVel(400);

    float forwardSpeed = 0.0;
    float rotationSpeed = 0.0;

    robot->runAsync(true);
    robot->enableMotors();

    auto aNode = std::make_shared<ariaNode>(&forwardSpeed, &rotationSpeed);

    while (!stopRunning) {
        rclcpp::spin_some(aNode);

        robot->lock();
        robot->setVel(forwardSpeed * 500);
        robot->setRotVel(rotationSpeed * 50);

        // Read pose from AriaCoda (mm -> m, deg -> rad)
        double x     = robot->getX()      / 1000.0;
        double y     = robot->getY()      / 1000.0;
        double theta = robot->getTh()     * M_PI / 180.0;
        double vx    = robot->getVel()    / 1000.0;
        double vth   = robot->getRotVel() * M_PI / 180.0;
        robot->unlock();

        aNode->publishOdometry(x, y, theta, vx, vth);
    }

    robot->disableMotors();
    robot->stopRunning();
    robot->waitForRunExit();

    Aria::exit(0);
    return 0;
}