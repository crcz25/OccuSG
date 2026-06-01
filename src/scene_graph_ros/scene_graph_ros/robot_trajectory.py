#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# Try to import tf transformations, fallback to manual conversion if not available
try:
    from tf_transformations import euler_from_quaternion
except ImportError:

    def euler_from_quaternion(quaternion):
        """Manual quaternion to euler conversion"""
        x, y, z, w = quaternion
        # Calculate yaw (rotation around z-axis)
        yaw = math.atan2(2.0 * (w * z + x * y), 1 - 2.0 * (y * y + z * z))
        return 0, 0, yaw  # Return (roll, pitch, yaw)

#######################
# HOSPITAL
#######################

waypoints_hospital_expl = [
    (0.0, 16.5, -1.57, 1),
    (0.0, 15, -1.57, 1),
    (0.0, 15, 1.57, 1),
]


#######################
# FULL OFFICE
#######################

# (x, y, yaw, hold_time)
waypoints_full_office04 = [
    (2.78, 18.2, 0, 1),
    (9.78, 18.2, 1.57, 1),
    (9.78, 19.7, 1.57, 1),
    (11, 20.0, 2.48, 1),
    (9.78, 20.0, -1.57, 1),
    (9.78, 16.2, -0.35, 1),
    (12.6, 15.2, 0, 1),
    (18.9, 15.2, 0, 1),
    (18.9, 17.0, 2.455, 1),
    (12.7, 17.3, 1.584, 1),
    (12.7, 21.7, 0, 1),
    (17.3, 21.7, 0, 1),
    (18.9, 20.5, -1.57, 1),
    (18.9, 13.9, 3.13, 1),
    (12.2, 13.9, -1.57, 1),
    (12.2, 10.6, -1.57, 1),
    (12.5, 10.6, -3.14, 1),
    (5.6, 10.6, -1.57, 1),
    (5.6, 8.3, -3.07, 1),
    (0.5, 7.9, -1.53, 1),
    (0.7, 4.6, 0, 1),
    (3.6, 4.8, 1.71, 1),
    (3.3, 8.2, 0.02, 1),
    (6.3, 8.4, -1.57, 1),
    (6.3, 5.4, 0, 1),
    (12.2, 5.4, 0, 1),
    (14.7, 5.4, 0, 1),
    (15.6, 8.7, 1.29, 1),
    (15.6, 1.5, -1.57, 1),
    (19.9, 1.5, 0, 1),
    (15.6, 1.5, -3.13, 1),
    (15.6, 5.4, 1.57, 1),
    (11.3, 5.4, -3.10, 1),
    (11.3, 2.4, -1.57, 1),
    (11.3, 4.0, 1.57, 1),
    (8.2, 4.0, -3.12, 1),
    (8.2, 2.0, -1.57, 1),
    (8.2, 4.0, 1.57, 1),
    (0.7, 4.0, -3.14, 1),
    (0.7, 1, -1.57, 1),
    (-5.5, 1, -3.14, 1),
    (-5.5, 5.5, 1.57, 1),
    (-25.8, 5.5, -3.14, 1),
    (-25.8, 10.5, 1.57, 1),
    (-5.5, 9.7, 0, 1),
    (-5.5, 13.8, 1.57, 1),
    (-22, 13.8, 3.14, 1),
    (-22, 18.8, 1.57, 1),
    (-26, 19.2, 2.92, 1),
    (-25.5, 17.9, -1.47, 1),
    (-22, 19, 0.0, 1),
    (-20.7, 18.9, -0.01, 1),
    (-21, 20, -14, 1),
    (-20, 18.5, 0, 1),
    (-3.5, 18.5, 1.57, 1),
    (-4, 20.4, 1.75, 1),
    (-4, 20.4, -1.40, 1),
    (-3.5, 18.5, 1.57, 1),
    (-1.3, 19.6, 0.80, 1),
    (1.4, 19.0, -0.20, 60),
]

#######################
# 2room
#######################
waypoints_2room = [
    (2.0, 17.0, 3.14, 1),
    (1.0, 17.0, 3.14, 1),
    (1.0, 13.0, 2.35619, 1),
    (3.0, 13.0, 2.35619, 1),
    (3.0, 5.0, 2.35619, 1),
    (-3.0, 5.0, 0.785398, 1),
    (-3.0, -3.0, 0.0, 1),
    (0.0, -1.77, 0.785398, 1),
    (1.0, -1.0, 1.57, 1),
    (1.0, 2.5, 3.14, 1),
    (-1.0, 3.0, -0.785398, 1),
]

#######################
# bookstore
#######################
bookstore_2room = [
    (-2.0, 6.0, 0.0, 1),  # (x, y, yaw, hold_time)
    (-2.0, 6.0, -1.57, 0.5),
    (-2.0, 6.0, -3.14, 0.5),
    (-7.0, 6.0, -3.14, 1),
    (-7.0, 6.0, -1.57, 1),
    (-7.0, -6.3, -1.57, 1),
    (-7.0, -6.3, 0.0, 1),
    (-3.5, -6.3, 0.0, 1),
    (-3.5, -6.3, 1.57, 1),
    (-3.5, 0.0, 1.57, 1),
    (-3.5, 0.0, 0.0, 1),
    (-1.3, 0.0, 0.0, 1),
    (-1.3, 0.0, -1.57, 1),
    (-1.3, -6.3, -1.57, 1),
    (-1.3, -6.3, 0.0, 1),
    (0.5, -6.3, 0.0, 1),
    (0.5, -6.3, 1.57, 1),
    (0.5, -3.0, 1.57, 1),
    (0.5, -3.0, 0.0, 1),
    (4.0, -3.0, 0.0, 1),
    (4.0, -3.0, 1.57, 1),
    (4.0, -0.1, 1.57, 1),
    (4.0, -0.1, 1.0, 1),
    (5.0, 1.5, 1.0, 1),
    (5.0, 1.5, 0.0, 1),
    (7.2, 1.5, 0.0, 1),
    (7.2, 1.5, -1.57, 1),
    (7.0, -6.0, -1.57, 1),
    (7.0, -6.0, -3.14, 1),
    (2.0, -6.0, -3.14, 1),
    (2.0, -6.0, 1.57, 1),
    (2.0, -6.0, 1.57, 1),
    (2.0, 0.0, 1.57, 1),
    (2.0, 0.0, 3.14, 1),
    (0.0, 0.0, 3.14, 1),
    (0.0, 0.0, 1.57, 1),
    (0.0, 2.0, 1.57, 1),
    (0.0, 2.0, 3.14, 1),
    (0.0, 2.0, 1.78, 1),
    (-0.6, 5.0, 1.78, 1),
    (-2.0, 6.0, 3.14, 60),
]


#######################
# hospital
#######################

hospital = [
    (0.0, 14.0, -1.57, 1),
    (0.0, 10.0, -1.57, 1),
    (0.0, 10.0, -3.14, 1),
    (-4.5, 10.0, -3.14, 1),
    (-4.5, 10.0, -1.57, 1),
    (-4.5, -9.0, -1.57, 1),
    (-4.5, -9.0, -3.14, 1),
    (-7.2, -9.0, -3.14, 1),
    (-7.2, -9.0, 1.57, 1),
    (-7.2, -1.5, 1.57, 1),
    (-7.2, -1.5, 3.14, 1),
    (-9.0, -1.5, 3.14, 1),
    (-9.0, -1.5, 1.57, 1),
    (-9.0, 3.0, 1.57, 1),
    (-9.0, 3.0, 3.14, 1),
    (-9.0, 3.0, -1.57, 1),
    (-9.0, -9.0, -1.57, 1),
    (-9.0, -9.0, -1.57, 1),
    (-9.0, -9.0, 0.0, 1),
    (-5.0, -9.0, 0.0, 1),
    (-5.0, -9.0, -1.57, 1),
    (-5.0, -23.0, -1.57, 1),
    (-5.0, -23.0, -3.14, 1),
    (-7.2, -23.0, -3.14, 1),
    (-7.2, -23.0, -3.14, 1),
    (-7.2, -23.0, 1.57, 1),
    (-7.2, -16.0, 1.57, 1),
    (-7.2, -16.0, 3.14, 1),
    (-9.2, -16.0, 3.14, 1),
    (-9.2, -16.0, 1.57, 1),
    (-9.2, -11.5, 1.57, 1),
    (-9.2, -11.5, 3.14, 1),
    (-9.2, -11.5, -1.57, 1),
    (-9.2, -23.0, -1.57, 1),
    (-9.2, -23.0, 0.0, 1),
    (-5.0, -23.0, 0.0, 1),
    (-5.0, -23.0, -1.57, 1),
    (-5.0, -25.0, -1.57, 1),
    (-5.0, -25.0, 3.14, 1),
    (-8.0, -25.0, 3.14, 1),
    (-8.0, -25.0, -2.2, 1),
    (-8.0, -25.0, -1.57, 1),
    (-8.0, -33.5, -1.57, 1),
    (-8.0, -33.5, 0.0, 1),
    (-2.5, -33.5, 0.0, 1),
    (-2.5, -33.5, 1.57, 1),
    (-2.5, -27.0, 1.57, 1),
    (-2.5, -27.0, 0.0, 1),
    (5.5, -27.0, 0.0, 1),
    (5.5, -27.0, 1.57, 1),
    (5.5, -23.0, 1.57, 1),
    (5.5, -23.0, 0.0, 1),
    (7.5, -23.0, 0.0, 1),
    (7.5, -23.0, 1.45, 1),
    (9.0, -11.0, 1.45, 1),
    (9.0, -11.0, 1.45, 1),
    (9.0, -11.0, -0.98, 1),
    (9.0, -11.0, -1.68, 1),
    (7.5, -23.0, -1.68, 1),
    (7.5, -23.0, 3.14, 1),
    (5.0, -23.0, 3.14, 1),
    (5.0, -23.0, 1.57, 1),
    (5.0, -9.0, 1.57, 1),
    (5.0, -9.0, 0.0, 1),
    (5.0, -9.0, 0.0, 1),
    (8.7, -9.0, 0.0, 1),
    (8.7, -9.0, 1.57, 1),
    (8.7, 2.6, 1.57, 1),
    (8.7, 2.6, 0.0, 1),
    (8.7, 2.6, -1.57, 1),
    (8.7, -9.0, -1.57, 1),
    (8.7, -9.0, 3.14, 1),
    (5.0, -9.0, 3.14, 1),
    (5.0, -9.0, 1.57, 1),
    (5.0, 4.5, 1.57, 1),
    (5.0, 4.5, 2.1, 1),
    (0.0, 12.6, 2.1, 1),
    (0.0, 12.6, 3.14, 1),
    (0.0, 12.6, -1.57, 60),
]

small_house = [
    # 0.0 derecha
    # 1.57 arriba
    # -1.57 abajo
    # 3.14 izq

    (-3.5, -2.5, 0.0, 1),
    (-3.5, -2.5, 1.57, 1),
    (-3.5, -2.5, 3.14, 1),

    (-8.8, -2.5, 3.14, 1),
    (-8.8, -2.5, -1.57, 1),
    (-8.8, -2.5, 0.0, 1),
    (-8.8, -2.5, 1.57, 1),

    (-8.8, 0.0, 1.57, 1),
    (-8.8, 0.0, 0.0, 1),
    (-8.8, 0.0, -1.57, 1),
    (-8.8, 0.0, 0.0, 1),

    (-1.5, 0.0, 0.0, 1),
    (-1.5, 0.0, 1.57, 1),
    (-1.5, 0.0, 3.14, 1),
    (-1.5, 0.0, -1.57, 1),

    (-1.5, -4.0, -1.57, 1),
    (-1.5, -4.0, 0.0, 1),
    (-1.5, -4.0, 1.57, 1),
    (-1.5, -4.0, 0.0, 1),

    (6.5, -4.0, 0.0, 1),

    (6.5, -2.0, 1.57, 1),
    (6.5, -2.0, 0.0, 1),
    (6.5, -2.0, -1.57, 1),
    (6.5, -2.0, 3.14, 1),
    (6.5, -2.0, 2.5, 1),

    (3.45, 1.2, 2.5, 1),
    (3.45, 1.2, 1.57, 1),
    (3.45, 1.2, 3.14, 1),
    # (3.45, 0.27, 0.0, 1),
    # (3.45, 0.27, -1.57, 1),
    # (3.45, 0.27, 3.14, 1),

    (0, 1.2, 3.14, 1),
    (0, 1.2, 1.57, 1),
    (0, 3, 1.57, 1),
    (0, 3, -2.09, 1),
]


husarion_office = [
    # up -1.57
    # down 1.57
    # Left 0.0
    # Right 3.14

    (0.6, 0.0, -1.57, 0.5),
    (0.6, 0.0, 0.0, 0.5),
    (0.6, 0.0, 1.57, 0.5),
    (0.6, 0.0, 3.14, 0.5),
    (0.6, 0.0, -1.57, 0.5),

    (0.6, -2.6, -1.57, 0.5),
    (0.6, -2.6, 0.0, 0.5),
    (0.6, -2.6, 1.57, 0.5),
    (0.6, -2.6, 3.14, 0.5),
    (0.6, -2.6, -1.57, 0.5),
    (0.6, -2.6, -0.87, 0.5),

    (2, -5.13, -0.87, 0.5),
    (2, -5.13, 0.0, 0.5),
    (2, -5.13, 1.57, 0.5),
    (2, -5.13, 3.14, 0.5),
    (2, -5.13, -0.87, 0.5),

    (3.3, -6.6, -0.87, 0.5),
    (3.3, -6.6, 1.57, 0.5),
    (3.3, -6.6, 3.14, 0.5),
    (3.3, -6.6, -1.57, 0.5),
    (3.3, -6.6, 0.0, 0.5),

    (4.5, -6.6, 0.0, 0.5),
    (4.5, -6.6, 1.57, 0.5),
    (4.5, -6.6, 3.14, 0.5),
    (4.5, -6.6, -1.57, 0.5),
    (4.5, -6.6, 0.37, 0.5),

    (6, -5.82, 0.37, 0.5),
    (6, -5.82, 0.0, 0.5),
    (8, -5.82, 0.0, 0.5),
    (8, -5.82, 1.57, 0.5),
    (8, -5.82, 3.14, 0.5),
    (8, -5.82, -1.57, 0.5),

    (8, -9.8, -1.57, 0.5),
    (8, -9.8, 0.0, 0.5),
    (8, -9.8, 1.57, 0.5),
    (8, -9.8, 3.14, 0.5),

    (4.1, -9.8, 3.14, 0.5),
    (4.1, -9.8, -2.62, 0.5),
    (1.6, -11.13, -2.62, 0.5),
    (1.6, -11.13, 3.14, 0.5),
    (1.6, -11.13, 1.57, 0.5),
    (4.1, -9.8, 0.54, 0.5),
    (4.1, -9.8, 0.0, 0.5),

    (9.5, -9.6, 0.0, 0.5),
    (9.5, -9.6, -1.57, 0.5),
    (9.5, -9.6, 0.0, 0.5),
    (9.5, -9.6, 1.57, 0.5),
    (9.5, -9.6, 3.14, 0.5),

    (8.3, -9.5, 3.14, 0.5),
    (8.3, -9.5, 1.57, 0.5),

    (8.3, -5, 1.57, 0.5),
    (8.3, -5, 0.0, 0.5),

    (12, -5, 0.0, 0.5),
    (12, -5, -1.12, 0.5),
    (12.5, -6.2, -1.12, 0.5),
    (12.5, -6.2, -1.57, 0.5),
    (12.5, -6.2, 3.14, 0.5),
    (12.5, -5, 1.12, 0.5),
    (12.5, -5, 3.14, 0.5),

    (8.3, -5, 3.14, 0.5),
    (8.3, -5, 1.57, 0.5),

    (8.3, -3, 1.57, 0.5),
    (8.3, -3, 3.14, 0.5),

    (6, -3, 3.14, 0.5),
    (6, -3, 0.0, 0.5),

    (8.3, -3, 0.0, 0.5),
    (8.3, -3, 1.57, 0.5),

    (8.3, -1.2, 1.57, 0.5),
    (8.3, -1.2, 3.14, 0.5),

    (6.6, -1.2, 3.14, 0.5),
    (6.6, -1.2, 1.57, 0.5),
    (6.6, -0.58, 1.57, 0.5),
    (6.6, -0.58, 2.83, 0.5),
    (6.6, -0.58, 0.23, 0.5),
    (6.6, -0.58, -1.57, 0.5),
    (6.6, -1.2, -1.57, 0.5),
    (6.6, -1.2, 0.0, 0.5),

    (8.5, -1.2, 0.0, 0.5),
    (8.5, -1.2, 1.57, 0.5),
    (8.5, 0, 1.57, 0.5),
    (8.5, 0, 0.38, 0.5),
    (8.5, 0, 2.71, 0.5),
    (8.5, 0, -1.57, 0.5),
    (8.5, -1.2, -1.57, 0.5),
    (8.5, -1.2, 0.0, 0.5),

    (10.5, -1.2, 0.0, 0.5),
    (10.5, -1.2, 1.57, 0.5),
    (10.5, -1.2, 0.0, 0.5),
    (10.5, -1.2, -1.57, 0.5),

]

# back_forth = bookstore_2room  # + bookstore_2room[::-1]
# back_forth = waypoints_full_office04  # + bookstore_2room[::-1]
# back_forth = hospital  # + hospital[::-1]
# back_forth = waypoints_2room  # + waypoints_2room[::-1]
# back_forth = small_house # + small_house[::-1]
back_forth = husarion_office # + husarion_office[::-1]


class RobotTrajectory(Node):
    def __init__(self):
        super().__init__("robot_trajectory")

        # QoS profile for reliable communication
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub = self.create_publisher(Twist, "/cmd_vel", qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile
        )

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        # Control parameters
        self.speed = 0.15
        self.angular_speed = 0.15
        self.position_tolerance = 0.1
        self.angle_tolerance = 0.05

        # Wait for odometry
        self.get_logger().info("Waiting for odometry...")
        while not self.odom_received and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Odometry received, starting trajectory")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Convert quaternion to euler
        orientation = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )
        self.odom_received = True

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def move_to_position(self, target_x, target_y):
        """Move to target position while maintaining current orientation"""
        self.get_logger().info(f"Moving to position ({target_x:.2f}, {target_y:.2f})")

        while rclpy.ok():
            # Calculate distance to target
            dx = target_x - self.x
            dy = target_y - self.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < self.position_tolerance:
                # Stop the robot
                self.pub.publish(Twist())
                self.get_logger().info("Position reached!")
                break

            # Calculate angle to target
            target_angle = math.atan2(dy, dx)
            angle_diff = self.normalize_angle(target_angle - self.yaw)

            twist = Twist()

            # If we're facing roughly the right direction, move forward
            if abs(angle_diff) < 0.3:  # ~17 degrees
                twist.linear.x = min(self.speed, distance)
                twist.angular.z = 0.3 * angle_diff  # Small correction
            else:
                # Turn towards target first
                twist.linear.x = 0.0
                twist.angular.z = (
                    self.angular_speed if angle_diff > 0 else -self.angular_speed
                )

            self.pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

    def rotate_to_angle(self, target_angle):
        """Rotate to target angle"""
        self.get_logger().info(f"Rotating to angle {target_angle:.2f}")

        while rclpy.ok():
            angle_diff = self.normalize_angle(target_angle - self.yaw)

            if abs(angle_diff) < self.angle_tolerance:
                # Stop rotation
                self.pub.publish(Twist())
                self.get_logger().info("Target angle reached!")
                break

            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = (
                self.angular_speed if angle_diff > 0 else -self.angular_speed
            )

            self.pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)

    def run_trajectory(self):
        """Execute the waypoint trajectory"""
        for waypoint in back_forth:  # Change to desired waypoints list
            if not rclpy.ok():
                break
            target_x, target_y, target_yaw, hold_time = waypoint
            self.get_logger().info(
                f"Going to waypoint: ({target_x:.2f}, {target_y:.2f}, {target_yaw:.2f})"
            )
            # Move to position
            self.move_to_position(target_x, target_y)
            # Rotate to final orientation
            self.rotate_to_angle(target_yaw)
            # Hold position for data recording
            self.get_logger().info(f"Holding position for {hold_time} seconds...")
            self.pub.publish(Twist())  # Ensure robot is stopped
            hold_start = self.get_clock().now()
            while (
                self.get_clock().now() - hold_start
            ).nanoseconds / 1e9 < hold_time and rclpy.ok():
                self.pub.publish(Twist())  # Keep publishing zero velocity
                rclpy.spin_once(self, timeout_sec=0.1)

        # Final stop command before exiting
        self.pub.publish(Twist())
        self.get_logger().info("All waypoints visited, exiting trajectory node.")


def main(args=None):
    rclpy.init(args=args)
    robot = None
    try:
        robot = RobotTrajectory()
        robot.run_trajectory()
    except KeyboardInterrupt:
        pass
    finally:
        if robot is not None:
            robot.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
