
#!/usr/bin/env python3

import math
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


def wrap_angle(angle):
    result = math.atan2(math.sin(angle), math.cos(angle))

    if abs(abs(result) - math.pi) < 0.01:
        return math.pi

    return result


def quaternion_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class WaypointController(Node):

    def __init__(self):
        super().__init__('waypoint_controller')

        self.motor_pub = self.create_publisher(
            Float32MultiArray, '/motor_commands', 10
        )

        self.imu_sub = self.create_subscription(
            Imu, '/imu_data', self.imu_callback, 10
        )

        # ===== Existing calibration =====
        self.turn_speed = 1.4
        self.left_turn_speed = 2.63
        self.right_turn_speed = 1.52

        self.forward_command = 0.20
        self.forward_speed = 0.12
        self.turn_command = 0.50

        self.coordinate_scale = 0.5

        # ===== IMU correction =====
        self.angle_threshold = math.radians(15)
        self.max_corrections = 1
        self.correction_pause = 0.5

        # False: log IMU error, but do not correct.
        # True: allow one bounded correction.
        self.enable_imu_correction = False

        self.max_correction_angle = math.radians(30)
        self.max_correction_duration = 0.5
        self.max_imu_age = 0.5

        self.pause_duration = 0.4

        # ===== IMU =====
        self.imu_yaw = None
        self.last_imu_time = None

        # ===== Waypoints =====
        self.waypoints = self.load_waypoints()
        self.current_index = 0

        # Estimated pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # ===== State machine =====
        self.state = 'START'
        self.state_start = time.monotonic()
        self.duration = 0.0

        self.turn_kind = 'TURN'
        self.turn_target = 0.0
        self.turn_start_yaw = None
        self.turn_end_heading = 0.0

        self.corrections = 0
        self.correction_error = 0.0

        self.timer = self.create_timer(
            0.05, self.control_loop
        )

        self.get_logger().info(
            f'Loaded {len(self.waypoints)} waypoints'
        )

    def load_waypoints(self):
        path = os.path.expanduser(
            '~/rubikpi_ros2/waypoints.txt'
        )

        points = []

        with open(path) as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                x, y, theta = map(float, line.split(','))

                points.append((
                    x * self.coordinate_scale,
                    y * self.coordinate_scale,
                    theta
                ))

        return points

    def imu_callback(self, msg):
        self.imu_yaw = quaternion_to_yaw(
            msg.orientation
        )
        self.last_imu_time = time.monotonic()

    def imu_fresh(self):
        return (
            self.last_imu_time is not None
            and time.monotonic() - self.last_imu_time
            < self.max_imu_age
        )

    def motor(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.motor_pub.publish(msg)

    def stop(self):
        self.motor(0.0, 0.0)

    def set_state(self, state, duration=0.0):
        self.state = state
        self.duration = max(0.0, duration)
        self.state_start = time.monotonic()

    def abort(self, reason):
        self.stop()
        self.state = 'ABORTED'
        self.get_logger().error(reason)

    # def turn_duration(self, angle):
    #     speed = (
    #         self.left_turn_speed
    #         if angle > 0
    #         else self.right_turn_speed
    #     )
    #     return abs(angle) / speed
    def turn_duration(self, angle):
        speed = (
            self.left_turn_speed
            if angle > 0
            else self.right_turn_speed
        )

        duration = abs(angle) / speed

        # Waypoint 5 -> 6: slightly extend the 135° left turn
        if self.current_index == 5 and angle > 0:
            duration += math.radians(5) / self.left_turn_speed

        return duration

    def start_turn(self, kind, angle, end_heading):
        self.turn_kind = kind
        self.turn_target = angle
        self.turn_end_heading = end_heading

        self.corrections = 0

        self.turn_start_yaw = (
            self.imu_yaw if self.imu_fresh() else None
        )

        duration = self.turn_duration(angle)

        self.set_state('TIME_TURN', duration)

        self.get_logger().info(
            f'{kind}: target={math.degrees(angle):.1f}°, '
            f'duration={duration:.2f}s'
        )

    def measured_turn(self):
        if (
            self.turn_start_yaw is None
            or not self.imu_fresh()
        ):
            return None

        return wrap_angle(
            self.imu_yaw - self.turn_start_yaw
        )

    def finish_turn(self):
        self.stop()

        # Keep the planned heading estimate.
        # Do not overwrite it with uncalibrated IMU yaw.
        self.theta = self.turn_end_heading

        if self.turn_kind == 'TURN':
            self.set_state(
                'PAUSE_AFTER_TURN',
                self.pause_duration
            )
        else:
            self.finish_waypoint()

    def finish_waypoint(self):
        self.get_logger().info(
            f'Waypoint {self.current_index + 1} completed'
        )

        self.current_index += 1
        self.set_state('START')

    def control_loop(self):
        elapsed = time.monotonic() - self.state_start

        if self.state in ('DONE', 'ABORTED'):
            self.stop()
            return

        if self.state == 'START':
            self.stop()

            if self.current_index >= len(self.waypoints):
                self.state = 'DONE'
                self.get_logger().info(
                    'All waypoints completed'
                )
                return

            tx, ty, ttheta = self.waypoints[
                self.current_index
            ]

            self.target_x = tx
            self.target_y = ty
            self.target_theta = ttheta

            dx = tx - self.x
            dy = ty - self.y

            self.distance = math.hypot(dx, dy)

            heading = (
                math.atan2(dy, dx)
                if self.distance > 0.001
                else self.theta
            )

            angle = wrap_angle(
                heading - self.theta
            )

            self.get_logger().info(
                f'Waypoint {self.current_index + 1}: '
                f'distance={self.distance:.2f}m, '
                f'turn={math.degrees(angle):.1f}°'
            )

            if abs(angle) > self.angle_threshold:
                self.start_turn(
                    'TURN', angle, heading
                )
            else:
                self.set_state(
                    'PAUSE_AFTER_TURN', 0.0
                )

            return

        # ===== Initial time-based turn =====
        if self.state == 'TIME_TURN':
            if elapsed >= self.duration:
                self.stop()
                self.set_state(
                    'WAIT_FOR_IMU',
                    self.correction_pause
                )
                return

            direction = (
                1.0 if self.turn_target > 0
                else -1.0
            )

            command = (
                direction * self.turn_command
            )

            self.motor(-command, command)
            return

        # ===== IMU check =====
        
        if self.state == 'WAIT_FOR_IMU':
            self.stop()

            if elapsed < self.duration:
                return

            # Time-based turning is the primary controller.
            # If IMU correction is disabled, do not rely on IMU.
            if not self.enable_imu_correction:
                self.get_logger().info(
                    'Timed turn completed; IMU correction disabled'
                )
                self.finish_turn()
                return

            actual = self.measured_turn()

            if actual is None:
                self.get_logger().warn(
                    'IMU unavailable; keeping timed turn'
                )
                self.finish_turn()
                return

            error = wrap_angle(
                self.turn_target - actual
            )

            self.get_logger().info(
                f'IMU measured={math.degrees(actual):.1f}°, '
                f'error={math.degrees(error):.1f}°'
            )

            if abs(error) <= self.angle_threshold:
                self.finish_turn()
                return

            if self.corrections >= self.max_corrections:
                self.get_logger().warn(
                    'Correction limit reached; continuing'
                )
                self.finish_turn()
                return

            if abs(error) > self.max_correction_angle:
                self.get_logger().warn(
                    'IMU error too large; skipping correction'
                )
                self.finish_turn()
                return

            self.correction_error = error
            self.corrections += 1

            correction_time = min(
                self.turn_duration(error),
                self.max_correction_duration
            )

            self.set_state(
                'CORRECT_TURN',
                correction_time
            )

            self.get_logger().info(
                f'Correcting {math.degrees(error):.1f}° '
                f'for {correction_time:.2f}s'
            )
            return

        # ===== One bounded correction =====
        if self.state == 'CORRECT_TURN':
            if elapsed >= self.duration:
                self.stop()
                self.set_state(
                    'WAIT_FOR_IMU',
                    self.correction_pause
                )
                return

            direction = (
                1.0 if self.correction_error > 0
                else -1.0
            )

            command = (
                direction * self.turn_command
            )

            self.motor(-command, command)
            return

        # ===== Drive forward =====
        if self.state == 'PAUSE_AFTER_TURN':
            self.stop()

            if elapsed >= self.duration:
                drive_time = (
                    self.distance / self.forward_speed
                    if self.distance > 0.001
                    else 0.0
                )

                self.set_state(
                    'DRIVE', drive_time
                )
            return

        if self.state == 'DRIVE':
            if elapsed < self.duration:
                self.motor(
                    self.forward_command,
                    self.forward_command
                )
            else:
                self.stop()

                self.x = self.target_x
                self.y = self.target_y

                self.set_state(
                    'PAUSE_AFTER_DRIVE',
                    self.pause_duration
                )
            return

        # ===== Final waypoint orientation =====
        if self.state == 'PAUSE_AFTER_DRIVE':
            self.stop()

            if elapsed < self.duration:
                return

            # angle = wrap_angle(
            #     self.target_theta - self.theta
            # )

            # if abs(angle) > self.angle_threshold:
            #     self.start_turn(
            #         'FINAL_TURN',
            #         angle,
            #         self.target_theta
            #     )
            # else:
            #     self.finish_waypoint()
            self.finish_waypoint()

            return


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = WaypointController()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            try:
                if rclpy.ok():
                    node.stop()
            except Exception:
                pass

            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
