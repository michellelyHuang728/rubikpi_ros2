
#!/usr/bin/env python3

import math
import time
import csv
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


class TurnCalibrator(Node):

    def __init__(self):
        super().__init__("turn_calibrator")

        self.publisher = self.create_publisher(
            Float32MultiArray, "/motor_commands", 10
        )

        self.subscription = self.create_subscription(
            Imu, "/imu_data", self.imu_callback, 10
        )

        self.yaw = None
        self.last_yaw = None
        self.unwrapped_yaw = None
        self.last_imu_time = None

        self.samples = []
        self.start_time = time.monotonic()

    def imu_callback(self, msg):
        yaw = quaternion_to_yaw(msg.orientation)

        if self.last_yaw is None:
            self.unwrapped_yaw = yaw
        else:
            delta = normalize_angle(yaw - self.last_yaw)
            self.unwrapped_yaw += delta

        self.last_yaw = yaw
        self.yaw = self.unwrapped_yaw
        self.last_imu_time = time.monotonic()

    def send_command(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.publisher.publish(msg)

    def collect(self, duration, left=0.0, right=0.0, phase="idle"):
        start = time.monotonic()
        next_publish = start

        while time.monotonic() - start < duration:
            now = time.monotonic()

            if now >= next_publish:
                self.send_command(left, right)
                next_publish = now + 0.05

            rclpy.spin_once(self, timeout_sec=0.01)

            if self.yaw is not None:
                self.samples.append({
                    "time": time.monotonic() - self.start_time,
                    "phase": phase,
                    "left": left,
                    "right": right,
                    "yaw_rad": self.yaw,
                    "yaw_deg": math.degrees(self.yaw),
                })

    def phase_samples(self, phase):
        return [s for s in self.samples if s["phase"] == phase]

    def save_csv(self):
        filename = Path.home() / "turn_calibration.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time", "phase", "left", "right",
                    "yaw_rad", "yaw_deg"
                ]
            )
            writer.writeheader()
            writer.writerows(self.samples)

        print(f"CSV saved: {filename}")

    def calibrate(self, left=0.2, right=0.3, duration=3.0):
        print("Waiting for ROS connections...")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)

            fresh_imu = (
                self.last_imu_time is not None
                and time.monotonic() - self.last_imu_time < 0.5
            )

            if self.publisher.get_subscription_count() > 0 and fresh_imu:
                break

        print("Measuring stationary IMU drift (5 seconds)...")
        self.collect(5.0, phase="before")

        before = self.phase_samples("before")
        before_start = before[0]
        before_end = before[-1]

        before_dt = before_end["time"] - before_start["time"]
        before_delta = before_end["yaw_rad"] - before_start["yaw_rad"]

        drift_rate = before_delta / before_dt

        print(
            f"Stationary drift: "
            f"{math.degrees(drift_rate):+.4f} deg/s"
        )

        print(f"Running motors: L={left}, R={right}")
        self.collect(duration, left, right, phase="turn")

        print("Stopping motors...")
        self.collect(1.0, phase="after")

        turn = self.phase_samples("turn")

        turn_start = turn[0]
        turn_end = turn[-1]

        turn_dt = turn_end["time"] - turn_start["time"]
        raw_delta = turn_end["yaw_rad"] - turn_start["yaw_rad"]

        corrected_delta = raw_delta - drift_rate * turn_dt
        corrected_omega = corrected_delta / turn_dt

        print()
        print("========== CALIBRATION RESULT ==========")
        print(f"Command: [{left}, {right}]")
        print(f"Duration: {turn_dt:.3f} s")
        print(f"Raw rotation: {math.degrees(raw_delta):+.2f} deg")
        print(
            f"Corrected rotation: "
            f"{math.degrees(corrected_delta):+.2f} deg"
        )
        print(
            f"Corrected angular velocity: "
            f"{corrected_omega:+.4f} rad/s"
        )
        print("========================================")

        self.save_csv()


def main():
    rclpy.init()
    node = TurnCalibrator()

    try:
        node.calibrate(left=0.2, right=0.3, duration=3.0)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        for _ in range(10):
            node.send_command(0.0, 0.0)
            time.sleep(0.05)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
