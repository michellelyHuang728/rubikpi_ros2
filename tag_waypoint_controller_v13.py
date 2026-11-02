#!/usr/bin/env python3
"""ROS 2 AprilTag pixel-geometry follower V12 with IMU heading. Dry run unless --arm.

Uses apparent tag size and center (not unstable solvePnP depth). Visual target
following only; does not establish world-frame waypoint pose or yaw.
"""
import argparse
import csv
import math
import time
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from apriltag_msgs.msg import AprilTagDetectionArray
from sensor_msgs.msg import CameraInfo, Imu
from std_msgs.msg import Float32MultiArray


class Controller(Node):
    def __init__(self, a):
        super().__init__('tag_waypoint_controller_v13')
        self.a = a
        self.tags = a.tags
        self.index = 0
        self.phase = None
        self.started = time.monotonic()
        self.last_seen = None
        self.command = (0., 0.)
        self.searching = False
        self.search_started = None
        self.confirmed = 0
        self.arrival_confirmed = 0
        self.align_started = None
        self.align_direction = 0
        self.align_pulse_complete = False
        self.align_wait_until = 0.0
        self.align_turn_total = 0.0
        self.align_turn_start = 0.0
        self.finished = False
        self.fx = self.fy = self.cx = None
        self.last_warning = 0.
        # Command-integrated planar dead reckoning. These gains MUST be calibrated.
        self.x = self.y = self.theta = 0.0
        self.path_m = 0.0
        self.turn_abs_rad = 0.0
        self.odom_last = time.monotonic()
        self.last_pose_log = self.odom_last
        self.pose_updates = 0
        # IMU heading is relative to the first received quaternion, not world north.
        self.imu_yaw_prev = None
        self.imu_yaw_total = 0.0
        self.imu_yaw_at_odom = 0.0
        self.imu_last_time = None
        self.search_turn_abs = 0.0
        self.search_turn_signed = 0.0
        self.imu_count = 0
        self.search_candidate_frames = 0
        self.landmarks = {}
        if a.landmark_map:
            with open(a.landmark_map, newline='') as lm_file:
                for row in csv.DictReader(lm_file):
                    self.landmarks[int(row['id'])] = (float(row['x_m']), float(row['y_m']))
        self.pose_file = open(a.pose_log, 'w', newline='')
        self.pose_csv = csv.writer(self.pose_file)
        self.pose_csv.writerow(['elapsed_s','x_m','y_m','theta_rad','theta_deg','distance_traveled_m','absolute_turn_rad','target_id','phase','left_command','right_command','pose_source','tag_updates','imu_age_s','search_turn_abs_rad'])
        self.events_file = open(a.events_log, 'w', newline='')
        self.events_csv = csv.writer(self.events_file)
        self.events_csv.writerow(['elapsed_s','tag_id','event','estimated_x_m','estimated_y_m','estimated_yaw_deg','measured_tag_depth_m','bearing_rad','range_error_m','bearing_error_rad','pose_source'])
        self.pub = self.create_publisher(Float32MultiArray, '/motor_commands', 10)
        self.create_subscription(Imu, '/imu_data', self.on_imu, 50)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_camera, 10)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.on_detections, 10)
        self.create_timer(.05, self.tick)
        self.file = open(a.log, 'w', newline='')
        self.csv = csv.writer(self.file)
        self.csv.writerow(['elapsed_s', 'tag_id', 'phase', 'pixel_side', 'size_distance_m', 'angle_rad', 'left', 'right'])
        self.get_logger().info(f"{'ARMED' if a.arm else 'DRY RUN'}: tags={a.tags}, stop={a.stop_distance:.2f}m, in-place spin power={a.search_turn_command:.2f}")

    def on_camera(self, m):
        if m.k[0] > 0 and m.k[4] > 0:
            self.fx, self.fy, self.cx = float(m.k[0]), float(m.k[4]), float(m.k[2])

    @staticmethod
    def wrap(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def on_imu(self, msg):
        q = msg.orientation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if not math.isfinite(norm) or norm < 0.5:
            return
        x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        now = time.monotonic()
        if self.imu_yaw_prev is not None:
            delta = self.wrap(yaw - self.imu_yaw_prev)
            # Reject implausible quaternion jumps rather than corrupting the path.
            if abs(delta) > self.a.max_imu_step:
                self.get_logger().warning('IMU yaw jump rejected')
                self.imu_yaw_prev = yaw
                self.imu_last_time = now
                return
            self.imu_yaw_total += delta * self.a.imu_yaw_sign
            if self.phase == 'ALIGN_SPIN':
                self.align_turn_total += abs(delta)
            if self.searching:
                self.search_turn_abs += abs(delta)
                self.search_turn_signed += delta * self.a.imu_yaw_sign
        self.imu_yaw_prev = yaw
        self.imu_last_time = now
        self.imu_count += 1

    def imu_fresh(self, now):
        return self.imu_last_time is not None and now - self.imu_last_time <= self.a.max_imu_age

    def integrate_pose(self, now):
        dt = max(0., min(now-self.odom_last, .20))
        self.odom_last = now
        if not self.a.arm or self.finished or not self.imu_fresh(now):
            self.imu_yaw_at_odom = self.imu_yaw_total
            return
        l, r = self.command
        vl = self.a.left_speed_gain * l
        vr = self.a.right_speed_gain * r
        v = .5*(vl+vr)
        new_theta = self.imu_yaw_total
        dtheta = new_theta - self.imu_yaw_at_odom
        heading_mid = self.imu_yaw_at_odom + .5*dtheta
        self.x += v*dt*math.cos(heading_mid)
        self.y += v*dt*math.sin(heading_mid)
        self.theta = self.wrap(new_theta)
        self.imu_yaw_at_odom = new_theta
        self.path_m += abs(v)*dt
        self.turn_abs_rad += abs(dtheta)

    def log_pose(self, now):
        source = 'tag_corrected_xy+imu_heading' if self.pose_updates else 'command_translation+imu_heading'
        self.pose_csv.writerow([round(now-self.started,3), round(self.x,4), round(self.y,4), round(self.theta,5), round(math.degrees(self.theta),2), round(self.path_m,4), round(self.turn_abs_rad,4), self.tags[self.index] if not self.finished else '', self.phase, round(self.command[0],3), round(self.command[1],3), source, self.pose_updates, round(now-self.imu_last_time,3) if self.imu_last_time is not None else '', round(self.search_turn_abs,4)])
        self.pose_file.flush()

    def correct_position_from_tag(self, tag_id, distance, bearing):
        # Known landmark world XY and current heading provide an XY observation.
        # Pixel-size depth is approximate; heading is NOT observed by this update.
        if tag_id not in self.landmarks or not self.a.enable_tag_correction:
            return
        tx,ty = self.landmarks[tag_id]
        # Approximate optical depth -> horizontal ray range (ignores tag pitch).
        ray_range = distance / max(.2, math.cos(bearing))
        camera_heading = self.theta + bearing
        cam_x = tx - ray_range*math.cos(camera_heading)
        cam_y = ty - ray_range*math.sin(camera_heading)
        bx = cam_x - self.a.camera_forward_offset*math.cos(self.theta)
        by = cam_y - self.a.camera_forward_offset*math.sin(self.theta)
        innovation = math.hypot(bx-self.x, by-self.y)
        if innovation <= self.a.max_tag_innovation:
            alpha = self.a.tag_position_alpha
            self.x = (1-alpha)*self.x+alpha*bx
            self.y = (1-alpha)*self.y+alpha*by
            self.pose_updates += 1

    def halt(self, reason):
        self.finished = True
        self.command = (0., 0.)
        self.motors()
        self.get_logger().error(reason + '; stopped')

    def motors(self, left=0., right=0.):
        m = Float32MultiArray()
        m.data = [float(left), float(right)] if self.a.arm and self.imu_fresh(time.monotonic()) and not self.finished else [0., 0.]
        self.pub.publish(m)

    def phase_to(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.get_logger().info(f'Tag {self.tags[self.index]}: {phase}')

    def on_detections(self, msg):
        if self.finished or self.fx is None:
            return
        target = self.tags[self.index]
        d = next((d for d in msg.detections if int(d.id) == target), None)
        if d is None:
            self.arrival_confirmed = 0
            self.confirmed = 0 if self.searching else self.confirmed
            return
        pts = np.asarray([[p.x, p.y] for p in d.corners], dtype=float)
        if pts.shape != (4, 2) or not np.isfinite(pts).all():
            return
        sides = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        side = float(np.median(sides))
        if side < 12. or max(sides) > 2.0 * min(sides):
            return
        # Approximate fronto-parallel depth from observed tag side in pixels.
        # This is a visual stop proxy, not a calibrated 3-D pose estimate.
        distance = self.a.tag_size * math.sqrt(self.fx * self.fy) / side
        center_x = float(np.mean(pts[:, 0]))
        angle = math.atan2(center_x - self.cx, self.fx)
        now = time.monotonic()
        if self.searching:
            self.search_candidate_frames += 1
            self.get_logger().info(f'Search candidate Tag {target}: frame={self.search_candidate_frames}, confirmations={self.confirmed + 1}/{self.a.search_confirmations}, angle={math.degrees(angle):.1f} deg, distance={distance:.2f} m')
            self.confirmed += 1
            if self.confirmed < self.a.search_confirmations:
                self.command = (0., 0.)
                self.motors()
                return
            self.searching = False
            self.search_started = None
            self.align_started = None
            self.align_pulse_complete = False
            self.command = (0., 0.)
            self.motors()
            self.get_logger().info(f'Tag {target} found; stopping search rotation')
        self.last_seen = now
        self.correct_position_from_tag(target, distance, angle)
        self.csv.writerow([round(now-self.started, 3), target, self.phase, round(side, 2), round(distance, 3), round(angle, 4), *self.command])
        self.file.flush()
        if distance <= self.a.stop_distance + self.a.distance_tolerance and abs(angle) <= self.a.align_threshold:
            self.arrival_confirmed += 1
            self.command = (0., 0.)
            self.align_started = None
            self.align_pulse_complete = False
            self.motors()
            if self.arrival_confirmed >= self.a.arrival_confirmations:
                self.get_logger().info(f'Tag {target}: ARRIVED (size-based z≈{distance:.3f}m)')
                self.events_csv.writerow([round(now-self.started,3), target, 'ARRIVED', round(self.x,4), round(self.y,4), round(math.degrees(self.theta),2), round(distance,4), round(angle,4), round(distance-self.a.stop_distance,4), round(angle,4), 'estimated_not_ground_truth'])
                self.events_file.flush()
                self.index += 1
                self.last_seen = None
                self.arrival_confirmed = 0
                if self.index == len(self.tags):
                    self.integrate_pose(now)
                    self.finished = True
                    self.get_logger().info('FINISHED visual targets; world-frame waypoint poses NOT verified; trajectory is estimated')
                else:
                    self.searching = True
                    self.search_started = now
                    self.search_candidate_frames = 0
                    self.search_turn_abs = 0.0
                    self.search_turn_signed = 0.0
                    self.confirmed = 0
                    self.get_logger().info(f'Next target Tag {self.tags[self.index]}: SEARCH_SPIN')
                    self.phase_to('SEARCH_SPIN')
            else:
                self.phase_to('ARRIVAL_CONFIRM')
            return
        self.arrival_confirmed = 0
        turn = max(-self.a.turn_speed, min(self.a.turn_speed, self.a.turn_gain * angle))
        if abs(angle) > self.a.align_threshold:
            # Positive image bearing means target to the right; pivot clockwise.
            direction = 1 if angle > 0 else -1
            # One bounded pulse per new vision observation, never free-running.
            if self.phase != 'ALIGN_SPIN':
                self.align_turn_total = 0.0
                self.align_pulse_complete = False
                self.align_started = None
            if self.align_turn_total >= self.a.max_align_turn_rad:
                self.halt(f'Alignment exceeded turn budget ({math.degrees(self.align_turn_total):.1f} deg)')
                return
            self.align_direction = direction
            self.phase_to('ALIGN_SPIN')
            if self.align_pulse_complete and now >= self.align_wait_until:
                self.align_started = None
                self.align_pulse_complete = False
            if self.align_started is None and not self.align_pulse_complete:
                self.align_started = now
                self.align_turn_start = self.imu_yaw_total
            self.command = (0., 0.)
            self.motors()
            return
        else:
            self.align_started = None
            self.align_direction = 0
            self.align_pulse_complete = False
            self.phase_to('APPROACH')
            speed = self.a.forward_speed
            left, right = speed + turn, speed - turn
        self.command = (max(-self.a.max_command, min(self.a.max_command, left)),
                        max(-self.a.max_command, min(self.a.max_command, right)))
        self.motors(*self.command)

    def tick(self):
        now = time.monotonic()
        self.integrate_pose(now)
        if now-self.last_pose_log >= self.a.pose_interval:
            self.log_pose(now)
            self.last_pose_log = now
        if self.finished:
            self.motors()
            return
        if now - self.started >= self.a.max_runtime:
            self.finished = True
            self.command = (0., 0.)
            self.motors()
            self.get_logger().error('Maximum runtime reached; stopped')
            return
        if not self.imu_fresh(now):
            self.command = (0., 0.)
            self.motors()
            self.phase_to('WAIT_IMU')
            if now-self.last_warning > 4.:
                self.get_logger().warning('IMU missing/stale: motors stopped')
                self.last_warning = now
            return
        if self.searching:
            if self.search_turn_abs >= self.a.max_search_turn_rad:
                self.finished = True
                self.command = (0., 0.)
                self.motors()
                self.get_logger().error(f'Search exceeded IMU turn budget ({math.degrees(self.search_turn_abs):.1f} deg); stopped. Tag candidate frames={self.search_candidate_frames}')
                return
            if now - self.search_started >= self.a.search_timeout:
                self.finished = True
                self.command = (0., 0.)
                self.motors()
                self.get_logger().error('Search timed out; stopped')
                return
            # In-place counter-rotating wheels; paused to let slow camera detect tags.
            cycle = self.a.search_pulse + self.a.search_pause
            in_pulse = self.confirmed == 0 and ((now - self.search_started) % cycle) < self.a.search_pulse
            power = self.a.search_turn_command
            self.command = (-power, power) if in_pulse else (0., 0.)
            self.motors(*self.command)
            return
        if self.last_seen is None or now-self.last_seen > self.a.max_observation_age:
            self.command = (0., 0.)
            self.align_started = None
            self.motors()
            self.phase_to('WAIT_TAG' if self.fx is not None else 'WAIT_CAMERA_INFO')
            if now-self.last_warning > 4.:
                self.get_logger().warning('No fresh tag detection; stopped')
                self.last_warning = now
        elif self.phase == 'ALIGN_SPIN':
            # End a pulse by time OR measured IMU angle; await a NEW camera frame.
            if self.align_started is None or self.align_pulse_complete:
                self.command = (0., 0.)
            elif (now - self.align_started >= self.a.align_pulse or
                  abs(self.imu_yaw_total - self.align_turn_start) >= self.a.max_align_pulse_rad):
                self.align_pulse_complete = True
                self.align_wait_until = now + self.a.align_pause
                self.command = (0., 0.)
            else:
                power = self.a.align_turn_command
                self.command = (power, -power) if self.align_direction > 0 else (-power, power)
            self.motors(*self.command)
        else:
            self.motors(*self.command)

    def destroy_node(self):
        if rclpy.ok():
            try:
                self.motors()
            except Exception as exc:
                self.get_logger().warning(f'Unable to publish final stop: {exc}')
        self.log_pose(time.monotonic())
        self.file.close()
        self.pose_file.close()
        self.events_file.close()
        return super().destroy_node()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', action='store_true')
    p.add_argument('--tags', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--tag-size', type=float, default=.12)
    p.add_argument('--stop-distance', type=float, default=.35)
    p.add_argument('--distance-tolerance', type=float, default=.05)
    p.add_argument('--align-threshold', type=float, default=.15)
    p.add_argument('--arrival-confirmations', type=int, default=2)
    p.add_argument('--forward-speed', type=float, default=.20)
    p.add_argument('--turn-speed', type=float, default=.08)
    p.add_argument('--turn-gain', type=float, default=.30)
    p.add_argument('--max-command', type=float, default=.45)
    p.add_argument('--search-turn-command', type=float, default=.30)
    p.add_argument('--align-turn-command', type=float, default=.30)
    p.add_argument('--align-pulse', type=float, default=.20)
    p.add_argument('--align-pause', type=float, default=.65)
    p.add_argument('--search-confirmations', type=int, default=2)
    p.add_argument('--search-pulse', type=float, default=.35)
    p.add_argument('--search-pause', type=float, default=.75)
    p.add_argument('--search-timeout', type=float, default=25.)
    p.add_argument('--max-observation-age', type=float, default=.8)
    p.add_argument('--max-runtime', type=float, default=150.)
    p.add_argument('--log', default=str(Path.home() / 'tag_navigation_v13_log.csv'))
    p.add_argument('--pose-log', default=str(Path.home() / 'tag_navigation_v13_trajectory.csv'))
    p.add_argument('--events-log', default=str(Path.home() / 'tag_navigation_v13_waypoints.csv'))
    p.add_argument('--pose-interval', type=float, default=.5)
    p.add_argument('--left-speed-gain', type=float, default=.0, help='Measured left wheel m/s per motor command; 0 means NOT calibrated')
    p.add_argument('--right-speed-gain', type=float, default=.0, help='Measured right wheel m/s per motor command; 0 means NOT calibrated')
    p.add_argument('--imu-yaw-sign', type=int, choices=(-1, 1), default=1,
                   help='Flip to -1 if IMU yaw increases opposite robot CCW rotation')
    p.add_argument('--max-imu-age', type=float, default=.5,
                   help='Stop motors if no IMU sample for this many seconds')
    p.add_argument('--max-imu-step', type=float, default=.7,
                   help='Reject a single implausibly large quaternion yaw jump (rad)')
    p.add_argument('--max-search-turn-rad', type=float, default=6.3,
                   help='Stop searching after this much accumulated IMU rotation (~413 deg)')
    p.add_argument('--max-align-pulse-rad', type=float, default=.20, help='Max IMU yaw change per alignment pulse (rad)')
    p.add_argument('--max-align-turn-rad', type=float, default=3.14, help='Total alignment turn budget per target (rad)')
    p.add_argument('--effective-track-width', type=float, default=.20, help='Effective skid-steer track width in meters; calibrate from turning')
    p.add_argument('--camera-forward-offset', type=float, default=.09)
    p.add_argument('--landmark-map', default='', help='Optional CSV with columns id,x_m,y_m in a measured common frame')
    p.add_argument('--enable-tag-correction', action='store_true', help='Use landmark map for approximate XY corrections; heading remains odometry only')
    p.add_argument('--tag-position-alpha', type=float, default=.15)
    p.add_argument('--max-tag-innovation', type=float, default=.5)
    a = p.parse_args()
    if not (0 < a.search_turn_command <= a.max_command <= .5 and 0 < a.align_turn_command <= a.max_command):
        p.error('Invalid search/max command')
    if min(a.search_pulse, a.search_pause, a.align_pulse, a.align_pause) <= 0:
        p.error('Search pulse and pause must be positive')
    if a.pose_interval <= 0 or a.effective_track_width <= 0 or a.left_speed_gain < 0 or a.right_speed_gain < 0 or not 0 <= a.tag_position_alpha <= 1:
        p.error('Invalid pose estimation parameters')
    if a.enable_tag_correction and not a.landmark_map:
        p.error('--enable-tag-correction requires --landmark-map')
    if a.max_imu_age <= 0 or a.max_imu_step <= 0 or a.max_search_turn_rad <= 0 or a.max_align_pulse_rad <= 0 or a.max_align_turn_rad <= 0:
        p.error('Invalid IMU safety parameters')
    if a.arrival_confirmations < 1 or a.search_confirmations < 1:
        p.error('Confirmations must be positive')
    rclpy.init()
    n = Controller(a)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
