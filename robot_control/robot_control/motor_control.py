import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import serial
import json
import time
import threading
import numpy as np
import math

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) in degrees to quaternion
    
    Args:
        roll (float): Roll angle in degrees (rotation around X-axis)
        pitch (float): Pitch angle in degrees (rotation around Y-axis)  
        yaw (float): Yaw angle in degrees (rotation around Z-axis)
    
    Returns:
        tuple: (w, x, y, z) quaternion components
    """
    # Convert degrees to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Calculate half angles
    cy = math.cos(yaw_rad * 0.5)    # cos(yaw/2)
    sy = math.sin(yaw_rad * 0.5)    # sin(yaw/2)
    cp = math.cos(pitch_rad * 0.5)  # cos(pitch/2)
    sp = math.sin(pitch_rad * 0.5)  # sin(pitch/2)
    cr = math.cos(roll_rad * 0.5)   # cos(roll/2)
    sr = math.sin(roll_rad * 0.5)   # sin(roll/2)
    
    # Quaternion multiplication (ZYX rotation order)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return w, x, y, z

class MotorControllerNode(Node):
    def __init__(self):
        super().__init__('motor_controller_node')
        
        # create subscription to motor commands
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'motor_commands',
            self.motor_command_callback,
            10
        )
        
        # create publisher for IMU data
        self.imu_publisher = self.create_publisher(Imu, 'imu_data', 10)
        
        # initialize serial connection
        self.robot = None
        self.init_serial()
        
        # motor command state
        self.motor_command = {"T": 1, "L": 0.0, "R": 0.0}
        self.imu_request_command = {"T": 126}
        self.running = True
        
        # safety timeout variables
        self.last_command_time = time.time()
        self.timeout_duration = 0.15  # 0.15 second timeout
        self.command_lock = threading.Lock()  # thread safety for command updates
        self.timeout_active = False  # flag to track if we're in timeout state
        
        # start threads
        self.motor_tx_thread = threading.Thread(target=self.send_motor_commands, daemon=True)
        self.motor_tx_thread.start()
        
        self.imu_tx_thread = threading.Thread(target=self.send_imu_requests, daemon=True)
        self.imu_tx_thread.start()
        
        self.rx_thread = threading.Thread(target=self.receive_data, daemon=True)
        self.rx_thread.start()
        
        self.get_logger().info('Motor Controller Node with IMU started')
        self.get_logger().info('Listening for motor commands on topic: motor_commands')
        self.get_logger().info('Publishing IMU data on topic: imu_data')
        self.get_logger().info(f'Safety timeout set to {self.timeout_duration} seconds')
        self.get_logger().info('Requesting IMU data at 50Hz')
        
    def init_serial(self):
        """initialize serial connection"""
        try:
            self.robot = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.1)
            self.get_logger().info('Serial connection established on /dev/ttyUSB0')
        except Exception as e:
            self.get_logger().warn(f'Failed to establish serial connection: {str(e)}')
            self.robot = None
    
    def motor_command_callback(self, msg):
        """receive motor commands from topic"""
        try:
            if len(msg.data) < 2:
                self.get_logger().warn('Motor command array must have at least 2 elements [L, R]')
                return
            
            L = float(msg.data[0])
            R = float(msg.data[1])
            
            # clip values to valid range
            L = np.clip(L, -0.5, 0.5)
            R = np.clip(R, -0.5, 0.5)
            
            # update command with thread safety
            with self.command_lock:
                self.motor_command = {"T": 1, "L": L, "R": R}
                self.last_command_time = time.time()
                
                # log if we're recovering from timeout
                if self.timeout_active:
                    self.get_logger().info('Motor commands resumed - exiting safety timeout')
                    self.timeout_active = False
            
            self.get_logger().debug(f'Motor command received: L={L:.2f}, R={R:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing motor command: {str(e)}')
    
    def check_timeout(self):
        """check if commands have timed out and apply safety stop if needed"""
        current_time = time.time()
        
        with self.command_lock:
            time_since_last_command = current_time - self.last_command_time
            
            if time_since_last_command > self.timeout_duration:
                # timeout occurred - set motors to stop
                if not self.timeout_active:
                    self.get_logger().warn(f'No motor commands received for {time_since_last_command:.1f}s - applying safety stop')
                    self.timeout_active = True
                
                # force stop command
                self.motor_command = {"T": 1, "L": 0.0, "R": 0.0}
                return True
            
        return False
    
    def send_motor_commands(self):
        """send motor commands to robot via serial at ~20Hz"""
        while self.running and rclpy.ok():
            # check for timeout before sending
            self.check_timeout()
            
            if self.robot:
                try:
                    # get current command safely
                    with self.command_lock:
                        current_command = self.motor_command.copy()
                    
                    command_str = json.dumps(current_command) + '\n'
                    self.robot.write(command_str.encode('utf-8'))
                    
                except Exception as e:
                    self.get_logger().error(f'Motor command serial write error: {str(e)}')
                    self.init_serial()
            time.sleep(0.05)  # 20Hz
    
    def send_imu_requests(self):
        """send IMU data requests to robot via serial at 50Hz"""
        while self.running and rclpy.ok():
            if self.robot:
                try:
                    command_str = json.dumps(self.imu_request_command) + '\n'
                    self.robot.write(command_str.encode('utf-8'))
                    
                except Exception as e:
                    self.get_logger().error(f'IMU request serial write error: {str(e)}')
                    # Don't reinit serial here to avoid conflict with motor thread
            time.sleep(0.02)  # 50Hz
    
    def receive_data(self):
        """receive and process data from robot"""
        buffer = ""
        while self.running and rclpy.ok():
            if self.robot:
                try:
                    # read available data
                    data = self.robot.read(1024)
                    if data:
                        buffer += data.decode('utf-8', errors='ignore')
                        
                        # process complete lines
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if line:
                                self.process_received_data(line)
                                
                except Exception as e:
                    self.get_logger().error(f'Serial read error: {str(e)}')
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
    
    def process_received_data(self, data_str):
        """process received data from robot"""
        try:
            data = json.loads(data_str)
            
            # check if this is IMU data (T: 1002)
            if isinstance(data, dict) and data.get('T') == 1002:
                self.process_imu_data(data)
                
        except json.JSONDecodeError:
            self.get_logger().debug(f'Received non-JSON data: {data_str}')
        except Exception as e:
            self.get_logger().error(f'Error processing received data: {str(e)}')
    
    def process_imu_data(self, imu_data):
        """process and publish IMU data"""
        try:
            # extract roll, pitch, yaw
            print(imu_data)
            roll = float(imu_data.get('r', 0))    # degrees
            pitch = float(imu_data.get('p', 0))   # degrees  
            yaw = float(imu_data.get('y', 0))     # degrees
            
            # convert to quaternion
            qw, qx, qy, qz = euler_to_quaternion(roll, pitch, yaw)
    
            imu_msg = Imu()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = "imu_link"
            
            # set orientation quaternion
            imu_msg.orientation.w = qw
            imu_msg.orientation.x = qx
            imu_msg.orientation.y = qy
            imu_msg.orientation.z = qz
            
            # angular velocity (rad/s) - convert from degrees/s if needed
            # imu_msg.angular_velocity.x = float(imu_data.get('gx', 0))  # roll rate
            # imu_msg.angular_velocity.y = float(imu_data.get('gy', 0))  # pitch rate  
            # imu_msg.angular_velocity.z = float(imu_data.get('gz', 0))  # yaw rate
            imu_msg.angular_velocity.x = math.radians(float(imu_data.get('gx', 0)))
            imu_msg.angular_velocity.y = math.radians(float(imu_data.get('gy', 0)))
            imu_msg.angular_velocity.z = math.radians(float(imu_data.get('gz', 0)))
            
            # linear acceleration (m/s^2) - convert from mg to m/s^2 if needed
            imu_msg.linear_acceleration.x = float(imu_data.get('ax', 0)) * 9.81 / 1000.0
            imu_msg.linear_acceleration.y = float(imu_data.get('ay', 0)) * 9.81 / 1000.0
            imu_msg.linear_acceleration.z = float(imu_data.get('az', 0)) * 9.81 / 1000.0
            
            self.imu_publisher.publish(imu_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {str(e)}')
    
    def destroy_node(self):
        """Cleanup operations when node is destroyed"""
        self.running = False
        
        # Send stop command before closing (only if ROS context is still valid)
        try:
            if self.robot and rclpy.ok():
                stop_command = json.dumps({"T": 1, "L": 0.0, "R": 0.0}) + '\n'
                self.robot.write(stop_command.encode('utf-8'))
                self.get_logger().info('Robot stopped')
        except Exception as e:
            # Ignore errors during shutdown as context may already be invalid
            pass
        
        # Close serial connection
        try:
            if self.robot:
                self.robot.close()
                self.get_logger().info('Serial connection closed')
        except Exception as e:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass
        return 0

if __name__ == '__main__':
    main()