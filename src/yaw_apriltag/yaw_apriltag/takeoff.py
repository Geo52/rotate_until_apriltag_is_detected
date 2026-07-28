"""ArduPilot GUIDED: takeoff to 1 m, spin in place until an AprilTag is seen.

Stops the rotation after N consecutive frames containing a tag, then holds
(hovers with zero-velocity setpoints) until you switch modes (e.g. LAND).
Also stops if no tag is found after MAX_SPIN_DEG of rotation, or if the
mode changes / vehicle disarms at any point.

Usage: python3 takeoff_yaw_tag.py [yaw_rate_deg_s]   (default 30, positive = CW)
"""
import math
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from apriltag_msgs.msg import AprilTagDetectionArray

TAKEOFF_ALT = 1.0        # meters
SETPOINT_HZ = 20.0       # must stay above ~2 Hz or ArduPilot stops the spin
TAG_CONFIRM_FRAMES = 3   # consecutive detection frames required to stop
MAX_SPIN_DEG = 1080.0     # give up after slightly more than one full rotation


class TakeoffYaw(Node):
    def __init__(self, yaw_rate_deg: float):
        super().__init__("takeoff_yaw")
        # ENU: positive angular.z is counter-clockwise, so negate for CW-positive input
        self.yaw_rate = -math.radians(yaw_rate_deg)
        self.state = State()
        self.pose = None
        self.tag_streak = 0
        self.tag_found = False

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(State, "/mavros/state", self._state_cb, qos)
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose", self._pose_cb, qos
        )
        self.create_subscription(
            AprilTagDetectionArray, "/detections", self._tag_cb, 10
        )

        self.vel_pub = self.create_publisher(
            TwistStamped, "/mavros/setpoint_velocity/cmd_vel", 10
        )

        self.mode_cli = self.create_client(SetMode, "/mavros/set_mode")
        self.arm_cli = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.tol_cli = self.create_client(CommandTOL, "/mavros/cmd/takeoff")

    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.pose = msg

    def _tag_cb(self, msg):
        if msg.detections:
            self.tag_streak += 1
            if self.tag_streak >= TAG_CONFIRM_FRAMES and not self.tag_found:
                self.tag_found = True
                ids = [d.id for d in msg.detections]
                self.get_logger().info(f"AprilTag confirmed (ids={ids})")
        else:
            self.tag_streak = 0

    def _call(self, cli, req):
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"waiting for {cli.srv_name}...")
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def _publish_twist(self, yaw_rate: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.z = yaw_rate
        self.vel_pub.publish(msg)

    def _pilot_took_over(self) -> bool:
        if self.state.mode != "GUIDED":
            self.get_logger().info(f"Mode changed to {self.state.mode}, stopping")
            return True
        if not self.state.armed:
            self.get_logger().info("Disarmed, stopping")
            return True
        return False

    def run(self):
        # Wait for FCU + first pose
        while rclpy.ok() and (not self.state.connected or self.pose is None):
            rclpy.spin_once(self, timeout_sec=0.5)
        z0 = self.pose.pose.position.z
        self.get_logger().info(f"FCU connected, ground z = {z0:.2f}")

        res = self._call(self.mode_cli, SetMode.Request(custom_mode="GUIDED"))
        if not res.mode_sent:
            self.get_logger().error("Failed to set GUIDED")
            return

        res = self._call(self.arm_cli, CommandBool.Request(value=True))
        if not res.success:
            self.get_logger().error("Arming failed (check pre-arm messages)")
            return
        self.get_logger().info("Armed")

        res = self._call(self.tol_cli, CommandTOL.Request(altitude=TAKEOFF_ALT))
        if not res.success:
            self.get_logger().error("Takeoff rejected")
            return
        self.get_logger().info("Taking off...")

        # Wait until we've climbed ~1 m relative to start (timeout 15 s)
        t_start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            dz = self.pose.pose.position.z - z0
            if dz >= TAKEOFF_ALT * 0.90:
                break
            if time.time() - t_start > 15.0:
                self.get_logger().warn(
                    f"Timeout waiting for altitude (dz={dz:.2f}), spinning anyway"
                )
                break

        self.get_logger().info(
            f"At altitude. Spinning at {math.degrees(-self.yaw_rate):.0f} deg/s "
            "until a tag is seen - switch to LAND to abort."
        )

        # Spin until tag confirmed, max rotation reached, or pilot takes over
        period = 1.0 / SETPOINT_HZ
        max_spin_s = MAX_SPIN_DEG / abs(math.degrees(self.yaw_rate))
        spin_start = time.time()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            if self._pilot_took_over():
                return
            if self.tag_found:
                break
            if time.time() - spin_start > max_spin_s:
                self.get_logger().warn(
                    f"No tag after {MAX_SPIN_DEG:.0f} deg of rotation, stopping"
                )
                break
            self._publish_twist(self.yaw_rate)
            time.sleep(period)

        # Stop rotating and hold position until the pilot takes over (e.g. LAND)
        self.get_logger().info("Holding. Switch to LAND when ready.")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)
            if self._pilot_took_over():
                return
            self._publish_twist(0.0)
            time.sleep(period)


def main():
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    rclpy.init()
    node = TakeoffYaw(rate)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()