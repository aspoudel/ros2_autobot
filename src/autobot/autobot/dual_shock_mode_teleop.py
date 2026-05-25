#!/usr/bin/env python3
"""
dual_shock_mode_teleop.py — Simple waypoint driver
===================================================
X   (btn 0) — drive to all waypoints in waypoints.txt, closest-first,
               arc-bypass obstacle avoidance, photo at each, return to (0,0)
O   (btn 1) — abort to MANUAL immediately
L2 + sticks — manual drive

POSE: EKF /odom_fused only (no SLAM)
"""

import collections
import math
import os
import threading
import time
from typing import List, Optional, Tuple

import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False


# ── Constants ──────────────────────────────────────────────────────────────────

OBSTACLE_DISTANCE_M  = 1.00    # arc-bypass triggers (m)
CHASSIS_MIN_RANGE_M  = 0.30    # ignore returns closer than this
FORWARD_CONE_DEG     = 30      # obstacle detection cone half-angle

MAX_LINEAR_SPEED     = 0.30    # m/s
MAX_ANGULAR_SPEED    = 0.60    # rad/s
HEADING_KP           = 1.2
DISTANCE_KP          = 0.6
CONTROL_DT           = 0.05    # 20 Hz
ANGLE_TOL_RAD        = math.radians(3.0)

WAYPOINT_TOLERANCE_M = 1.00    # reached within this radius (m)
SETTLE_TIME_S        = 3.0     # pause at each waypoint
RETURN_TOL_M         = 0.40    # close enough to call origin reached

BYPASS_SIDE_M        = 1.20    # lateral step during bypass
BYPASS_FWD_M         = 1.60    # forward step to clear obstacle

# Emergency stop
ESTOP_DISTANCE_M     = 0.25    # closer than this during driving = E-stop (m)
ESTOP_CONE_DEG       = 40      # wider cone for E-stop detection
ESTOP_BUFFER_SECS    = 5.0     # seconds of rolling buffer before E-stop
ESTOP_FPS            = 10      # frames per second in rolling buffer

CAMERA_SAVE_DIR = "/root/auto4508_ws/photos"
WAYPOINTS_FILE  = "/root/auto4508_ws/src/project1/config/waypoints.txt"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _norm(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def load_waypoints(filepath: str) -> List[Tuple[float, float]]:
    points = []
    if not os.path.exists(filepath):
        return points
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        points.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[load_waypoints] {e}")
    return points


# ── Node ───────────────────────────────────────────────────────────────────────

class DualShockModeTeleop(Node):

    MODE_MANUAL   = "manual"
    MODE_WAYPOINT = "waypoint"

    def __init__(self):
        super().__init__("dualshock_mode_teleop")
        self.cb_group = ReentrantCallbackGroup()

        # Pose — EKF only
        self.current_x   = 0.0
        self.current_y   = 0.0
        self.current_yaw = 0.0
        self.pose_valid  = False

        # Mode
        self.mode            = self.MODE_MANUAL
        self._mission_active = False

        # LiDAR
        self.lidar_ready    = False
        self.obstacle_ahead = False
        self.latest_scan: Optional[LaserScan] = None

        # Controller
        self.last_joy: Optional[Joy] = None

        # Camera
        self.photo_count  = 0
        self.camera_ready = False
        self.pipeline     = None
        self.q_rgb        = None

        # Rolling frame buffer for E-stop footage (deque of (timestamp, frame))
        self._frame_buffer    = collections.deque()
        self._estop_triggered = False
        self._estop_lock      = threading.Lock()

        # Locks
        self._lock        = threading.Lock()
        self._camera_lock = threading.Lock()

        # Publishers
        self.cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",      10)
        self.status_pub = self.create_publisher(String, "/robot_status", 10)

        # Subscriptions
        self.create_subscription(
            Joy,       "/joy",        self.joy_callback,   10,
            callback_group=self.cb_group)
        self.create_subscription(
            LaserScan, "/scan",       self.lidar_callback, 10,
            callback_group=self.cb_group)
        self.create_subscription(
            Odometry,  "/odom_fused", self.odom_callback,  10,
            callback_group=self.cb_group)

        self.create_timer(CONTROL_DT, self.control_loop,
                          callback_group=self.cb_group)

        self._init_camera()
        self._log("Ready | X=WAYPOINTS  O=ABORT  L2+sticks=MANUAL")

    # ── Pose ───────────────────────────────────────────────────────────────────

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        with self._lock:
            self.current_x   = msg.pose.pose.position.x
            self.current_y   = msg.pose.pose.position.y
            self.current_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.pose_valid  = True

    def _pose(self) -> Tuple[float, float, float]:
        with self._lock:
            return self.current_x, self.current_y, self.current_yaw

    # ── LiDAR ──────────────────────────────────────────────────────────────────

    def lidar_callback(self, msg: LaserScan):
        self.latest_scan = msg
        with self._lock:
            if not self.lidar_ready:
                self.lidar_ready = True
                self._log("LiDAR ready")

        cone_rad = math.radians(FORWARD_CONE_DEG)
        found = False
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            if abs(a) > cone_rad:
                continue
            if not math.isfinite(r):
                continue
            if r < CHASSIS_MIN_RANGE_M or r > msg.range_max:
                continue
            if r < OBSTACLE_DISTANCE_M:
                found = True
                break

        with self._lock:
            self.obstacle_ahead = found

    # ── Controller ─────────────────────────────────────────────────────────────

    def joy_callback(self, msg: Joy):
        self.last_joy = msg

        # O (btn 1) — abort to MANUAL
        if self._btn(msg, 1):
            with self._lock:
                self.mode = self.MODE_MANUAL
                self._mission_active = False
            self._stop()
            self._log("MANUAL / ABORT")
            return

        # X (btn 0) — start waypoint mission
        if self._btn(msg, 0):
            with self._lock:
                already = self._mission_active
                if not already:
                    self._mission_active = True
                    self.mode = self.MODE_WAYPOINT
            if not already:
                self._log("WAYPOINT mission starting...")
                threading.Thread(
                    target=self._drive_closest_waypoints,
                    daemon=True).start()
            else:
                self._log("Mission running — press O to abort first")

    def _btn(self, joy: Joy, idx: int) -> bool:
        return len(joy.buttons) > idx and joy.buttons[idx] == 1

    def control_loop(self):
        """Manual drive — L2 must be held. Only active in MANUAL mode."""
        if self.last_joy is None:
            return
        with self._lock:
            mode = self.mode
        if mode != self.MODE_MANUAL:
            return
        joy = self.last_joy
        if len(joy.axes) > 4 and joy.axes[4] < 0.5:
            t = Twist()
            t.linear.x  = float(joy.axes[3]) if len(joy.axes) > 3 else 0.0
            t.angular.z = float(joy.axes[0]) if len(joy.axes) > 0 else 0.0
            self.cmd_pub.publish(t)

    def _aborted(self) -> bool:
        with self._lock:
            return self.mode == self.MODE_MANUAL

    def _end_mission(self):
        with self._lock:
            self.mode = self.MODE_MANUAL
            self._mission_active = False
        self._log("Mission ended — MANUAL restored")

    # ── Motion primitives ──────────────────────────────────────────────────────

    def _turn_to_yaw(self, target_yaw: float) -> bool:
        """Rotate in place to target_yaw. Returns False on abort."""
        stable = 0
        while rclpy.ok():
            if self._aborted():
                self._stop()
                return False
            _, _, cyaw = self._pose()
            err = _norm(target_yaw - cyaw)
            if abs(err) < ANGLE_TOL_RAD:
                stable += 1
                if stable >= 5:
                    self._stop()
                    return True
            else:
                stable = 0
            speed = max(0.15, min(MAX_ANGULAR_SPEED, 1.8 * abs(err)))
            cmd = Twist()
            cmd.angular.z = math.copysign(speed, err)
            self.cmd_pub.publish(cmd)
            time.sleep(CONTROL_DT)
        return False

    def _drive_straight(self, distance_m: float, hold_yaw: float) -> bool:
        """
        Drive forward distance_m along hold_yaw.
        Arc-bypasses any obstacle, then continues remaining distance.
        Returns False on abort.
        """
        if distance_m <= 0.0:
            return True

        sx, sy, _ = self._pose()

        while rclpy.ok():
            if self._aborted():
                self._stop()
                return False

            cx, cy, cyaw = self._pose()
            progress  = ((cx - sx) * math.cos(hold_yaw) +
                         (cy - sy) * math.sin(hold_yaw))
            remaining = distance_m - max(0.0, progress)

            if remaining <= 0.05:
                self._stop()
                return True

            with self._lock:
                obstacle = self.obstacle_ahead

            if obstacle:
                self._stop()
                ox, oy, _ = self._pose()
                self._log(f"  Obstacle detected at ({ox:.2f},{oy:.2f}) — "
                          f"bypass ({remaining:.2f} m left) — taking photo")
                self._take_obstacle_photo()
                if not self._arc_bypass(hold_yaw):
                    return False
                # Resume only remaining portion
                sx, sy, _ = self._pose()
                distance_m = remaining
                if not self._turn_to_yaw(hold_yaw):
                    return False
                continue

            yaw_err = _norm(hold_yaw - cyaw)
            fwd = max(0.05, min(MAX_LINEAR_SPEED, DISTANCE_KP * remaining))
            ang = max(-MAX_ANGULAR_SPEED,
                      min(MAX_ANGULAR_SPEED, HEADING_KP * yaw_err))
            cmd = Twist()
            cmd.linear.x  = fwd
            cmd.angular.z = ang
            self.cmd_pub.publish(cmd)
            time.sleep(CONTROL_DT)

        return False

    def _drive_simple(self, distance_m: float, hold_yaw: float) -> bool:
        """
        Drive forward distance_m with heading hold. NO bypass logic.
        Used inside arc bypass to avoid infinite recursion.
        Hard safety stop at 0.35 m chassis clearance.
        """
        if distance_m <= 0.01:
            return True
        sx, sy, _ = self._pose()

        while rclpy.ok():
            if self._aborted():
                self._stop()
                return False

            cx, cy, cyaw = self._pose()
            if math.hypot(cx - sx, cy - sy) >= distance_m - 0.02:
                self._stop()
                return True

            # Hard chassis safety
            scan = self.latest_scan
            if scan is not None:
                for i, r in enumerate(scan.ranges):
                    a = scan.angle_min + i * scan.angle_increment
                    if abs(a) > math.radians(20):
                        continue
                    if math.isfinite(r) and CHASSIS_MIN_RANGE_M < r < 0.35:
                        self._stop()
                        time.sleep(0.15)
                        sx, sy, _ = self._pose()
                        break

            yaw_err = _norm(hold_yaw - cyaw)
            cmd = Twist()
            cmd.linear.x  = MAX_LINEAR_SPEED
            cmd.angular.z = max(-0.5, min(0.5, 2.0 * yaw_err))
            self.cmd_pub.publish(cmd)
            time.sleep(CONTROL_DT)

        return False

    def _arc_bypass(self, hold_yaw: float) -> bool:
        """
        U-shaped bypass:
          1. Choose left/right by LiDAR clearance
          2. Turn 90 deg to that side
          3. Drive BYPASS_SIDE_M sideways
          4. Turn back to hold_yaw
          5. Drive BYPASS_FWD_M forward (clear obstacle)
          6. Turn to opposite side
          7. Drive BYPASS_SIDE_M back to original ray line
          8. Turn to hold_yaw (restored)
        """
        side  = self._bypass_side(hold_yaw)
        label = "LEFT" if side > 0 else "RIGHT"
        self._log(f"  Arc bypass {label}")

        side_yaw = _norm(hold_yaw + side * math.pi / 2)
        ret_yaw  = _norm(hold_yaw - side * math.pi / 2)

        if not self._turn_to_yaw(side_yaw):             return False
        if not self._drive_simple(BYPASS_SIDE_M, side_yaw): return False
        if not self._turn_to_yaw(hold_yaw):             return False
        if not self._drive_simple(BYPASS_FWD_M,  hold_yaw): return False
        if not self._turn_to_yaw(ret_yaw):              return False
        if not self._drive_simple(BYPASS_SIDE_M, ret_yaw):  return False
        if not self._turn_to_yaw(hold_yaw):             return False

        self._log("    bypass complete")
        return True

    def _bypass_side(self, hold_yaw: float) -> float:
        """Returns +1.0 (left) or -1.0 (right) based on LiDAR clearance."""
        scan = self.latest_scan
        if scan is None:
            return 1.0
        with self._lock:
            ryaw = self.current_yaw
        left_min  = float('inf')
        right_min = float('inf')
        for i, r in enumerate(scan.ranges):
            if not math.isfinite(r) or r < CHASSIS_MIN_RANGE_M:
                continue
            a_rel = _norm((scan.angle_min + i * scan.angle_increment) -
                          _norm(ryaw - hold_yaw))
            if math.pi / 6 <= a_rel <= 5 * math.pi / 6:
                left_min  = min(left_min, r)
            elif -5 * math.pi / 6 <= a_rel <= -math.pi / 6:
                right_min = min(right_min, r)
        return 1.0  # always turn left first

    # ── Waypoint mission ───────────────────────────────────────────────────────

    def _drive_to_pose(self, tx: float, ty: float,
                       tol: float = WAYPOINT_TOLERANCE_M) -> bool:
        """
        Drive to (tx, ty) with arc-bypass avoidance.
        Drives in 1.5 m bursts — heading recalculated every burst.
        Returns True within tol metres, False on abort.
        """
        self._log(f"  -> ({tx:.2f}, {ty:.2f})")
        while rclpy.ok():
            if self._aborted():
                self._stop()
                return False

            # ── E-stop: sudden close obstacle while driving ──────────────────
            scan = self.latest_scan
            if scan is not None:
                estop_cone = math.radians(ESTOP_CONE_DEG)
                for i, r in enumerate(scan.ranges):
                    a = scan.angle_min + i * scan.angle_increment
                    if abs(a) > estop_cone:
                        continue
                    if not math.isfinite(r):
                        continue
                    if r < CHASSIS_MIN_RANGE_M:
                        continue
                    if r < ESTOP_DISTANCE_M:
                        self._trigger_estop(
                            f"object at {r:.2f} m while driving to "
                            f"({tx:.2f},{ty:.2f})")
                        return False
            # ────────────────────────────────────────────────────────────────

            cx, cy, cyaw = self._pose()
            dist = math.hypot(tx - cx, ty - cy)
            if dist < tol:
                self._stop()
                return True
            target_yaw = math.atan2(ty - cy, tx - cx)
            if abs(_norm(target_yaw - cyaw)) > 0.4:
                if not self._turn_to_yaw(target_yaw):
                    return False
                continue
            if not self._drive_straight(min(dist, 1.5), target_yaw):
                return False
        return False

    def _plan_visit_order(self, waypoints: List[Tuple[float, float]],
                          start: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Nearest-neighbour greedy path planning from start through all waypoints
        returning to origin. Returns ordered list of waypoints to visit.
        """
        remaining = list(waypoints)
        ordered   = []
        current   = start
        while remaining:
            idx = min(range(len(remaining)),
                      key=lambda i: math.hypot(remaining[i][0] - current[0],
                                               remaining[i][1] - current[1]))
            current = remaining.pop(idx)
            ordered.append(current)
        return ordered

    def _log_planned_route(self, ordered: List[Tuple[float, float]],
                           start: Tuple[float, float],
                           compute_s: float):
        """Log the planned route with distances and total estimated length."""
        self._log(f"Path planning complete ({compute_s*1000:.3f} ms) — "
                  f"algorithm: nearest-neighbour greedy")
        self._log(f"Planned route ({len(ordered)} waypoints):")
        self._log(f"  Origin ({start[0]:.2f}, {start[1]:.2f})")

        total = 0.0
        prev  = start
        for i, (wx, wy) in enumerate(ordered, start=1):
            d = math.hypot(wx - prev[0], wy - prev[1])
            total += d
            self._log(f"  WP{i} -> ({wx:.2f}, {wy:.2f})  leg={d:.2f} m  "
                      f"cumulative={total:.2f} m")
            prev = (wx, wy)

        ret_d = math.hypot(prev[0] - start[0], prev[1] - start[1])
        total += ret_d
        self._log(f"  Return -> ({start[0]:.2f}, {start[1]:.2f})  "
                  f"leg={ret_d:.2f} m  cumulative={total:.2f} m")
        self._log(f"Total estimated path length: {total:.2f} m")

    def _drive_closest_waypoints(self):
        """
        Load waypoints.txt, plan visit order (nearest-neighbour greedy),
        display planned route and compute time, record journey CSV,
        photo at each waypoint, return to (0, 0) when done.
        """
        waypoints = load_waypoints(WAYPOINTS_FILE)
        if not waypoints:
            self._log(f"No waypoints found at {WAYPOINTS_FILE}")
            self._end_mission()
            return

        self._log(f"Loaded {len(waypoints)} waypoints")

        # Wait up to 15 s for valid pose
        deadline = time.time() + 15.0
        while not self.pose_valid:
            if time.time() > deadline:
                self._log("No pose available — aborting")
                self._end_mission()
                return
            if self._aborted():
                self._end_mission()
                return
            time.sleep(0.1)

        self._log("Pose: EKF /odom_fused")

        # ── Path planning ──────────────────────────────────────────────────────
        cx0, cy0, _ = self._pose()
        start        = (cx0, cy0)

        t_plan_start = time.time()
        ordered      = self._plan_visit_order(waypoints, start)
        compute_s    = time.time() - t_plan_start

        self._log_planned_route(ordered, start, compute_s)

        # ── Journey log setup ──────────────────────────────────────────────────
        self._leg_recording = False
        self._leg_frames    = []
        mission_start_time  = time.time()

        # ── Drive planned route ────────────────────────────────────────────────
        for wp_num, (tx, ty) in enumerate(ordered, start=1):
            if self._aborted():
                self._stop_leg_recording()
                break

            cx, cy, _ = self._pose()
            leg_dist   = math.hypot(tx - cx, ty - cy)
            self._log(f"=== WP {wp_num}/{len(ordered)}: "
                      f"({tx:.2f}, {ty:.2f})  {leg_dist:.2f} m ===")

            # Start recording this leg
            leg_frames = self._start_leg_recording()
            leg_start  = time.time()

            if not self._drive_to_pose(tx, ty):
                self._stop_leg_recording()
                self._end_mission()
                return

            self._stop_leg_recording()
            leg_time = time.time() - leg_start
            ax, ay, _ = self._pose()
            self._log(f"  Arrived WP{wp_num} in {leg_time:.1f} s  "
                      f"actual=({ax:.2f},{ay:.2f})")

            # Save leg video (includes still photo at end)
            threading.Thread(
                target=self._save_leg_video,
                args=(wp_num, list(leg_frames)),
                daemon=True).start()

            self._stop()
            time.sleep(SETTLE_TIME_S)

        # ── Return to origin ───────────────────────────────────────────────────
        self._log("All waypoints done — returning to (0, 0)")

        ret_start = time.time()
        if self._drive_to_pose(0.0, 0.0, tol=RETURN_TOL_M):
            ret_time  = time.time() - ret_start
            fx, fy, _ = self._pose()
            self._log(f"Origin reached in {ret_time:.1f} s  "
                      f"final=({fx:.2f},{fy:.2f})")
        else:
            self._log("Return interrupted")

        total_time = time.time() - mission_start_time
        self._log(f"=== Mission complete — total time {total_time:.1f} s ===")
        self._end_mission()

    # ── Leg video recording ────────────────────────────────────────────────────

    def _start_leg_recording(self):
        """Start collecting frames for the current leg. Returns a shared list."""
        leg_frames = []
        self._leg_frames     = leg_frames
        self._leg_recording  = True
        threading.Thread(
            target=self._leg_record_thread,
            args=(leg_frames,),
            daemon=True).start()
        return leg_frames

    def _leg_record_thread(self, leg_frames: list):
        """Capture frames at ESTOP_FPS into leg_frames while _leg_recording."""
        interval = 1.0 / ESTOP_FPS
        while getattr(self, '_leg_recording', False):
            if self.camera_ready and self.q_rgb is not None:
                try:
                    pkt = self.q_rgb.tryGet()
                    if pkt is not None:
                        leg_frames.append(pkt.getCvFrame())
                except Exception:
                    pass
            time.sleep(interval)

    def _stop_leg_recording(self):
        """Signal the recording thread to stop."""
        self._leg_recording = False

    def _save_leg_video(self, wp_num: int, leg_frames: list):
        """
        Write leg_frames to journeyw{N}.mp4 in CAMERA_SAVE_DIR.
        Grabs a final still from the camera and appends it for SETTLE_TIME_S
        worth of frames so the destination photo is visible at the end.
        """
        if not self.camera_ready or not leg_frames:
            self._log(f"  Journey WP{wp_num}: no frames — video skipped")
            return

        fn = os.path.join(CAMERA_SAVE_DIR, f"journeyw{wp_num}.mp4")
        h, w = leg_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(fn, fourcc, float(ESTOP_FPS), (w, h))

        # Write driving footage
        for frame in leg_frames:
            writer.write(frame)

        # Grab arrival still and hold it for SETTLE_TIME_S at end of video
        still = None
        try:
            pkt = self.q_rgb.tryGet()
            if pkt is not None:
                still = pkt.getCvFrame()
                # Save still as separate JPEG too
                jpg = os.path.join(
                    CAMERA_SAVE_DIR, f"waypoint_{wp_num:03d}.jpg")
                cv2.imwrite(jpg, still)
                self._log(f"  Photo: {jpg}")
        except Exception as e:
            self._log(f"  Photo error: {e}")

        if still is not None:
            hold_frames = int(SETTLE_TIME_S * ESTOP_FPS)
            for _ in range(hold_frames):
                writer.write(still)

        writer.release()
        self._log(
            f"  Journey video: {fn}  "
            f"({len(leg_frames)} driving frames + still)")

    # ── Camera ─────────────────────────────────────────────────────────────────

    def _init_camera(self):
        if not DEPTHAI_AVAILABLE:
            return
        try:
            os.makedirs(CAMERA_SAVE_DIR, exist_ok=True)
            self.pipeline = dai.Pipeline()
            cam = self.pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A)
            out = cam.requestOutput(
                (1920, 1080), type=dai.ImgFrame.Type.BGR888p)
            self.q_rgb = out.createOutputQueue()
            self.pipeline.start()
            self.camera_ready = True
            self._log("Camera ready")
            # Start rolling buffer thread — always running
            threading.Thread(
                target=self._rolling_buffer_thread, daemon=True).start()
        except Exception as e:
            self.camera_ready = False
            self._log(f"Camera unavailable: {e}")

    def _rolling_buffer_thread(self):
        """
        Continuously reads frames at ESTOP_FPS and keeps the last
        ESTOP_BUFFER_SECS worth in self._frame_buffer (a deque).
        Runs forever as a daemon — silently discards oldest frames.
        """
        interval   = 1.0 / ESTOP_FPS
        max_frames = int(ESTOP_BUFFER_SECS * ESTOP_FPS)
        while rclpy.ok():
            if not self.camera_ready or self.q_rgb is None:
                time.sleep(0.1)
                continue
            try:
                pkt = self.q_rgb.tryGet()
                if pkt is not None:
                    frame = pkt.getCvFrame()
                    with self._estop_lock:
                        self._frame_buffer.append((time.time(), frame))
                        while len(self._frame_buffer) > max_frames:
                            self._frame_buffer.popleft()
            except Exception:
                pass
            time.sleep(interval)

    def _save_estop_footage(self, reason: str):
        """
        Save the last 5 seconds of frames from the rolling buffer
        as a single MP4 video directly into CAMERA_SAVE_DIR.
        """
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(CAMERA_SAVE_DIR, f"estop_{ts_str}.mp4")

        with self._estop_lock:
            frames = list(self._frame_buffer)

        if not frames:
            self._log("E-STOP: no frames in buffer — video not saved")
            return

        h, w = frames[0][1].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(fn, fourcc, float(ESTOP_FPS), (w, h))

        for _, frame in frames:
            writer.write(frame)

        writer.release()
        self._log(
            f"E-STOP footage saved: {fn} "
            f"({len(frames)} frames, ~{len(frames)/ESTOP_FPS:.1f}s) "
            f"| reason: {reason}")

    def _trigger_estop(self, reason: str):
        """Stop robot and save last 5 s of footage in background."""
        self._stop()
        with self._lock:
            self.mode = self.MODE_MANUAL
            self._mission_active = False
        self._log(f"E-STOP triggered: {reason}")
        threading.Thread(
            target=self._save_estop_footage,
            args=(reason,),
            daemon=True).start()

    def _take_obstacle_photo(self):
        """Capture and save a photo of the detected obstacle immediately."""
        if not self.camera_ready or self.q_rgb is None:
            return
        try:
            pkt = self.q_rgb.tryGet()
            if pkt is not None:
                self.photo_count += 1
                fn = os.path.join(
                    CAMERA_SAVE_DIR,
                    f"obstacle_{self.photo_count:03d}.jpg")
                cv2.imwrite(fn, pkt.getCvFrame())
                self._log(f"  Obstacle photo: {fn}")
        except Exception as e:
            self._log(f"  Obstacle photo error: {e}")

    def _capture_photo(self):
        with self._camera_lock:
            if not self.camera_ready:
                return
            try:
                pkt = self.q_rgb.tryGet()
                if pkt is not None:
                    self.photo_count += 1
                    fn = os.path.join(
                        CAMERA_SAVE_DIR,
                        f"waypoint_{self.photo_count:03d}.jpg")
                    cv2.imwrite(fn, pkt.getCvFrame())
                    self._log(f"Photo: {fn}")
            except Exception as e:
                self._log(f"Photo error: {e}")

    def _trigger_photo(self):
        threading.Thread(target=self._capture_photo, daemon=True).start()

    # ── Utility ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.get_logger().info(msg)
        s = String()
        s.data = msg
        self.status_pub.publish(s)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def destroy_node(self):
        self._stop()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        super().destroy_node()


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DualShockModeTeleop()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()