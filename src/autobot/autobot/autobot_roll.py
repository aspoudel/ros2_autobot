#!/usr/bin/env python3
"""
autobot_roll.py — Rotate-and-Go mapper for Pioneer (15 × 15 m grid)
=====================================================================

Behaviour
---------
AUTO mode (press X):
  1. HOME = current pose (centre of a 15 × 15 m field).
  2. Slowly rotate in place at HOME.
  3. A NARROW forward LiDAR cone (±DETECT_CONE_DEG/2) watches for objects.
  4. When something unvisited appears INSIDE the field, straight ahead:
       a. Stop rotating.
       b. Drive straight forward (heading hold) until STANDOFF_M.
       c. Take a photo with the OAK-D, classify Greek letter (YOLO),
          detect orange/red colour blobs.
       d. Publish a bright sphere + text marker, drop a hit-ping marker.
       e. Reverse straight back to HOME along the same heading.
  5. Resume slow rotation. Yaw is accumulated ONLY while spinning at home.
  6. Done when 360° accumulated. Save SLAM map.

MANUAL mode  : hold L2 + sticks.
Circle button: abort AUTO → MANUAL instantly (next X = fresh mission).
X after emergency: resume AUTO, keeping visited/trail/markers.

Foxglove
--------
3D panel  (display frame = map): /tf, /map, /scan, /obstacle_markers
Image panel: /oak/rgb/image_raw

Markers under /obstacle_markers:
  home, trail, scan_cone (live), obstacles, labels,
  claimed_perimeter, hits, emergency.

Safety
------
While in AUTO, anything that intrudes within EMERGENCY_STOP_M of the robot
in a forward cone of EMERGENCY_CONE_DEG triggers an immediate stop and a
drop to MANUAL.  A bright red emergency marker is published at the
intrusion point.  Press X again to resume AUTO (mapping data is preserved
across an emergency abort; Circle wipes it for a fresh mission).
"""

import csv
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclpyDuration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Joy, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:
    from slam_toolbox.srv import SaveMap
    SLAM_TOOLBOX_AVAILABLE = True
except ImportError:
    SLAM_TOOLBOX_AVAILABLE = False

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# ──────────────────────────── Tunables ──────────────────────────────────────

AREA_SIZE      = 15.0
FIELD_HALF     = AREA_SIZE / 2.0
FIELD_BUFFER   = 0.50

# Detection — narrow forward cone, "right in front"
DETECT_CONE_DEG     = 20.0                       # total cone width
DETECT_MIN_RANGE_M  = 0.30                       # ignore chassis noise
DETECT_MAX_RANGE_M  = FIELD_HALF * math.sqrt(2)  # field diagonal
CLUSTER_GAP_M       = 0.25                       # range jump that splits a cluster
CLUSTER_MIN_BEAMS   = 3
VISITED_RADIUS_M    = 1.00                       # never visit the same object twice

# Approach geometry
STANDOFF_M             = 0.60   # stop this far from object face
APPROACH_SPEED         = 0.30   # m/s forward/reverse during a visit
SAFETY_STOP_M          = 0.25   # hard chassis stop during forward drive
SAFETY_CONE_DEG        = 30.0   # cone for the chassis safety check

# Emergency intrusion — sudden obstacle within this distance in heading cone
# while AUTO is armed → drop to MANUAL.  Press X to resume AUTO.
EMERGENCY_STOP_M       = 1.00
EMERGENCY_CONE_DEG     = 60.0

# Spinning
SCAN_TURN_SPEED        = 0.35   # rad/s slow rotation while scanning
TURN_SPEED_MAX         = 0.80   # rad/s max for point turns
ANGLE_TOL_RAD          = math.radians(2.0)
CONTROL_DT             = 0.03

# Camera (single-shot capture per visit)
CAPTURE_SETTLE_S       = 0.35
SWEEP_CONF_THR         = 0.50

# Vision tunables
VISION_USE_WHITE_CROP  = True
VISION_MIN_INK_RATIO   = 0.005
VISION_MAX_INK_RATIO   = 0.35
VISION_MIN_INK_AREA    = 80
VISION_WHITE_MIN_AREA  = 5000
COLOUR_MIN_AREA_PX     = 2500
COLOUR_MIN_RATIO       = 0.015

# Manual
MANUAL_LINEAR_SPEED    = 0.50
MANUAL_ANGULAR_SPEED   = 1.20
STARTUP_DELAY_S        = 3.0

# Paths
_HERE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_HERE, "best.pt")
FOR_ML_DIR   = "/root/ros2_autobot/for_ml"
BIN_CSV_PATH = "/root/ros2_autobot/for_ml/bin_positions.csv"
MAP_DIR      = "/root/ros2_autobot/maps"

# Markers
MARKER_FRAME_PREFERRED = "map"
MARKER_FRAME_FALLBACK  = "odom"
MARKER_NS_OBJ          = "obstacles"
MARKER_NS_LABEL        = "labels"
MARKER_NS_HOME         = "home"
MARKER_NS_TRAIL        = "trail"
MARKER_NS_CONE         = "scan_cone"
MARKER_NS_CLAIMED      = "claimed_perimeter"
MARKER_NS_HIT          = "hits"
MARKER_NS_EMERGENCY    = "emergency"
MARKER_SPHERE_SCALE    = 0.32
MARKER_TEXT_SCALE      = 0.42
MARKER_REBROADCAST_HZ  = 1.0
CONE_TICK_HZ           = 8.0

TRAIL_DROP_DIST_M      = 0.20
TRAIL_LINE_WIDTH       = 0.06
TRAIL_TICK_HZ          = 5.0

PALETTE: List[Tuple[float, float, float]] = [
    (0.95, 0.20, 0.20), (0.20, 0.55, 0.95), (0.30, 0.85, 0.30),
    (0.95, 0.78, 0.20), (0.66, 0.36, 0.95), (0.98, 0.55, 0.18),
    (0.20, 0.85, 0.85), (0.98, 0.45, 0.78), (0.60, 0.95, 0.20),
    (0.95, 0.30, 0.55),
]
GREEK_COLOURS = {
    "alpha":   (0.95, 0.20, 0.20), "beta":    (0.20, 0.55, 0.95),
    "gamma":   (0.30, 0.85, 0.30), "delta":   (0.95, 0.78, 0.20),
    "epsilon": (0.66, 0.36, 0.95), "zeta":    (0.98, 0.55, 0.18),
    "eta":     (0.20, 0.85, 0.85), "theta":   (0.98, 0.45, 0.78),
    "iota":    (0.60, 0.95, 0.20), "kappa":   (0.95, 0.30, 0.55),
    "lambda":  (0.95, 0.65, 0.30), "mu":      (0.40, 0.75, 0.40),
}
UNCONFIRMED_LABELS = {
    "", "none", "vision_unavailable", "no_white_page",
    "no_ink", "low_confidence", "not_letter", "no_detection",
}


@dataclass
class Detection:
    world_x:     float
    world_y:     float
    field_x:     float
    field_y:     float
    range_m:     float
    bearing_rad: float   # in the robot body frame


class AutobotRotateAndGo(Node):

    MODE_MANUAL   = "manual"
    MODE_AUTO     = "auto"
    PHASE_IDLE    = "IDLE"
    PHASE_WAITING = "WAITING"
    PHASE_RUNNING = "RUNNING"
    PHASE_DONE    = "DONE"

    # ══════════════════════════════════════════════════════════════════════
    # Setup
    # ══════════════════════════════════════════════════════════════════════

    def __init__(self) -> None:
        super().__init__("autobot_rotate_and_go")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("btn_x",           0)
        self.declare_parameter("btn_circle",      1)
        self.declare_parameter("axis_left_x",     0)
        self.declare_parameter("axis_right_y",    3)
        self.declare_parameter("axis_l2",         4)
        self.declare_parameter("deadzone",        0.10)
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("odom_topic",      "/odom_fused")
        self.declare_parameter("lidar_topic",     "/scan")

        self.btn_x           = int(self.get_parameter("btn_x").value)
        self.btn_circle      = int(self.get_parameter("btn_circle").value)
        self.axis_left_x     = int(self.get_parameter("axis_left_x").value)
        self.axis_right_y    = int(self.get_parameter("axis_right_y").value)
        self.axis_l2         = int(self.get_parameter("axis_l2").value)
        self.deadzone        = float(self.get_parameter("deadzone").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        odom_topic           = str(self.get_parameter("odom_topic").value)
        lidar_topic          = str(self.get_parameter("lidar_topic").value)

        self._lock        = threading.Lock()
        self._camera_lock = threading.Lock()
        self._marker_lock = threading.Lock()

        self.mode                = self.MODE_MANUAL
        self.auto_phase          = self.PHASE_IDLE
        self.auto_start_time     = 0.0
        self.auto_thread_started = False
        self.last_joy: Optional[Joy] = None

        self.odom_ready  = False
        self.odom_x = 0.0; self.odom_y = 0.0; self.odom_yaw = 0.0
        self.home_x = 0.0; self.home_y = 0.0; self.home_yaw = 0.0

        self.lidar_ready     = False
        self.latest_scan: Optional[LaserScan] = None
        self.safety_obstacle = False

        # Emergency intrusion — armed during AUTO scan/rotation, disarmed
        # during a visit's forward-approach so the target itself doesn't fire it.
        self._emergency_armed      = False
        self._was_emergency_abort  = False

        self.camera_ready = False
        self.pipeline = None
        self.q_rgb = None
        self.bridge = CvBridge()

        self.visited_positions: List[Tuple[float, float]] = []
        self.waypoint_counts: dict = {}
        self.photo_count = 0
        self.event_count = 0

        self._markers: List[Marker] = []
        self._trail_points: List[Tuple[float, float, str]] = []
        self._last_trail_xy: Optional[Tuple[float, float]] = None

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",           10)
        self.mode_pub   = self.create_publisher(String, "/control_mode",      10)
        self.status_pub = self.create_publisher(String, "/robot_status",      10)
        self.image_pub  = self.create_publisher(Image,  "/oak/rgb/image_raw", 10)

        _mq = QoSProfile(depth=200)
        _mq.durability  = DurabilityPolicy.TRANSIENT_LOCAL
        _mq.reliability = ReliabilityPolicy.RELIABLE
        self.marker_pub = self.create_publisher(MarkerArray, "/obstacle_markers", _mq)

        self.yolo_model = None
        self._init_vision_model()

        self.create_subscription(Joy,       "/joy",      self.joy_callback,   10, callback_group=self.cb_group)
        self.create_subscription(Odometry,  odom_topic,  self.odom_callback,  20, callback_group=self.cb_group)
        self.create_subscription(LaserScan, lidar_topic, self.lidar_callback, 20, callback_group=self.cb_group)

        self.save_map_client = None
        if SLAM_TOOLBOX_AVAILABLE:
            os.makedirs(MAP_DIR, exist_ok=True)
            self.save_map_client = self.create_client(SaveMap, "/slam_toolbox/save_map")

        self.create_timer(1.0 / self.publish_rate_hz, self.control_loop,
                          callback_group=self.cb_group)
        self.create_timer(1.0 / MARKER_REBROADCAST_HZ, self._republish_markers_tick,
                          callback_group=self.cb_group)
        self.create_timer(1.0 / TRAIL_TICK_HZ, self._trail_tick,
                          callback_group=self.cb_group)
        self.create_timer(1.0 / CONE_TICK_HZ, self._cone_tick,
                          callback_group=self.cb_group)
        self.create_timer(0.20, self._camera_preview_tick,
                          callback_group=self.cb_group)

        self._init_dirs_csv()
        self._init_camera()

        self._log(
            f"Rotate-and-Go ready | field={AREA_SIZE}×{AREA_SIZE} m | "
            f"cone=±{DETECT_CONE_DEG/2:.0f}° | "
            f"range≤{DETECT_MAX_RANGE_M:.1f} m | "
            f"standoff={STANDOFF_M:.2f} m | "
            f"spin={SCAN_TURN_SPEED:.2f} rad/s | "
            f"emergency={EMERGENCY_STOP_M:.1f} m ±{EMERGENCY_CONE_DEG/2:.0f}° | "
            f"YOLO={'OK' if self.yolo_model else 'MISSING'} | "
            f"L2=manual  X=auto  Circle=abort"
        )

    def _init_dirs_csv(self) -> None:
        os.makedirs(FOR_ML_DIR, exist_ok=True)
        if not os.path.exists(BIN_CSV_PATH):
            with open(BIN_CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerow([
                    "event_id", "timestamp", "record_type", "display_label",
                    "world_x", "world_y", "field_x", "field_y",
                    "range_m", "heading_deg",
                    "best_photo",
                    "greek_letter", "letter_confidence",
                    "colour_class", "vision_detail",
                ])

    def _init_camera(self) -> None:
        if not DEPTHAI_AVAILABLE:
            self._log("WARNING: depthai not installed — camera disabled"); return
        try:
            self.pipeline = dai.Pipeline()
            cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            out = cam.requestOutput((1920, 1080), type=dai.ImgFrame.Type.BGR888p)
            self.q_rgb = out.createOutputQueue()
            self.pipeline.start()
            self.camera_ready = True
            self._log(f"OAK-D ready → {FOR_ML_DIR}")
        except Exception as e:
            self._log(f"WARNING: OAK-D init failed: {e}")

    def _init_vision_model(self) -> None:
        if not ULTRALYTICS_AVAILABLE:
            self._log("WARNING: ultralytics not installed"); return
        if not os.path.exists(MODEL_PATH):
            self._log(f"WARNING: best.pt not found at {MODEL_PATH}"); return
        try:
            self.yolo_model = YOLO(MODEL_PATH)
            self._log(f"YOLO loaded: {MODEL_PATH}")
        except Exception as e:
            self._log(f"WARNING: YOLO load failed: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════

    def _log(self, msg: str) -> None:
        self.get_logger().info(msg)
        s = String(); s.data = msg
        self.status_pub.publish(s)

    def _send_cmd(self, lin: float = 0.0, ang: float = 0.0) -> None:
        t = Twist(); t.linear.x = float(lin); t.angular.z = float(ang)
        self.cmd_pub.publish(t)

    def _stop(self) -> None:
        self._send_cmd()

    @staticmethod
    def _norm_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _apply_deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else v

    def _axis_pressed(self, joy: Joy, idx: int, thr: float = 0.5) -> bool:
        return len(joy.axes) > idx and joy.axes[idx] < thr

    def _btn_down(self, joy: Joy, idx: int) -> bool:
        return len(joy.buttons) > idx and joy.buttons[idx] == 1

    def _abort_requested(self) -> bool:
        with self._lock:
            return self.mode == self.MODE_MANUAL

    def _now(self):
        return self.get_clock().now().to_msg()

    # ══════════════════════════════════════════════════════════════════════
    # Pose / field frame
    # ══════════════════════════════════════════════════════════════════════

    def odom_callback(self, msg: Odometry) -> None:
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        if not self.odom_ready:
            self.odom_ready = True
            self._log(f"Odom ready ({self.odom_x:.2f},{self.odom_y:.2f}) "
                      f"yaw={math.degrees(self.odom_yaw):.1f}°")

    def _set_mission_home(self) -> None:
        self.home_x   = self.odom_x
        self.home_y   = self.odom_y
        self.home_yaw = self.odom_yaw
        self._log(f"Home: ({self.home_x:.2f},{self.home_y:.2f}) "
                  f"yaw={math.degrees(self.home_yaw):.1f}°")
        self._publish_home_marker()

    def _world_to_field(self, wx: float, wy: float) -> Tuple[float, float]:
        dx = wx - self.home_x; dy = wy - self.home_y
        c = math.cos(self.home_yaw); s = math.sin(self.home_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _inside_field(self, fx: float, fy: float, buffer: float = 0.0) -> bool:
        lim = FIELD_HALF - buffer
        return abs(fx) <= lim and abs(fy) <= lim

    def _inside_field_world(self, wx: float, wy: float, buffer: float = 0.0) -> bool:
        return self._inside_field(*self._world_to_field(wx, wy), buffer)

    # ══════════════════════════════════════════════════════════════════════
    # LiDAR
    # ══════════════════════════════════════════════════════════════════════

    def lidar_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        if not self.lidar_ready:
            self.lidar_ready = True
            self._log(f"LiDAR ready: {len(msg.ranges)} rays "
                      f"[{msg.range_min:.2f},{msg.range_max:.2f}] m")

        # Chassis safety brake + emergency intrusion check.
        # Angles are normalised so Lakibeam's [0..2π] scan still maps
        # "forward" to a≈0 instead of wrapping to 2π.
        safety_half    = math.radians(SAFETY_CONE_DEG    / 2.0)
        emergency_half = math.radians(EMERGENCY_CONE_DEG / 2.0)
        hit            = False
        emergency: Optional[Tuple[float, float]] = None  # (range, bearing)
        closest_r      = math.inf
        closest_b      = 0.0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):       continue
            if r < DETECT_MIN_RANGE_M:     continue
            a = self._norm_angle(msg.angle_min + i * msg.angle_increment)
            abs_a = abs(a)
            if abs_a <= safety_half and r < SAFETY_STOP_M:
                hit = True
            if abs_a <= emergency_half and r < EMERGENCY_STOP_M and r < closest_r:
                closest_r = r
                closest_b = a

        self.safety_obstacle = hit
        if not math.isinf(closest_r):
            emergency = (closest_r, closest_b)

        # Fire emergency only when armed and we are still in AUTO.
        with self._lock:
            armed = self._emergency_armed and self.mode == self.MODE_AUTO
        if armed and emergency is not None:
            self._trigger_emergency_abort(emergency[0], emergency[1])

    def _front_detection(self) -> Optional[Detection]:
        """
        Look only at the narrow forward cone (±DETECT_CONE_DEG/2).
        Cluster contiguous beams with similar range; take the cluster whose
        nearest beam is closest to the robot.  Return a Detection in the
        world frame, or None.
        """
        scan = self.latest_scan
        if scan is None:
            return None

        half = math.radians(DETECT_CONE_DEG / 2.0)
        beams: List[Tuple[float, float]] = []  # (angle, range)
        for i, r in enumerate(scan.ranges):
            a = self._norm_angle(scan.angle_min + i * scan.angle_increment)
            if abs(a) > half:                continue
            if not math.isfinite(r):         continue
            if r < DETECT_MIN_RANGE_M:       continue
            if r > DETECT_MAX_RANGE_M:       continue
            if r > scan.range_max:           continue
            beams.append((a, r))
        if not beams:
            return None

        beams.sort(key=lambda x: x[0])
        clusters: List[List[Tuple[float, float]]] = [[beams[0]]]
        for b in beams[1:]:
            if abs(b[1] - clusters[-1][-1][1]) <= CLUSTER_GAP_M:
                clusters[-1].append(b)
            else:
                clusters.append([b])

        best_cluster = None
        best_r = float("inf")
        for cl in clusters:
            if len(cl) < CLUSTER_MIN_BEAMS: continue
            r_min = min(r for _, r in cl)
            if r_min < best_r:
                best_r = r_min; best_cluster = cl
        if best_cluster is None:
            return None

        bearing = sum(a for a, _ in best_cluster) / len(best_cluster)
        rng     = best_r
        wb      = self._norm_angle(self.odom_yaw + bearing)
        wx      = self.odom_x + rng * math.cos(wb)
        wy      = self.odom_y + rng * math.sin(wb)
        fx, fy  = self._world_to_field(wx, wy)

        if not self._inside_field(fx, fy):
            return None
        if self._already_visited(wx, wy):
            return None

        return Detection(wx, wy, fx, fy, rng, bearing)

    def _already_visited(self, wx: float, wy: float) -> bool:
        return any(
            math.hypot(wx - px, wy - py) <= VISITED_RADIUS_M
            for px, py in self.visited_positions
        )

    # ── Emergency abort (1 m intrusion in heading cone) ───────────────────
    def _trigger_emergency_abort(self, range_m: float, bearing_rad: float) -> None:
        with self._lock:
            if self.mode == self.MODE_MANUAL:
                return
            self.mode                 = self.MODE_MANUAL
            self.auto_phase           = self.PHASE_IDLE
            self.auto_thread_started  = False
            self._emergency_armed     = False
            self._was_emergency_abort = True

        for _ in range(3):
            self._stop()

        m = String(); m.data = self.MODE_MANUAL
        self.mode_pub.publish(m)

        wb = self._norm_angle(self.odom_yaw + bearing_rad)
        wx = self.odom_x + range_m * math.cos(wb)
        wy = self.odom_y + range_m * math.sin(wb)
        self._publish_emergency_marker(wx, wy)

        self._log(f"⚠ EMERGENCY ABORT | obstacle at {range_m:.2f} m, "
                  f"bearing={math.degrees(bearing_rad):+.1f}° → MANUAL "
                  f"(press X to resume AUTO)")

    def _publish_emergency_marker(self, wx: float, wy: float) -> None:
        mx, my, frame = self._odom_to_map(wx, wy)
        m = Marker()
        m.header.frame_id = frame; m.header.stamp = self._now()
        m.ns = MARKER_NS_EMERGENCY
        m.id = int(time.time() * 10) & 0xFFFF
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = mx; m.pose.position.y = my; m.pose.position.z = 0.30
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.55
        m.color.r = 1.0; m.color.g = 0.05; m.color.b = 0.05; m.color.a = 0.95
        m.lifetime = Duration(sec=8, nanosec=0)
        with self._marker_lock: self._markers.append(m)
        self._publish_markers()

    # ══════════════════════════════════════════════════════════════════════
    # Markers — all bright, no greys
    # ══════════════════════════════════════════════════════════════════════

    def _odom_to_map(self, ox: float, oy: float) -> Tuple[float, float, str]:
        try:
            tr = self.tf_buffer.lookup_transform(
                "map", "odom", rclpy.time.Time(),
                timeout=RclpyDuration(seconds=0.2),
            )
            tx = tr.transform.translation.x
            ty = tr.transform.translation.y
            q  = tr.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y),
                             1 - 2*(q.y*q.y + q.z*q.z))
            c, s = math.cos(yaw), math.sin(yaw)
            return tx + c*ox - s*oy, ty + s*ox + c*oy, MARKER_FRAME_PREFERRED
        except Exception:
            return ox, oy, MARKER_FRAME_FALLBACK

    def _colour_for(self, event_id: int, label: str) -> Tuple[float, float, float]:
        key = (label or "").strip().lower()
        if key in GREEK_COLOURS:
            return GREEK_COLOURS[key]
        return PALETTE[event_id % len(PALETTE)]

    def _publish_markers(self) -> None:
        with self._marker_lock:
            ma = MarkerArray(); ma.markers = list(self._markers)
        self.marker_pub.publish(ma)

    def _republish_markers_tick(self) -> None:
        with self._marker_lock:
            if not self._markers: return
        self._publish_markers()

    def _publish_home_marker(self) -> None:
        mx, my, frame = self._odom_to_map(self.home_x, self.home_y)
        m = Marker()
        m.header.frame_id = frame; m.header.stamp = self._now()
        m.ns = MARKER_NS_HOME; m.id = 0
        m.type = Marker.CYLINDER; m.action = Marker.ADD
        m.pose.position.x = mx; m.pose.position.y = my; m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = 0.45; m.scale.y = 0.45; m.scale.z = 0.05
        m.color.r = 0.0; m.color.g = 0.55; m.color.b = 1.0; m.color.a = 1.0
        m.lifetime = Duration(sec=0, nanosec=0)
        with self._marker_lock: self._markers.append(m)
        self._publish_markers()

    def _publish_obstacle_markers(self, event_id: int, det: Detection,
                                  label: str, conf: float) -> None:
        mx, my, frame = self._odom_to_map(det.world_x, det.world_y)
        now = self._now(); lt = Duration(sec=0, nanosec=0)
        confirmed = (conf >= SWEEP_CONF_THR and label not in UNCONFIRMED_LABELS)
        r, g, b = self._colour_for(event_id, label if confirmed else "")

        sphere = Marker()
        sphere.header.frame_id = frame; sphere.header.stamp = now
        sphere.ns = MARKER_NS_OBJ; sphere.id = event_id
        sphere.type = Marker.SPHERE; sphere.action = Marker.ADD
        sphere.pose.position.x = mx; sphere.pose.position.y = my
        sphere.pose.position.z = 0.20; sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = MARKER_SPHERE_SCALE
        sphere.color.r, sphere.color.g, sphere.color.b = r, g, b
        sphere.color.a = 1.0
        sphere.lifetime = lt

        text = Marker()
        text.header.frame_id = frame; text.header.stamp = now
        text.ns = MARKER_NS_LABEL; text.id = event_id
        text.type = Marker.TEXT_VIEW_FACING; text.action = Marker.ADD
        text.pose.position.x = mx; text.pose.position.y = my
        text.pose.position.z = 0.70; text.pose.orientation.w = 1.0
        text.scale.z = MARKER_TEXT_SCALE
        text.color.r, text.color.g, text.color.b = r, g, b
        text.color.a = 1.0
        text.text = (f"{label.capitalize()}\n{conf:.0%}" if confirmed
                     else f"#{event_id}\nunknown")
        text.lifetime = lt

        with self._marker_lock:
            self._markers.extend([sphere, text])
        self._publish_markers()

    def _publish_claimed_perimeter(self, wx: float, wy: float, idx: int) -> None:
        mx, my, frame = self._odom_to_map(wx, wy)
        m = Marker()
        m.header.frame_id = frame; m.header.stamp = self._now()
        m.ns = MARKER_NS_CLAIMED; m.id = idx
        m.type = Marker.CYLINDER; m.action = Marker.ADD
        m.pose.position.x = mx; m.pose.position.y = my; m.pose.position.z = 0.005
        m.pose.orientation.w = 1.0
        m.scale.x = 2.0 * VISITED_RADIUS_M
        m.scale.y = 2.0 * VISITED_RADIUS_M
        m.scale.z = 0.01
        r, g, b = PALETTE[idx % len(PALETTE)]
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.18
        m.lifetime = Duration(sec=0, nanosec=0)
        with self._marker_lock: self._markers.append(m)
        self._publish_markers()

    def _publish_hit_marker(self, event_id: int,
                            wx: float, wy: float,
                            colour: Tuple[float, float, float]) -> None:
        mx, my, frame = self._odom_to_map(wx, wy)
        m = Marker()
        m.header.frame_id = frame; m.header.stamp = self._now()
        m.ns = MARKER_NS_HIT; m.id = event_id
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position.x = mx; m.pose.position.y = my; m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.55
        m.color.r, m.color.g, m.color.b = colour
        m.color.a = 0.55
        m.lifetime = Duration(sec=2, nanosec=0)
        with self._marker_lock: self._markers.append(m)
        self._publish_markers()

    # ── Trail (breadcrumbs in AUTO) ────────────────────────────────────────
    def _trail_tick(self) -> None:
        with self._lock:
            running = (self.auto_phase == self.PHASE_RUNNING)
        if not running or not self.odom_ready:
            return
        if (self._last_trail_xy is None or
            math.hypot(self.odom_x - self._last_trail_xy[0],
                       self.odom_y - self._last_trail_xy[1]) >= TRAIL_DROP_DIST_M):
            self._last_trail_xy = (self.odom_x, self.odom_y)
            mx, my, frame = self._odom_to_map(self.odom_x, self.odom_y)
            self._trail_points.append((mx, my, frame))
            self._publish_trail_marker()

    def _publish_trail_marker(self) -> None:
        if not self._trail_points: return
        frame = self._trail_points[-1][2]
        m = Marker()
        m.header.frame_id = frame; m.header.stamp = self._now()
        m.ns = MARKER_NS_TRAIL; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = TRAIL_LINE_WIDTH
        m.color.r = 0.10; m.color.g = 1.0; m.color.b = 0.20; m.color.a = 0.95
        m.lifetime = Duration(sec=0, nanosec=0)
        for x, y, _ in self._trail_points:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.04
            m.points.append(p)
        with self._marker_lock:
            self._markers = [mk for mk in self._markers
                             if not (mk.ns == MARKER_NS_TRAIL and mk.id == 0)]
            self._markers.append(m)
        self._publish_markers()

    # ── Live forward scan cone ─────────────────────────────────────────────
    def _cone_tick(self) -> None:
        if not self.odom_ready:
            return
        with self._lock:
            running = (self.auto_phase == self.PHASE_RUNNING)
        if not running:
            return

        half = math.radians(DETECT_CONE_DEG / 2.0)
        r    = min(DETECT_MAX_RANGE_M, 3.5)

        pts_body = [
            (0.0, 0.0),
            (r * math.cos(+half), r * math.sin(+half)),
            (r * math.cos(-half), r * math.sin(-half)),
        ]
        c, s = math.cos(self.odom_yaw), math.sin(self.odom_yaw)
        pts_world = [(self.odom_x + c*x - s*y, self.odom_y + s*x + c*y)
                     for x, y in pts_body]
        pts_map   = [self._odom_to_map(x, y) for x, y in pts_world]
        frame = pts_map[0][2]

        m = Marker()
        m.header.frame_id = frame; m.header.stamp = self._now()
        m.ns = MARKER_NS_CONE; m.id = 0
        m.type = Marker.TRIANGLE_LIST; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color.r = 1.0; m.color.g = 0.85; m.color.b = 0.20; m.color.a = 0.25
        m.lifetime = Duration(sec=0, nanosec=int(0.5 * 1e9))
        for mx, my, _ in pts_map:
            p = Point(); p.x = float(mx); p.y = float(my); p.z = 0.05
            m.points.append(p)
        with self._marker_lock:
            self._markers = [mk for mk in self._markers
                             if not (mk.ns == MARKER_NS_CONE and mk.id == 0)]
            self._markers.append(m)
        self._publish_markers()

    # ══════════════════════════════════════════════════════════════════════
    # Camera + Vision
    # ══════════════════════════════════════════════════════════════════════

    def _camera_preview_tick(self) -> None:
        if not self.camera_ready or self.q_rgb is None: return
        if not self._camera_lock.acquire(blocking=False): return
        try:
            pkt = self.q_rgb.tryGet()
            if pkt is None: return
            frame = pkt.getCvFrame()
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp    = self._now()
            msg.header.frame_id = "oak_camera"
            self.image_pub.publish(msg)
        except Exception:
            pass
        finally:
            self._camera_lock.release()

    def _grab_frame(self) -> Optional[np.ndarray]:
        if not self.camera_ready or self.q_rgb is None: return None
        deadline = time.time() + 1.5
        while time.time() < deadline:
            try:
                pkt = self.q_rgb.tryGet()
                if pkt: return pkt.getCvFrame()
            except Exception: pass
            time.sleep(0.05)
        return None

    def _save_and_publish_frame(self, frame: np.ndarray, name: str) -> str:
        path = os.path.join(FOR_ML_DIR, name)
        try:
            if cv2.imwrite(path, frame):
                self._log(f"  Photo → {path}")
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self._now()
                self.image_pub.publish(msg)
                self.photo_count += 1
                return name
        except Exception as e:
            self._log(f"  Photo save error: {e}")
        return ""

    def _detect_red_orange(self, frame) -> Tuple[str, float, str]:
        if frame is None: return "none", 0.0, "no frame"
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red1   = cv2.inRange(hsv, (0,   80, 80), (10,  255, 255))
        red2   = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        red    = cv2.morphologyEx(
            cv2.morphologyEx(cv2.bitwise_or(red1, red2), cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))),
            cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        orange = cv2.inRange(hsv, (11,  80, 80), (22,  255, 255))
        yellow = cv2.inRange(hsv, (23,  80, 80), (38,  255, 255))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        orange = cv2.morphologyEx(cv2.morphologyEx(orange, cv2.MORPH_OPEN, k),
                                    cv2.MORPH_CLOSE, k)
        yellow = cv2.morphologyEx(cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k),
                                  cv2.MORPH_CLOSE, k)
        tot = frame.shape[0] * frame.shape[1]
        def sc(m):
            px = cv2.countNonZero(m); rt = px / tot if tot > 0 else 0.0
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return rt, max((cv2.contourArea(c) for c in cs), default=0.0)
        rr, ra = sc(red); o_r, oa = sc(orange); yr, ya = sc(yellow)
        winner = max(("red", rr, ra), ("orange", o_r, oa), ("yellow", yr, ya),
                     key=lambda t: t[2])
        name, ratio, area = winner
        if area >= COLOUR_MIN_AREA_PX and ratio >= COLOUR_MIN_RATIO:
            return name, ratio, f"{name} ratio={ratio:.3f}"
        return "none", max(rr, o_r, yr), f"none (r={rr:.3f} o={o_r:.3f} y={yr:.3f})"

    def _crop_white_page(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        msk = cv2.morphologyEx(cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k),
                               cv2.MORPH_OPEN, k)
        cs, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs: return None, None, msk
        lg = max(cs, key=cv2.contourArea)
        if cv2.contourArea(lg) < VISION_WHITE_MIN_AREA: return None, None, msk
        x, y, w, h = cv2.boundingRect(lg); m = 20
        return (frame[max(0,y-m):min(frame.shape[0],y+h+m),
                      max(0,x-m):min(frame.shape[1],x+w+m)],
                (x, y, w, h), msk)

    def _has_black_ink(self, crop):
        if crop is None or crop.size == 0: return False, 0.0, None
        g   = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, ink = cv2.threshold(g, 130, 255, cv2.THRESH_BINARY_INV)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k),
                               cv2.MORPH_CLOSE, k)
        rt  = cv2.countNonZero(ink) / (crop.shape[0] * crop.shape[1] or 1)
        cs, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mx = max((cv2.contourArea(c) for c in cs), default=0)
        return (VISION_MIN_INK_RATIO <= rt <= VISION_MAX_INK_RATIO
                and mx >= VISION_MIN_INK_AREA), rt, ink

    def _extract_ink_roi(self, crop, mask):
        if crop is None or mask is None: return crop
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good  = [c for c in cs if cv2.contourArea(c) >= VISION_MIN_INK_AREA]
        if not good: return crop
        x, y, w, h = cv2.boundingRect(np.vstack(good)); m = 30
        return crop[max(0,y-m):min(crop.shape[0],y+h+m),
                    max(0,x-m):min(crop.shape[1],x+w+m)]

    def _run_yolo_on_frame(self, frame) -> Tuple[str, float]:
        if self.yolo_model is None or frame is None:
            return "vision_unavailable", 0.0
        page, _, _ = (self._crop_white_page(frame)
                      if VISION_USE_WHITE_CROP else (frame, None, None))
        if page is None: return "no_white_page", 0.0
        has_ink, _, ink_mask = self._has_black_ink(page)
        if not has_ink: return "no_ink", 0.0
        roi = self._extract_ink_roi(page, ink_mask)
        try:
            res  = self.yolo_model.predict(roi, imgsz=224, verbose=False)
            cid  = int(res[0].probs.top1)
            conf = float(res[0].probs.top1conf)
            return res[0].names[cid], conf
        except Exception as e:
            self._log(f"  YOLO error: {e}")
            return "vision_unavailable", 0.0

    def _make_waypoint_label(self, greek: str) -> str:
        key = greek.strip().lower()
        self.waypoint_counts[key] = self.waypoint_counts.get(key, 0) + 1
        base = key.capitalize(); cnt = self.waypoint_counts[key]
        return base if cnt == 1 else f"{base}{cnt}"

    def _log_to_csv(self, event_id: int, det: Detection,
                    label: str, conf: float, frame,
                    photo_name: str) -> str:
        colour_class, _, colour_detail = self._detect_red_orange(frame)
        if colour_class in ("red", "orange", "yellow"):
            record_type   = "OBJECT"
            display_label = f"OBJECT_{colour_class.upper()}"
            greek_letter  = "none"; letter_conf = 0.0
            vision_detail = colour_detail
        else:
            greek_letter = label; letter_conf = conf
            confirmed = (letter_conf >= SWEEP_CONF_THR
                         and greek_letter not in UNCONFIRMED_LABELS)
            vision_detail = f"yolo: {greek_letter} conf={letter_conf:.2f}"
            if confirmed:
                record_type   = "WAYPOINT"
                display_label = self._make_waypoint_label(greek_letter)
            else:
                record_type   = "OBJECT"
                display_label = "OBJECT_UNKNOWN"

        try:
            with open(BIN_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    event_id, time.strftime("%Y-%m-%dT%H:%M:%S"),
                    record_type, display_label,
                    f"{det.world_x:.4f}", f"{det.world_y:.4f}",
                    f"{det.field_x:.4f}", f"{det.field_y:.4f}",
                    f"{det.range_m:.3f}", f"{math.degrees(det.bearing_rad):.1f}",
                    photo_name,
                    greek_letter, f"{letter_conf:.3f}",
                    colour_class, vision_detail,
                ])
        except Exception as e:
            self._log(f"CSV write failed: {e}")

        self._log(f"#{event_id} [{record_type}] {display_label} | "
                  f"field=({det.field_x:+.2f},{det.field_y:+.2f}) | "
                  f"letter={greek_letter}({letter_conf:.0%}) colour={colour_class}")
        return display_label

    # ══════════════════════════════════════════════════════════════════════
    # Motion primitives
    # ══════════════════════════════════════════════════════════════════════

    def _turn_to_yaw(self, target_yaw: float, timeout_s: float = 8.0) -> bool:
        stable = 0; deadline = time.time() + timeout_s
        while True:
            if self._abort_requested(): self._stop(); return False
            if time.time() > deadline:  self._stop(); self._log("Turn timeout"); return False
            err = self._norm_angle(target_yaw - self.odom_yaw)
            if abs(err) < ANGLE_TOL_RAD:
                stable += 1
                self._send_cmd(0.0, 0.0)
                if stable >= 5: return True
                time.sleep(CONTROL_DT)
                continue
            stable = 0
            mag = max(0.10, min(TURN_SPEED_MAX, abs(1.6 * err)))
            self._send_cmd(0.0, math.copysign(mag, err))
            time.sleep(CONTROL_DT)

    def _drive_straight(self, dist_m: float, forward: bool,
                        safety: bool = True) -> bool:
        if dist_m <= 0.01: return True
        sx, sy = self.odom_x, self.odom_y
        hold_yaw = self.odom_yaw
        sign = 1.0 if forward else -1.0
        deadline = time.time() + max(5.0, dist_m / max(0.1, APPROACH_SPEED) * 3.0)

        while True:
            if self._abort_requested(): self._stop(); return False
            if time.time() > deadline:  self._stop(); return True

            if math.hypot(self.odom_x - sx, self.odom_y - sy) >= dist_m - 0.02:
                self._stop(); return True

            ahead = APPROACH_SPEED * 0.4 * sign
            nx = self.odom_x + ahead * math.cos(self.odom_yaw)
            ny = self.odom_y + ahead * math.sin(self.odom_yaw)
            if not self._inside_field_world(nx, ny, FIELD_BUFFER):
                self._stop(); return True

            if safety and forward and self.safety_obstacle:
                self._stop()
                self._log("  Safety brake during approach")
                return True

            yaw_err = self._norm_angle(hold_yaw - self.odom_yaw)
            ang = max(-0.35, min(0.35, (2.0 if forward else -2.0) * yaw_err))
            self._send_cmd(sign * APPROACH_SPEED, ang)
            time.sleep(CONTROL_DT)

    # ══════════════════════════════════════════════════════════════════════
    # Visit one object
    # ══════════════════════════════════════════════════════════════════════

    def _visit(self, det: Detection) -> bool:
        approach = max(0.0, det.range_m - STANDOFF_M)
        self.event_count += 1
        evt = self.event_count

        self.visited_positions.append((det.world_x, det.world_y))
        self._publish_claimed_perimeter(det.world_x, det.world_y, evt)
        self._publish_hit_marker(evt, det.world_x, det.world_y,
                                 PALETTE[evt % len(PALETTE)])

        self._log(f"→ Object #{evt}: range={det.range_m:.2f} m | "
                  f"field=({det.field_x:+.2f},{det.field_y:+.2f}) | "
                  f"approach={approach:.2f} m")

        if approach > 0.05:
            if not self._drive_straight(approach, forward=True):
                return False
        else:
            self._log("  Already inside standoff — skipping drive")

        time.sleep(CAPTURE_SETTLE_S)
        with self._camera_lock:
            frame = self._grab_frame()
        photo_name = ""
        label, conf = "no_detection", 0.0
        if frame is not None:
            ts   = time.strftime("%Y%m%d_%H%M%S")
            name = f"target_{evt:03d}_{ts}.jpg"
            photo_name = self._save_and_publish_frame(frame, name)
            label, conf = self._run_yolo_on_frame(frame)
            self._log(f"  YOLO → {label} ({conf:.0%})")
        else:
            self._log("  No camera frame — skipping vision")

        self._publish_obstacle_markers(evt, det, label, conf)
        self._log_to_csv(evt, det, label, conf, frame, photo_name)

        actual_moved = math.hypot(self.odom_x - self.home_x,
                                  self.odom_y - self.home_y)
        if actual_moved > 0.05:
            self._log(f"  Reversing {actual_moved:.2f} m to home")
            if not self._drive_straight(actual_moved, forward=False, safety=False):
                return False

        self._log(f"  Visit #{evt} complete")
        return True

    # ══════════════════════════════════════════════════════════════════════
    # Mission top-level — rotate-and-go state machine
    # ══════════════════════════════════════════════════════════════════════

    def _run_mission(self) -> None:
        self._log(f"══ ROTATE-AND-GO START | spin 360° at home, "
                  f"cone=±{DETECT_CONE_DEG/2:.0f}° | "
                  f"emergency={EMERGENCY_STOP_M:.1f} m / "
                  f"±{EMERGENCY_CONE_DEG/2:.0f}° ══")

        accum_rad  = 0.0
        prev_yaw   = self.odom_yaw
        visits     = 0

        with self._lock:
            self._emergency_armed = True

        try:
            while accum_rad < 2.0 * math.pi:
                if self._abort_requested():
                    self._stop()
                    self._log("Mission aborted — MANUAL")
                    return

                det = self._front_detection()

                if det is not None:
                    self._stop()
                    self._log(f"  ⟡ Hit in front: range={det.range_m:.2f} m "
                              f"bearing={math.degrees(det.bearing_rad):+.1f}° "
                              f"world=({det.world_x:+.2f},{det.world_y:+.2f})")

                    # Target is intentionally inside 1 m at standoff — disarm
                    # the heading-cone emergency for the approach leg.
                    with self._lock:
                        self._emergency_armed = False
                    ok = self._visit(det)
                    with self._lock:
                        self._emergency_armed = (self.mode == self.MODE_AUTO)
                    if not ok:
                        return
                    visits += 1

                    prev_yaw = self.odom_yaw
                    time.sleep(0.25)
                    continue

                self._send_cmd(0.0, SCAN_TURN_SPEED)
                time.sleep(CONTROL_DT)
                curr_yaw  = self.odom_yaw
                accum_rad += abs(self._norm_angle(curr_yaw - prev_yaw))
                prev_yaw  = curr_yaw

            self._stop()
            self._log(f"══ DONE | 360° complete | {visits} object(s) | "
                      f"{self.photo_count} photo(s) → {FOR_ML_DIR} ══")
            with self._lock:
                self.auto_phase          = self.PHASE_DONE
                self.mode                = self.MODE_MANUAL
                self.auto_thread_started = False
            m = String(); m.data = self.MODE_MANUAL; self.mode_pub.publish(m)
            self._save_slam_map(f"rotate_and_go_{time.strftime('%Y%m%d_%H%M%S')}")

        finally:
            with self._lock:
                self._emergency_armed = False

    def _save_slam_map(self, stem: str) -> None:
        if not SLAM_TOOLBOX_AVAILABLE or self.save_map_client is None:
            self._log("slam_toolbox unavailable — map save skipped"); return
        if not self.save_map_client.wait_for_service(timeout_sec=4.0):
            self._log("WARNING: save_map service not responding"); return
        req = SaveMap.Request(); path = os.path.join(MAP_DIR, stem)
        try:    req.name.data = path
        except AttributeError: req.name = path
        future = self.save_map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.done() and future.result() is not None:
            self._log(f"Map saved: {path}.pgm/.yaml")
        else:
            self._log("WARNING: map save timed out")

    # ══════════════════════════════════════════════════════════════════════
    # Joy + control loop
    # ══════════════════════════════════════════════════════════════════════

    def joy_callback(self, msg: Joy) -> None:
        self.last_joy = msg
        if self._btn_down(msg, self.btn_circle):
            with self._lock:
                self.mode                 = self.MODE_MANUAL
                self.auto_phase           = self.PHASE_IDLE
                self.auto_thread_started  = False
                self._emergency_armed     = False
                self._was_emergency_abort = False
            self._stop()
            m = String(); m.data = self.MODE_MANUAL; self.mode_pub.publish(m)
            self._log("→ MANUAL (Circle)")
        if self._btn_down(msg, self.btn_x):
            with self._lock:
                if self.auto_phase in (self.PHASE_RUNNING, self.PHASE_WAITING):
                    return
                self.mode            = self.MODE_AUTO
                self.auto_phase      = self.PHASE_WAITING
                self.auto_start_time = time.time()
                self.auto_thread_started = False
            m = String(); m.data = self.MODE_AUTO; self.mode_pub.publish(m)
            self._log(f"→ AUTO in {STARTUP_DELAY_S:.0f} s")

    def control_loop(self) -> None:
        if self.last_joy is None:
            self._send_cmd(); return
        joy = self.last_joy

        if self.mode == self.MODE_MANUAL:
            if not self._axis_pressed(joy, self.axis_l2):
                self._send_cmd(); return
            lin = self._apply_deadzone(
                joy.axes[self.axis_right_y] if self.axis_right_y < len(joy.axes) else 0.0)
            ang = self._apply_deadzone(
                joy.axes[self.axis_left_x]  if self.axis_left_x  < len(joy.axes) else 0.0)
            self._send_cmd(MANUAL_LINEAR_SPEED * lin, MANUAL_ANGULAR_SPEED * ang)
            return

        if self.mode == self.MODE_AUTO:
            with self._lock:
                if self.auto_phase != self.PHASE_WAITING: return
                if (time.time() - self.auto_start_time) < STARTUP_DELAY_S: return
                if not self.odom_ready or not self.lidar_ready: return
                if self.auto_thread_started: return
                self.auto_thread_started = True
                self.auto_phase          = self.PHASE_RUNNING
                resuming = self._was_emergency_abort
                self._was_emergency_abort = False

            if resuming:
                self._log("⟲ Resume AUTO after emergency — keeping visited "
                          f"({len(self.visited_positions)} obj) + trail")
                self._publish_home_marker()
            else:
                self._set_mission_home()
                self.visited_positions.clear()
                self.waypoint_counts.clear()
                self._trail_points.clear()
                self._last_trail_xy = None
                with self._marker_lock: self._markers.clear()
                self._publish_home_marker()
                self.photo_count = 0; self.event_count = 0

            threading.Thread(target=self._run_mission, daemon=True).start()

    def destroy_node(self) -> None:
        try: self._stop()
        except Exception: pass
        if self.pipeline:
            try: self.pipeline.stop()
            except Exception: pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutobotRotateAndGo()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try: node._stop()
        except Exception: pass
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == "__main__":
    main()