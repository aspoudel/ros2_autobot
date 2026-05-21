#!/usr/bin/env python3
"""
autobot_roll.py  —  Greedy nearest-neighbour obstacle hunter
=============================================================

Algorithm (AUTO mode, press X):
  1. Set current pose as mission home (field centre).
  2. SCAN: read one 360° LiDAR snapshot from current position.
           Cluster returns into obstacle candidates inside field.
           Filter out already-visited ones.
  3. PICK: choose nearest unvisited obstacle from current position.
           If none found → go to step 5.
  4. VISIT:
       a. Turn to face obstacle.
       b. Drive forward to STANDOFF_M from obstacle.
       c. Take photo with OAK-D.
       d. Red/yellow colour detect + YOLO Greek letter detect.
       e. Write CSV row, save image to for_ml/.
       f. Mark obstacle world-position as visited.
       → Back to step 2 (scan from new position).
  5. RETURN: drive back to home. Save SLAM map.

MANUAL: hold L2 + sticks.
ABORT:  Circle → MANUAL instantly.
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
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy, LaserScan
from std_msgs.msg import String

# ── Optional heavy deps ────────────────────────────────────────────────────────
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


# ─────────────────────────────── Tunables ─────────────────────────────────────

AREA_SIZE   = 15.0
FIELD_HALF  = AREA_SIZE / 2.0

# LiDAR clustering
RANGE_MIN_VALID   = 0.25          # m — ignore chassis noise
RANGE_MAX_USE     = FIELD_HALF * 1.5
OBJ_WIDTH_MIN_M   = 0.05          # m — narrowest cluster accepted
OBJ_WIDTH_MAX_M   = 1.20          # m — widest cluster accepted
CLUSTER_GAP_M     = 0.35          # m — range jump that splits clusters
CLUSTER_MIN_BEAMS = 3

# Visited dedup
VISITED_RADIUS_M  = 0.50          # m — same obstacle if centres within this

# Approach
STANDOFF_M        = 0.50          # m — stop this far from obstacle face
MIN_APPROACH_M    = 0.10

# Speed / turn
AUTO_SPEED           = 0.35       # m/s
MANUAL_LINEAR_SPEED  = 0.50
MANUAL_ANGULAR_SPEED = 1.20
TURN_SPEED_MAX       = 0.80       # rad/s
ANGLE_TOL_RAD        = math.radians(2.0)
CONTROL_DT           = 0.03       # s

# Safety brake while driving forward
SAFETY_STOP_M   = 0.25
SAFETY_CONE_DEG = 30.0

STARTUP_DELAY_S = 3.0

# Vision (identical values to existing code)
VISION_CONFIDENCE_THR = 0.70
VISION_USE_WHITE_CROP = True
VISION_MIN_INK_RATIO  = 0.005
VISION_MAX_INK_RATIO  = 0.35
VISION_MIN_INK_AREA   = 80
VISION_WHITE_MIN_AREA = 5000
COLOUR_MIN_AREA_PX    = 2500
COLOUR_MIN_RATIO      = 0.015

_HERE      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "best.pt")

FOR_ML_DIR   = "/root/ros2_autobot/for_ml"
BIN_CSV_PATH = "/root/ros2_autobot/for_ml/bin_positions.csv"
MAP_DIR      = "/root/ros2_autobot/maps"


@dataclass
class Obstacle:
    world_x:     float
    world_y:     float
    field_x:     float
    field_y:     float
    range_m:     float
    bearing_rad: float


class AutobotGreedy(Node):

    MODE_MANUAL = "manual"
    MODE_AUTO   = "auto"
    PHASE_IDLE    = "IDLE"
    PHASE_WAITING = "WAITING"
    PHASE_RUNNING = "RUNNING"
    PHASE_DONE    = "DONE"

    def __init__(self) -> None:
        super().__init__("autobot_greedy")
        self.cb_group = ReentrantCallbackGroup()

        self.declare_parameter("btn_x",           0)
        self.declare_parameter("btn_circle",      1)
        self.declare_parameter("axis_left_x",     0)
        self.declare_parameter("axis_right_y",    3)
        self.declare_parameter("axis_l2",         4)
        self.declare_parameter("deadzone",        0.10)
        self.declare_parameter("publish_rate_hz", 20.0)

        self.btn_x           = int(self.get_parameter("btn_x").value)
        self.btn_circle      = int(self.get_parameter("btn_circle").value)
        self.axis_left_x     = int(self.get_parameter("axis_left_x").value)
        self.axis_right_y    = int(self.get_parameter("axis_right_y").value)
        self.axis_l2         = int(self.get_parameter("axis_l2").value)
        self.deadzone        = float(self.get_parameter("deadzone").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self._lock        = threading.Lock()
        self._camera_lock = threading.Lock()

        self.mode                = self.MODE_MANUAL
        self.auto_phase          = self.PHASE_IDLE
        self.auto_start_time     = 0.0
        self.auto_thread_started = False
        self.last_joy: Optional[Joy] = None

        self.odom_ready = False
        self.odom_x = 0.0; self.odom_y = 0.0; self.odom_yaw = 0.0

        self.home_x = 0.0; self.home_y = 0.0; self.home_yaw = 0.0

        self.lidar_ready     = False
        self.latest_scan: Optional[LaserScan] = None
        self.safety_obstacle = False

        self.camera_ready = False
        self.pipeline     = None
        self.q_rgb        = None
        self.bridge       = CvBridge()

        self.visited_positions: List[Tuple[float, float]] = []
        self.waypoint_counts = {}
        self.photo_count  = 0
        self.event_count  = 0

        self.cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",           10)
        self.mode_pub   = self.create_publisher(String, "/control_mode",      10)
        self.status_pub = self.create_publisher(String, "/robot_status",      10)
        self.image_pub  = self.create_publisher(Image,  "/oak/rgb/image_raw", 10)

        self.yolo_model = None
        self._init_vision_model()

        self.create_subscription(Joy,       "/joy",  self.joy_callback,   10, callback_group=self.cb_group)
        self.create_subscription(Odometry,  "/odom", self.odom_callback,  10, callback_group=self.cb_group)
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10, callback_group=self.cb_group)

        self.save_map_client = None
        if SLAM_TOOLBOX_AVAILABLE:
            os.makedirs(MAP_DIR, exist_ok=True)
            self.save_map_client = self.create_client(SaveMap, "/slam_toolbox/save_map")

        self.create_timer(1.0 / self.publish_rate_hz, self.control_loop, callback_group=self.cb_group)

        self._init_dirs_csv()
        self._init_camera()

        self._log(
            f"Greedy obstacle hunter ready | {AREA_SIZE}×{AREA_SIZE} m | "
            f"standoff={STANDOFF_M} m | YOLO={'OK' if self.yolo_model else 'MISSING'} | "
            f"MANUAL: hold L2 | AUTO: X | ABORT: Circle"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Setup
    # ══════════════════════════════════════════════════════════════════════════

    def _init_dirs_csv(self) -> None:
        os.makedirs(FOR_ML_DIR, exist_ok=True)
        if not os.path.exists(BIN_CSV_PATH):
            with open(BIN_CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerow([
                    "event_id", "timestamp", "record_type", "display_label",
                    "world_x", "world_y", "field_x", "field_y",
                    "range_m", "heading_deg", "photo_file",
                    "greek_letter", "letter_confidence", "colour_class", "vision_detail",
                ])

    def _init_camera(self) -> None:
        if not DEPTHAI_AVAILABLE:
            self._log("WARNING: depthai not installed — camera disabled")
            return
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
            self._log("WARNING: ultralytics not installed")
            return
        if not os.path.exists(MODEL_PATH):
            self._log(f"WARNING: best.pt not found at {MODEL_PATH}")
            return
        try:
            self.yolo_model = YOLO(MODEL_PATH)
            self._log(f"YOLO loaded: {MODEL_PATH}")
        except Exception as e:
            self._log(f"WARNING: YOLO load failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════════
    # Odometry + field frame
    # ══════════════════════════════════════════════════════════════════════════

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
            self._log(f"Odom ready ({self.odom_x:.2f},{self.odom_y:.2f}) yaw={math.degrees(self.odom_yaw):.1f}°")

    def _set_mission_home(self) -> None:
        self.home_x = self.odom_x; self.home_y = self.odom_y; self.home_yaw = self.odom_yaw
        self._log(f"Home: ({self.home_x:.2f},{self.home_y:.2f}) yaw={math.degrees(self.home_yaw):.1f}°")

    def _world_to_field(self, wx: float, wy: float) -> Tuple[float, float]:
        dx = wx - self.home_x; dy = wy - self.home_y
        c = math.cos(self.home_yaw); s = math.sin(self.home_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _inside_field(self, fx: float, fy: float) -> bool:
        return abs(fx) <= FIELD_HALF and abs(fy) <= FIELD_HALF

    # ══════════════════════════════════════════════════════════════════════════
    # LiDAR
    # ══════════════════════════════════════════════════════════════════════════

    def lidar_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        if not self.lidar_ready:
            self.lidar_ready = True
            self._log(f"LiDAR ready: {len(msg.ranges)} rays [{msg.range_min:.2f},{msg.range_max:.2f}] m")

        half = math.radians(SAFETY_CONE_DEG / 2.0)
        hit = False
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            if abs(a) > half or not math.isfinite(r) or r < 0.10:
                continue
            if r < SAFETY_STOP_M:
                hit = True; break
        self.safety_obstacle = hit

    def _scan_obstacles(self) -> List[Obstacle]:
        """
        Read the current LiDAR snapshot (full 360°).
        Cluster contiguous beams by range continuity.
        Convert each cluster to an Obstacle in world frame.
        Filter: inside field, plausible width, not already visited.
        """
        scan = self.latest_scan
        if scan is None:
            return []

        pts: List[Tuple[float, float]] = []
        for i, r in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            if not math.isfinite(r):
                continue
            if r < RANGE_MIN_VALID or r > min(scan.range_max, RANGE_MAX_USE):
                continue
            pts.append((a, r))

        if not pts:
            return []

        clusters: List[List[Tuple[float, float]]] = []
        cluster = [pts[0]]
        for p in pts[1:]:
            if abs(p[1] - cluster[-1][1]) <= CLUSTER_GAP_M:
                cluster.append(p)
            else:
                clusters.append(cluster); cluster = [p]
        clusters.append(cluster)

        results: List[Obstacle] = []
        for cl in clusters:
            if len(cl) < CLUSTER_MIN_BEAMS:
                continue
            span  = abs(cl[-1][0] - cl[0][0])
            r_mid = sum(r for _, r in cl) / len(cl)
            if not (OBJ_WIDTH_MIN_M <= r_mid * span <= OBJ_WIDTH_MAX_M):
                continue

            bearing = sum(a for a, _ in cl) / len(cl)
            rng     = min(r for _, r in cl)
            wb      = self._norm_angle(self.odom_yaw + bearing)
            wx      = self.odom_x + rng * math.cos(wb)
            wy      = self.odom_y + rng * math.sin(wb)
            fx, fy  = self._world_to_field(wx, wy)

            if not self._inside_field(fx, fy):
                continue
            if self._already_visited(wx, wy):
                continue

            results.append(Obstacle(wx, wy, fx, fy, rng, bearing))

        return results

    def _already_visited(self, wx: float, wy: float) -> bool:
        return any(
            math.hypot(wx - px, wy - py) <= VISITED_RADIUS_M
            for px, py in self.visited_positions
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Vision — identical logic to existing working code
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_red_yellow_object(self, frame) -> Tuple[str, float, str]:
        if frame is None:
            return "none", 0.0, "No frame"
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        r1  = cv2.inRange(hsv, (0,   80, 80), (10,  255, 255))
        r2  = cv2.inRange(hsv, (170, 80, 80), (180, 255, 255))
        rm  = cv2.bitwise_or(r1, r2)
        ym  = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        rm  = cv2.morphologyEx(cv2.morphologyEx(rm, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        ym  = cv2.morphologyEx(cv2.morphologyEx(ym, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        tot = frame.shape[0] * frame.shape[1]

        def sc(m):
            px = cv2.countNonZero(m)
            rt = px / tot if tot > 0 else 0.0
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ar = max((cv2.contourArea(c) for c in cs), default=0.0)
            return rt, ar

        rr, ra = sc(rm); yr, ya = sc(ym)
        if ra >= COLOUR_MIN_AREA_PX and rr >= COLOUR_MIN_RATIO and rr >= yr:
            return "red",    rr, f"red ratio={rr:.3f}"
        if ya >= COLOUR_MIN_AREA_PX and yr >= COLOUR_MIN_RATIO:
            return "yellow", yr, f"yellow ratio={yr:.3f}"
        return "none", max(rr, yr), f"none (r={rr:.3f} y={yr:.3f})"

    def _crop_white_page(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
        k   = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        msk = cv2.morphologyEx(cv2.morphologyEx(msk, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
        cs, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None, None, msk
        lg = max(cs, key=cv2.contourArea)
        if cv2.contourArea(lg) < VISION_WHITE_MIN_AREA:
            return None, None, msk
        x, y, w, h = cv2.boundingRect(lg); m = 20
        return (frame[max(0,y-m):min(frame.shape[0],y+h+m),
                      max(0,x-m):min(frame.shape[1],x+w+m)],
                (x, y, w, h), msk)

    def _has_black_ink(self, crop):
        if crop is None or crop.size == 0:
            return False, 0.0, None
        g   = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, ink = cv2.threshold(g, 130, 255, cv2.THRESH_BINARY_INV)
        k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink = cv2.morphologyEx(cv2.morphologyEx(ink, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        rt  = cv2.countNonZero(ink) / (crop.shape[0] * crop.shape[1] or 1)
        cs, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mx  = max((cv2.contourArea(c) for c in cs), default=0)
        return (VISION_MIN_INK_RATIO <= rt <= VISION_MAX_INK_RATIO and mx >= VISION_MIN_INK_AREA), rt, ink

    def _extract_ink_roi(self, crop, mask):
        if crop is None or mask is None:
            return crop
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good  = [c for c in cs if cv2.contourArea(c) >= VISION_MIN_INK_AREA]
        if not good:
            return crop
        x, y, w, h = cv2.boundingRect(np.vstack(good)); m = 30
        return crop[max(0,y-m):min(crop.shape[0],y+h+m),
                    max(0,x-m):min(crop.shape[1],x+w+m)]

    def _detect_greek_letter(self, frame) -> Tuple[str, float, str]:
        if self.yolo_model is None or frame is None:
            return "vision_unavailable", 0.0, "YOLO not loaded or no frame"
        page, _, _ = self._crop_white_page(frame) if VISION_USE_WHITE_CROP else (frame, None, None)
        if page is None:
            return "no_white_page", 0.0, "No white page"
        has_ink, ratio, ink_mask = self._has_black_ink(page)
        if not has_ink:
            return "no_ink", 0.0, f"ink_ratio={ratio:.4f}"
        roi = self._extract_ink_roi(page, ink_mask)
        try:
            res   = self.yolo_model.predict(roi, imgsz=224, verbose=False)
            cid   = int(res[0].probs.top1)
            conf  = float(res[0].probs.top1conf)
            label = res[0].names[cid]
        except Exception as e:
            return "vision_unavailable", 0.0, str(e)
        if conf < VISION_CONFIDENCE_THR:
            return "low_confidence", conf, f"{label} conf={conf:.2f}"
        if label == "not_letter":
            return "not_letter", conf, f"not_letter conf={conf:.2f}"
        return label, conf, f"{label} conf={conf:.2f} ink={ratio:.4f}"

    # ══════════════════════════════════════════════════════════════════════════
    # Camera
    # ══════════════════════════════════════════════════════════════════════════

    def _capture_frame_and_save(self, event_id: int) -> Tuple[str, Optional[np.ndarray]]:
        with self._camera_lock:
            if not self.camera_ready:
                self._log("Camera not ready"); return "", None
            frame = None; deadline = time.time() + 1.5
            while time.time() < deadline:
                try:
                    pkt = self.q_rgb.tryGet()
                    if pkt:
                        frame = pkt.getCvFrame(); break
                except Exception:
                    pass
                time.sleep(0.05)
            if frame is None:
                self._log("No camera frame"); return "", None

            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(msg)

            self.photo_count += 1
            name = f"target_{event_id:03d}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(FOR_ML_DIR, name)
            try:
                if not cv2.imwrite(path, frame):
                    name = ""
                else:
                    self._log(f"Photo → {path}")
            except Exception as e:
                self._log(f"Photo save error: {e}"); name = ""
            return name, frame

    # ══════════════════════════════════════════════════════════════════════════
    # CSV logging
    # ══════════════════════════════════════════════════════════════════════════

    def _make_waypoint_label(self, greek: str) -> str:
        key = greek.strip().lower()
        self.waypoint_counts[key] = self.waypoint_counts.get(key, 0) + 1
        base = key.capitalize()
        cnt  = self.waypoint_counts[key]
        return base if cnt == 1 else f"{base}{cnt}"

    def _log_to_csv(self, event_id: int, obs: Obstacle,
                    photo_name: str, frame) -> None:
        colour_class, _, colour_detail = self._detect_red_yellow_object(frame)
        if colour_class in ("red", "yellow"):
            record_type   = "OBJECT"
            display_label = f"OBJECT_{colour_class.upper()}"
            greek_letter  = "none"; letter_conf = 0.0; vision_detail = colour_detail
        else:
            greek_letter, letter_conf, vision_detail = self._detect_greek_letter(frame)
            valid = (
                greek_letter not in (
                    "vision_unavailable", "no_white_page",
                    "no_ink", "low_confidence", "not_letter",
                ) and letter_conf >= VISION_CONFIDENCE_THR
            )
            if valid:
                record_type = "WAYPOINT"; display_label = self._make_waypoint_label(greek_letter)
            else:
                record_type = "OBJECT"; display_label = "OBJECT_UNKNOWN"

        try:
            with open(BIN_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    event_id, time.strftime("%Y-%m-%dT%H:%M:%S"),
                    record_type, display_label,
                    f"{obs.world_x:.4f}", f"{obs.world_y:.4f}",
                    f"{obs.field_x:.4f}", f"{obs.field_y:.4f}",
                    f"{obs.range_m:.3f}", f"{math.degrees(obs.bearing_rad):.1f}",
                    photo_name, greek_letter, f"{letter_conf:.3f}",
                    colour_class, vision_detail,
                ])
        except Exception as e:
            self._log(f"CSV write failed: {e}")

        self._log(
            f"#{event_id} [{record_type}] {display_label} | "
            f"field=({obs.field_x:.2f},{obs.field_y:.2f}) | "
            f"letter={greek_letter} ({letter_conf:.0%}) | colour={colour_class}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Motion primitives
    # ══════════════════════════════════════════════════════════════════════════

    def _turn_to_yaw(self, target_yaw: float) -> bool:
        stable = 0
        while True:
            with self._lock:
                if self.mode == self.MODE_MANUAL:
                    self._stop(); return False
            err = self._norm_angle(target_yaw - self.odom_yaw)
            if abs(err) < ANGLE_TOL_RAD:
                stable += 1
                if stable >= 5:
                    self._stop(); return True
            else:
                stable = 0
            mag = min(TURN_SPEED_MAX, max(0.18, abs(1.8 * err)))
            self._send_cmd(0.0, math.copysign(mag, err))
            time.sleep(CONTROL_DT)

    def _drive(self, dist_m: float, forward: bool = True) -> bool:
        if dist_m <= 0.01:
            return True
        sx = self.odom_x; sy = self.odom_y
        hold_yaw = self.odom_yaw
        sign = 1.0 if forward else -1.0
        while True:
            with self._lock:
                if self.mode == self.MODE_MANUAL:
                    self._stop(); return False
            if math.hypot(self.odom_x - sx, self.odom_y - sy) >= dist_m - 0.01:
                self._stop(); return True
            if forward and self.safety_obstacle:
                self._stop()
                self._log("Safety stop"); return True
            yaw_err = self._norm_angle(hold_yaw - self.odom_yaw)
            ang = max(-0.6, min(0.6, (2.0 if forward else -2.0) * yaw_err))
            self._send_cmd(sign * AUTO_SPEED, ang)
            time.sleep(CONTROL_DT)

    # ══════════════════════════════════════════════════════════════════════════
    # Mission
    # ══════════════════════════════════════════════════════════════════════════

    def _visit(self, obs: Obstacle) -> bool:
        dx = obs.world_x - self.odom_x
        dy = obs.world_y - self.odom_y
        self._log(f"  Face ({obs.field_x:.2f},{obs.field_y:.2f}) range={obs.range_m:.2f} m")

        if not self._turn_to_yaw(math.atan2(dy, dx)):
            return False

        approach = max(0.0, math.hypot(dx, dy) - STANDOFF_M)
        if approach >= MIN_APPROACH_M:
            if not self._drive(approach):
                return False

        self.event_count += 1
        photo, frame = self._capture_frame_and_save(self.event_count)
        self._log_to_csv(self.event_count, obs, photo, frame)
        self.visited_positions.append((obs.world_x, obs.world_y))
        return True

    def _return_to_home(self) -> bool:
        dx = self.home_x - self.odom_x
        dy = self.home_y - self.odom_y
        dist = math.hypot(dx, dy)
        if dist < 0.15:
            self._log("Already at home"); return True
        self._log(f"Return home: {dist:.2f} m")
        if not self._turn_to_yaw(math.atan2(dy, dx)):
            return False
        return self._drive(dist)

    def _save_slam_map(self, stem: str) -> None:
        if not SLAM_TOOLBOX_AVAILABLE or self.save_map_client is None:
            return
        if not self.save_map_client.wait_for_service(timeout_sec=4.0):
            self._log("WARNING: save_map service unavailable"); return
        req = SaveMap.Request()
        path = os.path.join(MAP_DIR, stem)
        try:
            req.name.data = path
        except AttributeError:
            req.name = path
        future = self.save_map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
        if future.done() and future.result() is not None:
            self._log(f"Map saved: {path}.pgm/.yaml")

    def _run_mission(self) -> None:
        self._log("══ GREEDY OBSTACLE HUNT START ══")
        step = 0

        while True:
            with self._lock:
                if self.mode == self.MODE_MANUAL:
                    self._stop(); return

            candidates = self._scan_obstacles()

            self._log(f"Step {step}: {len(candidates)} unvisited obstacle(s) visible")

            if not candidates:
                self._log("None found — heading home")
                break

            # Greedy: nearest from current robot position
            target = min(
                candidates,
                key=lambda o: math.hypot(o.world_x - self.odom_x, o.world_y - self.odom_y),
            )

            if not self._visit(target):
                return

            step += 1

        self._return_to_home()

        self._log(f"══ DONE — {self.event_count} obstacle(s) | Photos → {FOR_ML_DIR} ══")
        with self._lock:
            self.auto_phase = self.PHASE_DONE

        self._save_slam_map(f"hunt_{time.strftime('%Y%m%d_%H%M%S')}")

    # ══════════════════════════════════════════════════════════════════════════
    # Joy + control loop (unchanged from working code)
    # ══════════════════════════════════════════════════════════════════════════

    def joy_callback(self, msg: Joy) -> None:
        self.last_joy = msg
        if self._btn_down(msg, self.btn_circle):
            with self._lock:
                self.mode = self.MODE_MANUAL
                self.auto_phase = self.PHASE_IDLE
                self.auto_thread_started = False
            self._stop()
            m = String(); m.data = self.MODE_MANUAL; self.mode_pub.publish(m)
            self._log("→ MANUAL")

        if self._btn_down(msg, self.btn_x):
            with self._lock:
                if self.auto_phase == self.PHASE_RUNNING:
                    return
                self.mode = self.MODE_AUTO
                self.auto_phase = self.PHASE_WAITING
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
            lin = self._apply_deadzone(joy.axes[self.axis_right_y] if self.axis_right_y < len(joy.axes) else 0.0)
            ang = self._apply_deadzone(joy.axes[self.axis_left_x]  if self.axis_left_x  < len(joy.axes) else 0.0)
            self._send_cmd(MANUAL_LINEAR_SPEED * lin, MANUAL_ANGULAR_SPEED * ang)
            return

        if self.mode == self.MODE_AUTO:
            with self._lock:
                phase = self.auto_phase; started = self.auto_thread_started
            if phase == self.PHASE_WAITING:
                if (time.time() - self.auto_start_time) < STARTUP_DELAY_S:
                    return
                if not self.odom_ready or not self.lidar_ready:
                    self._log("Waiting for odom/lidar..."); return
                if started:
                    return
                self._set_mission_home()
                self.visited_positions.clear()
                self.waypoint_counts.clear()
                self.photo_count = 0; self.event_count = 0
                with self._lock:
                    self.auto_phase = self.PHASE_RUNNING
                    self.auto_thread_started = True
                threading.Thread(target=self._run_mission, daemon=True).start()

    def destroy_node(self) -> None:
        self._stop()
        if self.pipeline:
            try: self.pipeline.stop()
            except Exception: pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutobotGreedy()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop(); node.destroy_node()
        try: rclpy.shutdown()
        except Exception: pass


if __name__ == "__main__":
    main()