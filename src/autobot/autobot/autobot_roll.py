#!/usr/bin/env python3
"""
autobot_roll.py — Rotate-and-visit bin hunter with Greek letter detection
==========================================================================
Behaviour
---------
  AUTO mode (press X):
    1. Rotate slowly in place from home (field centre).
    2. Every control tick, check a narrow forward cone for objects
       inside the field boundary.
    3. When something is detected straight ahead:
         a. Stop rotating.
         b. Drive forward to APPROACH_STANDOFF_M from the object.
         c. Capture one camera frame.
            → Save JPEG to FOR_ML_DIR (for_ml/).
            → Run Greek-letter vision pipeline on the SAME in-memory frame.
               (crop white page → ink gate → YOLO classify)
         d. Log to CSV: bin position + detected Greek letter + confidence.
         e. Reverse straight back to home.
         f. Resume rotating.
    4. When 360° of accumulated rotation is complete → mission done.
    5. Save SLAM map.

  MANUAL mode (hold L2 + right-stick): normal teleop.
  Circle → abort AUTO → MANUAL at any time.

Greek-letter model
------------------
  Place best.pt in the same directory as this file:
    /root/ros2_autobot/src/autobot/autobot/best.pt

  Required pip packages (add to Dockerfile, then rebuild):
    pip3 install --break-system-packages "numpy<2" torch torchvision ultralytics

Vision pipeline (ported from alpha_search_node.py)
--------------------------------------------------
  1. crop_white_page  — find largest white region via HSV
  2. has_black_ink    — ink-ratio + connected-component gate
  3. extract_ink_roi  — tight crop around ink blobs
  4. YOLO predict     — classification model, imgsz=224
  Outputs one of: alpha / beta / gamma / not_letter / no_white_page /
                  no_ink / low_confidence / vision_unavailable
"""

import csv
import math
import os
import threading
import time
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


# ─────────────────────────── Tunables ────────────────────────────────────────

AREA_SIZE            = 5.0    # m — change this to resize the field (e.g. 15.0)
FIELD_HALF           = AREA_SIZE / 2.0

# Detection
MAX_DETECT_RANGE     = FIELD_HALF * math.sqrt(2)  # field diagonal; auto-scales
DETECT_CONE_DEG      = 20.0   # total cone — fires only when object is truly ahead
CHASSIS_MIN_M        = 0.25   # m — ignore returns closer (chassis noise)
DEDUP_RADIUS_M       = 0.60   # m — don't re-visit within this radius

# Approach / photo
APPROACH_STANDOFF_M  = 0.70   # m — stop this far from object face
AUTO_SPEED           = 0.35   # m/s — forward and reverse (same magnitude)

# Rotation
SCAN_TURN_SPEED      = 0.50   # rad/s — slow scan rotation
TURN_SPEED_MAX       = 0.80   # rad/s
CONTROL_DT           = 0.03   # s
ANGLE_TOL_RAD        = math.radians(2.0)

# Startup
STARTUP_DELAY_S      = 3.0

# Manual
MANUAL_LINEAR_SPEED  = 0.50
MANUAL_ANGULAR_SPEED = 1.20

# Safety brake during forward drive
SAFETY_STOP_M        = 0.25   # m — emergency stop threshold
SAFETY_CONE_DEG      = 30.0   # total cone width for safety brake

# ── Vision tunables (mirrors alpha_search_node.py defaults) ───────────────────
VISION_CONFIDENCE_THR = 0.70  # minimum confidence to trust label
VISION_USE_WHITE_CROP = True  # crop to white page before classifying
VISION_MIN_INK_RATIO  = 0.005 # minimum ink pixel fraction
VISION_MAX_INK_RATIO  = 0.35  # maximum ink pixel fraction (avoid solid blobs)
VISION_MIN_INK_AREA   = 80    # minimum connected-component area (pixels)
VISION_WHITE_MIN_AREA = 5000  # minimum white-blob area to be a page

# ── Model path — same directory as this script ────────────────────────────────
_HERE            = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(_HERE, "best.pt")

# ── Output paths ──────────────────────────────────────────────────────────────
FOR_ML_DIR       = "/root/ros2_autobot/for_ml"
BIN_CSV_PATH     = "/root/ros2_autobot/for_ml/bin_positions.csv"
MAP_DIR          = "/root/ros2_autobot/maps"

# ─────────────────────────────────────────────────────────────────────────────


class DualShockBinHunt(Node):

    MODE_MANUAL = "manual"
    MODE_AUTO   = "auto"

    PHASE_IDLE    = "IDLE"
    PHASE_WAITING = "WAITING"
    PHASE_RUNNING = "RUNNING"
    PHASE_DONE    = "DONE"

    # ══════════════════════════════════════════════════════════════════════════
    # Init
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(self) -> None:
        super().__init__("dualshock_bin_hunt")
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

        # Shared state
        self._lock        = threading.Lock()
        self._camera_lock = threading.Lock()

        self.mode                = self.MODE_MANUAL
        self.auto_phase          = self.PHASE_IDLE
        self.auto_start_time     = 0.0
        self.auto_thread_started = False
        self.last_joy: Optional[Joy] = None

        # Odometry
        self.odom_ready = False
        self.odom_x     = 0.0
        self.odom_y     = 0.0
        self.odom_yaw   = 0.0

        # Mission home
        self.home_x   = 0.0
        self.home_y   = 0.0
        self.home_yaw = 0.0

        # LiDAR
        self.lidar_ready          = False
        self.latest_scan: Optional[LaserScan] = None
        self.safety_obstacle      = False

        # Camera (depthai pipeline)
        self.camera_ready = False
        self.pipeline     = None
        self.q_rgb        = None
        self.bridge       = CvBridge()


        # Mission bookkeeping
        self.visited_positions: List[Tuple[float, float]] = []
        self.photo_count = 0

        # Publishers
        self.cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",           10)
        self.mode_pub   = self.create_publisher(String, "/control_mode",      10)
        self.status_pub = self.create_publisher(String, "/robot_status",      10)
        self.image_pub  = self.create_publisher(Image,  "/oak/rgb/image_raw", 10)

        # Vision model
        self.yolo_model   = None
        self._init_vision_model()

        # Subscriptions
        self.create_subscription(Joy,       "/joy",  self.joy_callback,   10, callback_group=self.cb_group)
        self.create_subscription(Odometry,  "/odom", self.odom_callback,  10, callback_group=self.cb_group)
        self.create_subscription(LaserScan, "/scan", self.lidar_callback, 10, callback_group=self.cb_group)

        # Map save service
        self.save_map_client = None
        if SLAM_TOOLBOX_AVAILABLE:
            os.makedirs(MAP_DIR, exist_ok=True)
            self.save_map_client = self.create_client(SaveMap, "/slam_toolbox/save_map")

        self.create_timer(
            1.0 / self.publish_rate_hz, self.control_loop,
            callback_group=self.cb_group,
        )

        self._init_dirs_csv()
        self._init_camera()

        self._log(
            f"Bin-hunt + Greek-letter detection ready\n"
            f"  Field:      {AREA_SIZE}x{AREA_SIZE} m\n"
            f"  Max detect: {MAX_DETECT_RANGE:.1f} m\n"
            f"  Standoff:   {APPROACH_STANDOFF_M} m\n"
            f"  Cone:       +-{DETECT_CONE_DEG/2:.0f} deg\n"
            f"  YOLO model: {'LOADED' if self.yolo_model else 'UNAVAILABLE'} "
            f"({MODEL_PATH})\n"
            f"  MANUAL: hold L2 + stick | AUTO: press X | ABORT: Circle"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Setup
    # ══════════════════════════════════════════════════════════════════════════

    def _init_dirs_csv(self) -> None:
        os.makedirs(FOR_ML_DIR, exist_ok=True)
        if not os.path.exists(BIN_CSV_PATH):
            with open(BIN_CSV_PATH, "w", newline="") as f:
                csv.writer(f).writerow([
                    "visit_id", "timestamp",
                    "world_x", "world_y",
                    "field_x", "field_y",
                    "range_m", "heading_deg",
                    "photo_file",
                    # Greek-letter columns
                    "greek_letter",
                    "letter_confidence",
                    "vision_detail",
                ])
        self._log(f"CSV: {BIN_CSV_PATH}")

    def _init_camera(self) -> None:
        if not DEPTHAI_AVAILABLE:
            self._log("WARNING: depthai not installed — camera disabled")
            return
        try:
            self.pipeline = dai.Pipeline()
            cam = self.pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A
            )
            out = cam.requestOutput((1920, 1080), type=dai.ImgFrame.Type.BGR888p)
            self.q_rgb = out.createOutputQueue()
            self.pipeline.start()
            self.camera_ready = True
            self._log(f"OAK-D camera ready -> {FOR_ML_DIR}")
        except Exception as e:
            self.camera_ready = False
            self._log(f"WARNING: OAK-D init failed: {e}")

    def _init_vision_model(self) -> None:
        """Load YOLO classification model (best.pt)."""
        if not ULTRALYTICS_AVAILABLE:
            self._log("WARNING: ultralytics not installed — Greek letter detection disabled")
            return
        if not os.path.exists(MODEL_PATH):
            self._log(f"WARNING: best.pt not found at {MODEL_PATH} — detection disabled")
            return
        try:
            self.yolo_model = YOLO(MODEL_PATH)
            self._log(f"YOLO model loaded: {MODEL_PATH}")
        except Exception as e:
            self.yolo_model = None
            self._log(f"WARNING: YOLO model load failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Greek-letter vision pipeline
    # (ported directly from alpha_search_node.py — same logic, no disk I/O)
    # ══════════════════════════════════════════════════════════════════════════

    def _crop_white_page(self, frame: np.ndarray):
        """
        Find the largest white region in the frame using HSV thresholding.
        Returns (crop, bbox, mask) or (None, None, mask) if nothing found.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = (0,   0,   150)
        upper_white = (180, 80,  255)
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < VISION_WHITE_MIN_AREA:
            return None, None, mask

        x, y, w, h = cv2.boundingRect(largest)
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2), mask

    def _has_black_ink(self, crop: np.ndarray):
        """
        Check whether the white crop contains enough dark ink.
        Returns (has_ink: bool, ink_ratio: float, ink_mask).
        """
        if crop is None or crop.size == 0:
            return False, 0.0, None

        gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, ink_mask = cv2.threshold(gray_blur, 130, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN,  kernel)
        ink_mask = cv2.morphologyEx(ink_mask, cv2.MORPH_CLOSE, kernel)

        ink_pixels   = cv2.countNonZero(ink_mask)
        total_pixels = crop.shape[0] * crop.shape[1]
        ink_ratio    = ink_pixels / total_pixels if total_pixels > 0 else 0.0

        contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = max((cv2.contourArea(c) for c in contours), default=0)

        has_ink = (
            VISION_MIN_INK_RATIO <= ink_ratio <= VISION_MAX_INK_RATIO
            and max_area >= VISION_MIN_INK_AREA
        )
        return has_ink, ink_ratio, ink_mask

    def _extract_ink_roi(self, crop: np.ndarray, ink_mask: np.ndarray) -> np.ndarray:
        """
        Crop tightly around ink blobs so YOLO sees the letter shape,
        not a large white background.
        """
        if crop is None or ink_mask is None:
            return crop

        contours, _ = cv2.findContours(ink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good = [c for c in contours if cv2.contourArea(c) >= VISION_MIN_INK_AREA]
        if not good:
            return crop

        all_pts = np.vstack(good)
        x, y, w, h = cv2.boundingRect(all_pts)
        margin = 30
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(crop.shape[1], x + w + margin)
        y2 = min(crop.shape[0], y + h + margin)
        return crop[y1:y2, x1:x2]

    def _yolo_predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Run YOLO classifier on image. Returns (label, confidence)."""
        results    = self.yolo_model.predict(image, imgsz=224, verbose=False)
        probs      = results[0].probs
        class_id   = int(probs.top1)
        confidence = float(probs.top1conf)
        label      = results[0].names[class_id]
        return label, confidence

    def _detect_greek_letter(self, frame: np.ndarray) -> Tuple[str, float, str]:
        """
        Full vision pipeline on an in-memory BGR frame.

        Returns
        -------
        (greek_letter, confidence, detail_string)

        greek_letter is one of:
            alpha | beta | gamma | <other model label>
            not_letter | no_white_page | no_ink | low_confidence
            vision_unavailable
        """
        if self.yolo_model is None:
            return "vision_unavailable", 0.0, "YOLO model not loaded"

        if frame is None:
            return "vision_unavailable", 0.0, "No frame"

        # Step 1 — find white page
        if VISION_USE_WHITE_CROP:
            page_crop, _bbox, _mask = self._crop_white_page(frame)
        else:
            page_crop = frame

        if page_crop is None:
            detail = "AWW — no white page found"
            self._log(f"[Vision] {detail}")
            return "no_white_page", 0.0, detail

        # Step 2 — ink gate
        has_ink, ink_ratio, ink_mask = self._has_black_ink(page_crop)
        if not has_ink:
            detail = f"AWW — white page found but not enough ink (ink_ratio={ink_ratio:.4f})"
            self._log(f"[Vision] {detail}")
            return "no_ink", 0.0, detail

        # Step 3 — tight crop around ink
        image_for_model = self._extract_ink_roi(page_crop, ink_mask)

        # Step 4 — YOLO classify
        try:
            label, confidence = self._yolo_predict(image_for_model)
        except Exception as e:
            detail = f"YOLO predict error: {e}"
            self._log(f"[Vision] {detail}")
            return "vision_unavailable", 0.0, detail

        # Step 5 — confidence gate
        if confidence < VISION_CONFIDENCE_THR:
            detail = (
                f"AWW — unsure: label={label}, "
                f"confidence={confidence:.2f}, ink_ratio={ink_ratio:.4f}"
            )
            self._log(f"[Vision] {detail}")
            return "low_confidence", confidence, detail

        if label == "not_letter":
            detail = f"AWW — not a letter (confidence={confidence:.2f}, ink_ratio={ink_ratio:.4f})"
            self._log(f"[Vision] {detail}")
            return "not_letter", confidence, detail

        # Greek letter found
        detail = (
            f"GREAT — Greek letter found: {label}, "
            f"confidence={confidence:.2f}, ink_ratio={ink_ratio:.4f}"
        )
        self._log(f"[Vision] {detail}")
        return label, confidence, detail

    # ══════════════════════════════════════════════════════════════════════════
    # ROS callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def odom_callback(self, msg: Odometry) -> None:
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny, cosy)
        if not self.odom_ready:
            self.odom_ready = True
            self._log(
                f"Odom ready: ({self.odom_x:.2f}, {self.odom_y:.2f})  "
                f"yaw={math.degrees(self.odom_yaw):.1f}deg"
            )

    def lidar_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        if not self.lidar_ready:
            self.lidar_ready = True
            self._log(
                f"LiDAR ready: {len(msg.ranges)} rays  "
                f"[{msg.range_min:.2f}, {msg.range_max:.2f}] m"
            )

        # Continuously update safety brake flag
        half = math.radians(SAFETY_CONE_DEG / 2.0)
        hit  = False
        for i, r in enumerate(msg.ranges):
            a = msg.angle_min + i * msg.angle_increment
            if abs(a) > half:
                continue
            if not math.isfinite(r):
                continue
            if r < CHASSIS_MIN_M:
                continue
            if r < SAFETY_STOP_M:
                hit = True
                break
        self.safety_obstacle = hit

    def joy_callback(self, msg: Joy) -> None:
        self.last_joy = msg

        if self._btn_down(msg, self.btn_circle):
            with self._lock:
                self.mode                = self.MODE_MANUAL
                self.auto_phase          = self.PHASE_IDLE
                self.auto_thread_started = False
            self._stop()
            m = String(); m.data = self.MODE_MANUAL
            self.mode_pub.publish(m)
            self._log("-> MANUAL (Circle)")

        if self._btn_down(msg, self.btn_x):
            with self._lock:
                if self.auto_phase == self.PHASE_RUNNING:
                    self._log("AUTO already running — ignoring X")
                    return
                self.mode                = self.MODE_AUTO
                self.auto_phase          = self.PHASE_WAITING
                self.auto_start_time     = time.time()
                self.auto_thread_started = False
            m = String(); m.data = self.MODE_AUTO
            self.mode_pub.publish(m)
            self._log(f"-> AUTO in {STARTUP_DELAY_S:.0f} s...")

    # ══════════════════════════════════════════════════════════════════════════
    # Control loop (20 Hz)
    # ══════════════════════════════════════════════════════════════════════════

    def control_loop(self) -> None:
        if self.last_joy is None:
            self._send_cmd()
            return

        joy = self.last_joy

        if self.mode == self.MODE_MANUAL:
            if not self._axis_pressed(joy, self.axis_l2):
                self._send_cmd()
                return
            lin = self._apply_deadzone(
                joy.axes[self.axis_right_y] if self.axis_right_y < len(joy.axes) else 0.0)
            ang = self._apply_deadzone(
                joy.axes[self.axis_left_x]  if self.axis_left_x  < len(joy.axes) else 0.0)
            self._send_cmd(MANUAL_LINEAR_SPEED * lin, MANUAL_ANGULAR_SPEED * ang)
            return

        if self.mode == self.MODE_AUTO:
            with self._lock:
                phase   = self.auto_phase
                started = self.auto_thread_started

            if phase == self.PHASE_WAITING:
                if (time.time() - self.auto_start_time) < STARTUP_DELAY_S:
                    return
                if not self.odom_ready or not self.lidar_ready:
                    self._log("Waiting for odom/lidar...")
                    return
                if started:
                    return

                self._set_mission_home()
                self.visited_positions.clear()
                self.photo_count = 0

                with self._lock:
                    self.auto_phase          = self.PHASE_RUNNING
                    self.auto_thread_started = True

                threading.Thread(target=self._run_mission, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # Mission thread
    # ══════════════════════════════════════════════════════════════════════════

    def _run_mission(self) -> None:
        self._log("== ROTATE-AND-VISIT MISSION START ==")
        self._log(
            f"  Field: {AREA_SIZE}x{AREA_SIZE} m | "
            f"Max range: {MAX_DETECT_RANGE:.1f} m | "
            f"Cone: +-{DETECT_CONE_DEG/2:.0f}deg"
        )

        accum_rad   = 0.0
        prev_yaw    = self.odom_yaw
        visit_count = 0

        while accum_rad < 2.0 * math.pi:
            with self._lock:
                if self.mode == self.MODE_MANUAL:
                    self._stop()
                    self._log("Mission aborted — MANUAL")
                    return

            detection = self._object_directly_ahead()

            if detection is not None:
                rng, wx, wy = detection

                if self._already_visited(wx, wy):
                    # Known — keep rotating but don't add to accumulator
                    self._send_cmd(0.0, SCAN_TURN_SPEED)
                    time.sleep(CONTROL_DT)
                else:
                    self._stop()
                    visit_count += 1
                    self._log(
                        f"Object #{visit_count} detected | "
                        f"range={rng:.2f} m | "
                        f"heading={math.degrees(self.odom_yaw):.1f}deg"
                    )
                    self.visited_positions.append((wx, wy))

                    ok = self._visit_object(rng, wx, wy, visit_count)
                    if not ok:
                        return

                    prev_yaw = self.odom_yaw   # reset delta after excursion
                    time.sleep(0.3)
                    continue

            # No detection — rotate and accumulate
            self._send_cmd(0.0, SCAN_TURN_SPEED)
            time.sleep(CONTROL_DT)

            curr_yaw  = self.odom_yaw
            delta     = abs(self._norm_angle(curr_yaw - prev_yaw))
            accum_rad += delta
            prev_yaw  = curr_yaw

        self._stop()
        self._log(
            f"== 360 COMPLETE == {visit_count} object(s) visited | "
            f"Photos -> {FOR_ML_DIR}"
        )

        with self._lock:
            self.auto_phase = self.PHASE_DONE

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._save_slam_map(f"bin_hunt_{ts}")

    # ══════════════════════════════════════════════════════════════════════════
    # Detection helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _object_directly_ahead(self) -> Optional[Tuple[float, float, float]]:
        """
        Find nearest LiDAR return inside the narrow forward cone that also
        falls within the square field boundary.
        Returns (range_m, world_x, world_y) or None.
        """
        scan = self.latest_scan
        if scan is None:
            return None

        half_cone = math.radians(DETECT_CONE_DEG / 2.0)
        best_r    = float("inf")
        best_wx   = 0.0
        best_wy   = 0.0

        for i, r in enumerate(scan.ranges):
            a = scan.angle_min + i * scan.angle_increment
            if abs(a) > half_cone:
                continue
            if not math.isfinite(r):
                continue
            if r < CHASSIS_MIN_M or r > MAX_DETECT_RANGE:
                continue
            if r >= best_r:
                continue

            world_bearing = self._norm_angle(self.odom_yaw + a)
            wx = self.odom_x + r * math.cos(world_bearing)
            wy = self.odom_y + r * math.sin(world_bearing)
            fx, fy = self._world_to_field(wx, wy)

            if abs(fx) > FIELD_HALF or abs(fy) > FIELD_HALF:
                continue

            best_r  = r
            best_wx = wx
            best_wy = wy

        return None if best_r == float("inf") else (best_r, best_wx, best_wy)

    def _already_visited(self, wx: float, wy: float) -> bool:
        return any(
            math.hypot(wx - vx, wy - vy) <= DEDUP_RADIUS_M
            for vx, vy in self.visited_positions
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Visit one object
    # ══════════════════════════════════════════════════════════════════════════

    def _visit_object(self, detected_range: float,
                      wx: float, wy: float,
                      visit_id: int) -> bool:
        """
        1. Drive to standoff.
        2. Capture one frame → save to disk + run Greek-letter vision.
        3. Log full result to CSV (position + letter + confidence).
        4. Reverse to home.
        Returns True on success, False on MANUAL abort.
        """
        approach_dist = max(0.0, detected_range - APPROACH_STANDOFF_M)

        # ── 1. Approach ───────────────────────────────────────────────────────
        if approach_dist > 0.05:
            self._log(f"  Approach: {approach_dist:.2f} m")
            if not self._drive(approach_dist, forward=True):
                return False
        else:
            self._log("  Already within standoff — skipping drive")

        # ── 2. Capture frame + detect Greek letter ─────────────────────────────
        self._log("  Capturing photo and running Greek-letter detection...")
        photo_name, frame = self._capture_frame_and_save(visit_id)

        # Run vision on the same in-memory frame (no disk round-trip needed)
        greek_letter, letter_conf, vision_detail = self._detect_greek_letter(frame)

        # Terminal output that matches your teammate's format
        self._log(
            f"[Vision Result] #{visit_id}: {vision_detail}"
        )

        # ── 3. CSV log ────────────────────────────────────────────────────────
        fx, fy = self._world_to_field(wx, wy)
        heading_deg = math.degrees(self.odom_yaw)
        try:
            with open(BIN_CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    visit_id,
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    f"{wx:.4f}", f"{wy:.4f}",
                    f"{fx:.4f}", f"{fy:.4f}",
                    f"{detected_range:.3f}",
                    f"{heading_deg:.1f}",
                    photo_name,
                    greek_letter,
                    f"{letter_conf:.3f}",
                    vision_detail,
                ])
        except Exception as e:
            self._log(f"WARNING: CSV write failed: {e}")

        self._log(
            f"  Logged #{visit_id}: "
            f"field=({fx:.2f},{fy:.2f})  "
            f"letter={greek_letter} ({letter_conf:.0%})  "
            f"photo={photo_name or 'NONE'}"
        )

        # ── 4. Reverse to home ────────────────────────────────────────────────
        actual_moved = math.hypot(
            self.odom_x - self.home_x, self.odom_y - self.home_y
        )
        if actual_moved > 0.05:
            self._log(f"  Reversing {actual_moved:.2f} m to home")
            if not self._drive(actual_moved, forward=False):
                return False

        self._log(f"  Visit #{visit_id} complete — letter: {greek_letter}")
        return True

    # ══════════════════════════════════════════════════════════════════════════
    # Camera
    # ══════════════════════════════════════════════════════════════════════════

    def _get_frame(self) -> Optional[np.ndarray]:
        if not self.camera_ready or self.q_rgb is None:
            return None
        try:
            pkt = self.q_rgb.tryGet()
            return pkt.getCvFrame() if pkt else None
        except Exception:
            return None

    def _capture_frame_and_save(self, visit_id: int) -> Tuple[str, Optional[np.ndarray]]:
        """
        Grab a frame from the OAK-D camera.
        Saves it to for_ml/ and publishes it to Foxglove.
        Returns (filename, frame_ndarray).
        frame may be None if camera unavailable — vision will handle gracefully.
        """
        with self._camera_lock:
            if not self.camera_ready:
                self._log("Camera not ready — skipping photo")
                return "", None

            frame    = None
            deadline = time.time() + 1.5
            while time.time() < deadline:
                frame = self._get_frame()
                if frame is not None:
                    break
                time.sleep(0.05)

            if frame is None:
                self._log("No camera frame available")
                return "", None

            # Publish to Foxglove
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.image_pub.publish(img_msg)

            # Save to disk
            self.photo_count += 1
            ts   = time.strftime("%Y%m%d_%H%M%S")
            name = f"bin_{visit_id:03d}_{ts}.jpg"
            path = os.path.join(FOR_ML_DIR, name)

            try:
                if cv2.imwrite(path, frame):
                    self._log(f"  Photo -> {path}")
                else:
                    self._log(f"  WARNING: imwrite failed for {path}")
                    name = ""
            except Exception as e:
                self._log(f"  Photo save error: {e}")
                name = ""

            # Return the in-memory frame alongside the filename
            # The vision pipeline will use this frame directly —
            # no need to reload from disk.
            return name, frame

    # ══════════════════════════════════════════════════════════════════════════
    # Motion primitives
    # ══════════════════════════════════════════════════════════════════════════

    def _drive(self, dist_m: float, forward: bool) -> bool:
        """
        Drive forward (forward=True) or reverse for dist_m metres.
        Heading-hold correction applied throughout.
        Safety brake on forward only.
        Returns False if switched to MANUAL.
        """
        if dist_m <= 0.01:
            return True

        start_x  = self.odom_x
        start_y  = self.odom_y
        hold_yaw = self.odom_yaw
        sign     = 1.0 if forward else -1.0

        while True:
            with self._lock:
                if self.mode == self.MODE_MANUAL:
                    self._stop()
                    self._log("Drive aborted — MANUAL")
                    return False

            travelled = math.hypot(
                self.odom_x - start_x, self.odom_y - start_y
            )

            if travelled >= dist_m - 0.01:
                self._stop()
                return True

            if forward and self.safety_obstacle:
                self._stop()
                self._log("Safety stop — obstacle too close")
                return True   # photo already taken; return as success

            # Correction sign flips when reversing
            yaw_err = self._norm_angle(hold_yaw - self.odom_yaw)
            ang     = max(-0.6, min(0.6, (2.0 if forward else -2.0) * yaw_err))

            self._send_cmd(sign * AUTO_SPEED, ang)
            time.sleep(CONTROL_DT)

    # ══════════════════════════════════════════════════════════════════════════
    # Map save
    # ══════════════════════════════════════════════════════════════════════════

    def _save_slam_map(self, stem: str) -> None:
        if not SLAM_TOOLBOX_AVAILABLE or self.save_map_client is None:
            self._log("slam_toolbox not available — map save skipped")
            return
        if not self.save_map_client.wait_for_service(timeout_sec=4.0):
            self._log("WARNING: /slam_toolbox/save_map not responding")
            return

        req      = SaveMap.Request()
        map_stem = os.path.join(MAP_DIR, stem)
        try:
            req.name.data = map_stem
        except AttributeError:
            req.name = map_stem

        future = self.save_map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if future.done() and future.result() is not None:
            self._log(f"Map saved: {map_stem}.pgm / .yaml")
        else:
            self._log("WARNING: map save timed out")

    # ══════════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════════

    def _set_mission_home(self) -> None:
        self.home_x   = self.odom_x
        self.home_y   = self.odom_y
        self.home_yaw = self.odom_yaw
        self._log(
            f"Home set: ({self.home_x:.2f}, {self.home_y:.2f})  "
            f"yaw={math.degrees(self.home_yaw):.1f}deg"
        )

    def _world_to_field(self, wx: float, wy: float) -> Tuple[float, float]:
        dx = wx - self.home_x
        dy = wy - self.home_y
        c  = math.cos(self.home_yaw)
        s  = math.sin(self.home_yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _log(self, msg: str) -> None:
        self.get_logger().info(msg)
        s = String(); s.data = msg
        self.status_pub.publish(s)

    def _send_cmd(self, lin: float = 0.0, ang: float = 0.0) -> None:
        t = Twist()
        t.linear.x  = float(lin)
        t.angular.z = float(ang)
        self.cmd_pub.publish(t)

    def _stop(self) -> None:
        self._send_cmd(0.0, 0.0)

    @staticmethod
    def _norm_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _apply_deadzone(self, v: float) -> float:
        return 0.0 if abs(v) < self.deadzone else v

    def _axis_pressed(self, joy: Joy, idx: int, threshold: float = 0.5) -> bool:
        return len(joy.axes) > idx and joy.axes[idx] < threshold

    def _btn_down(self, joy: Joy, idx: int) -> bool:
        return len(joy.buttons) > idx and joy.buttons[idx] == 1

    # ══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ══════════════════════════════════════════════════════════════════════════

    def destroy_node(self) -> None:
        self._stop()
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception as e:
                self.get_logger().warn(f"Camera stop error: {e}")
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = DualShockBinHunt()
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
