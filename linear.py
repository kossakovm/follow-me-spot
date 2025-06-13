#!/usr/bin/env python3
"""
Main follow-me script for Spot with smooth velocity ramp and live visualization.

Uses modular components:
  - human_detection.load_model, detect_humans
  - human_tracking.init_tracker, update_tracks
  - depth_estimation.estimate_distance_bbox

Velocity is 0 m/s at or below MIN_DIST (0.5m) and at or above MAX_DIST (5.0m),
and ramps linearly between those distances up to MAX_LIN_SPEED.
Shows a pop-up with the arm camera view, bounding box, and overlayed logs.
"""
import sys
import time
import numpy as np
import cv2
import bosdyn.client
from bosdyn.client.util import setup_logging
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

# Modular detection/tracking/depth
from human_detection import load_model, detect_humans
from human_tracking import init_tracker, update_tracks
from depth_estimation import estimate_distance_bbox

# === Spot Connection Details ===
SPOT_IP     = "192.168.80.3"
USERNAME    = "username"
PASSWORD    = "password"

# === Follow Parameters ===
MIN_DIST         = 0.5   # [m] at or below this → 0 m/s
MAX_DIST         = 5.0   # [m] at or above this → 0 m/s
MAX_LIN_SPEED    = 1.0   # [m/s] peak at MAX_DIST
K_YAW            = 0.5   # yaw gain [rad/s per unit error]
MAX_YAW_RATE     = 0.5   # [rad/s]
YAW_DEADBAND     = 0.05  # [rad]
CONF_THRESH      = 0.3   # detection confidence threshold

# Initialize SDK & Clients
sdk = bosdyn.client.create_standard_sdk('SpotFollowSmoothViz')
setup_logging()
robot = sdk.create_robot(SPOT_IP)
robot.authenticate(USERNAME, PASSWORD)
if robot.is_estopped():
    print("❌ Spot is estopped; clear E-Stop and retry.")
    sys.exit(1)

# Lease & sync
device = robot.ensure_client(LeaseClient.default_service_name)
lease = LeaseKeepAlive(device, must_acquire=True, return_at_exit=True)
robot.time_sync.wait_for_sync()

# Power on & stand
if not robot.is_powered_on():
    robot.power_on()
robot_state_client   = robot.ensure_client(RobotStateClient.default_service_name)
robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
stand_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
    goal_x=0.0, goal_y=0.0, goal_heading=0.0,
    frame_name=bosdyn.client.frame_helpers.ODOM_FRAME_NAME,
    params=RobotCommandBuilder.mobility_params(stair_hint=False)
)
robot_command_client.robot_command(stand_cmd)
print("✅ Spot standing. Starting smooth follow loop with visualization.")

# Image client
image_client = robot.ensure_client(ImageClient.default_service_name)

def get_frame():
    """Capture BGR frame from hand camera."""
    req = [image_pb2.ImageRequest(
        image_source_name="hand_color_image",
        pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
        quality_percent=80)]
    resp = image_client.get_image(req)
    if not resp or not resp[0].shot.image:
        return None
    buf = np.frombuffer(resp[0].shot.image.data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

# Load detection model
model = load_model()
# Initialize tracker
tracker = init_tracker()
lock_id = None
# Warm-up detection model
_dummy = np.zeros((480,640,3), dtype=np.uint8)
_ = model.predict(_dummy, imgsz=(480,640), conf=CONF_THRESH, verbose=False)

# Visualization window
WINDOW_NAME = "Arm Camera View"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

# Main loop
try:
    while True:
        frame = get_frame()
        if frame is None:
            print("⚠️ Frame unavailable; retrying...")
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]
        # Detection
        detections = detect_humans(model, frame, imgsz=640, conf_threshold=CONF_THRESH)
        # Prepare detections for tracking
        raw_dets = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            raw_dets.append(([float(x1), float(y1), float(x2-x1), float(y2-y1)],
                             det['confidence'], 'person'))

        # Tracking
        tracks = update_tracks(tracker, raw_dets, frame)
        # Lock onto first confirmed track
        if lock_id is None and tracks:
            lock_id = tracks[0]['track_id']
            print(f"🔒 Locked onto track ID {lock_id}")

        # Find locked track
        target = None
        for tr in tracks:
            if tr['track_id'] == lock_id:
                target = tr
                break

        # Default velocities
        v_x = 0.0
        yaw_rate = 0.0
        dist = None

        vis_frame = frame.copy()
        if target:
            x1, y1, x2, y2 = target['bbox']
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Yaw control
            cx = (x1 + x2) / 2.0
            err_norm = (cx - w/2.0) / (w/2.0)
            yaw_rate = -K_YAW * err_norm
            yaw_rate = float(np.clip(yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))
            if abs(yaw_rate) < YAW_DEADBAND:
                yaw_rate = 0.0
            # Distance estimation
            px_width = x2 - x1
            try:
                dist = estimate_distance_bbox(px_width)
            except ValueError:
                dist = MIN_DIST
            # Smooth ramp speed
            if dist <= MIN_DIST or dist >= MAX_DIST:
                v_x = 0.0
            else:
                v_x = MAX_LIN_SPEED * (dist - MIN_DIST) / (MAX_DIST - MIN_DIST)
            # Overlay text
            text = f"ID:{lock_id} Dist:{dist:.2f}m Vx:{v_x:.2f} Yaw:{yaw_rate:.2f}"
            cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis_frame, "No target detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        # Show visualization
        cv2.imshow(WINDOW_NAME, vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Send velocity command
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x, 0.0, yaw_rate)
        robot_command_client.robot_command(cmd, end_time_secs=time.time()+0.2)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("🛑 Exiting.")

finally:
    cv2.destroyAllWindows()
