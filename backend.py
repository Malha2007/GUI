#!/usr/bin/env python3
"""Prana Rover dashboard backend with ROS 2 telemetry and RealSense image streaming.

Run in a ROS 2 Humble-sourced terminal:
    source /opt/ros/humble/setup.bash
    python3 backend_updated.py

Dependencies (ROS packages should be installed through apt):
    sudo apt install ros-humble-cv-bridge
    python3 -m pip install fastapi uvicorn opencv-python

Optional environment variables:
    REALSENSE_TOPIC=/camera/camera/color/image_raw
    ZED_INDEX=0
    CAMERA_FPS=15
"""

import asyncio
import json
import logging
import os
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, Image, NavSatFix
from std_msgs.msg import Bool, Float32MultiArray, String
import uvicorn

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("rover-backend")

REALSENSE_TOPIC = os.getenv("REALSENSE_TOPIC", "/camera/camera/color/image_raw")
ZED_INDEX = int(os.getenv("ZED_INDEX", "0"))
CAMERA_FPS = max(1, int(os.getenv("CAMERA_FPS", "15")))
JPEG_QUALITY = max(30, min(95, int(os.getenv("JPEG_QUALITY", "80"))))

state_lock = threading.Lock()
image_lock = threading.Lock()
latest_realsense_frame = None
last_realsense_error = "Waiting for RealSense image topic"


class RoverNode(Node):
    def __init__(self):
        super().__init__("rover_backend")
        self.bridge = CvBridge()
        self.state = {
            "battery": {"voltage": None, "percent": None, "current": None},
            "speed": None,
            "power": None,
            "autonomy": None,
            "gps": {"lat": None, "lon": None, "alt": None},
            "wheels": {"fl": None, "fr": None, "rl": None, "rr": None},
            "abex": {
                "soil_temp": None, "atm_temp": None, "soil_hum": None,
                "atm_hum": None, "soil_ph": None, "atm_ph": None,
                "soil_pres": None, "atm_pres": None
            },
            "object": None,
            "panorama": None,
        }

        # Publishers used by the dashboard command buttons.
        self.kill_pub = self.create_publisher(Bool, "/rover/kill_switch", 10)
        self.reloc_pub = self.create_publisher(Bool, "/autonomy/relocalize", 10)
        self.clear_pub = self.create_publisher(Bool, "/autonomy/clear_costmap", 10)

        # Telemetry subscriptions. Confirm these names/types match the rover publishers.
        self.create_subscription(BatteryState, "/battery/state", self.battery_cb, 10)
        self.create_subscription(NavSatFix, "/gps/fix", self.gps_cb, 10)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(Float32MultiArray, "/wheel_states", self.wheel_cb, 10)
        self.create_subscription(Float32MultiArray, "/abex/data", self.abex_cb, 10)
        self.create_subscription(String, "/rover/autonomy_status", self.autonomy_cb, 10)

        # RealSense color images arrive through ROS 2, not OpenCV camera index 1.
        self.create_subscription(
            Image, REALSENSE_TOPIC, self.realsense_image_cb, qos_profile_sensor_data
        )
        self.get_logger().info(f"Subscribed to RealSense topic: {REALSENSE_TOPIC}")

    def battery_cb(self, msg):
        percent = None
        try:
            if msg.percentage >= 0:
                percent = round(float(msg.percentage) * 100, 1)
        except (TypeError, ValueError):
            pass
        with state_lock:
            voltage = float(msg.voltage)
            current = float(msg.current)
            if not (voltage == voltage and abs(voltage) != float("inf")):
                voltage = None
            if not (current == current and abs(current) != float("inf")):
                current = None
            self.state["battery"] = {
                "voltage": round(voltage, 2) if voltage is not None else None,
                "percent": percent,
                "current": round(current, 2) if current is not None else None,
            }
            self.state["power"] = (
                round(abs(voltage * current), 1)
                if voltage is not None and current is not None else None
            )

    def gps_cb(self, msg):
        with state_lock:
            self.state["gps"] = {
                "lat": float(msg.latitude),
                "lon": float(msg.longitude),
                "alt": float(msg.altitude),
            }

    def odom_cb(self, msg):
        v = msg.twist.twist.linear
        speed = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
        with state_lock:
            self.state["speed"] = round(speed, 3)

    def wheel_cb(self, msg):
        if len(msg.data) < 4:
            self.get_logger().warning("/wheel_states needs at least 4 values")
            return
        names = ("fl", "fr", "rl", "rr")
        with state_lock:
            for i, name in enumerate(names):
                self.state["wheels"][name] = float(msg.data[i])

    def abex_cb(self, msg):
        if len(msg.data) < 3:
            self.get_logger().warning("/abex/data needs at least 3 values")
            return
        with state_lock:
            self.state["abex"].update({
                "soil_temp": float(msg.data[0]),
                "atm_temp": float(msg.data[1]),
                "soil_ph": float(msg.data[2]),
            })

    def autonomy_cb(self, msg):
        value = msg.data.strip().upper()
        if value:
            with state_lock:
                self.state["autonomy"] = value

    def realsense_image_cb(self, msg):
        global latest_realsense_frame, last_realsense_error
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with image_lock:
                latest_realsense_frame = frame.copy()
                last_realsense_error = ""
        except Exception as exc:
            with image_lock:
                last_realsense_error = f"Image conversion failed: {exc}"
            self.get_logger().error(f"RealSense image conversion failed: {exc}")


# Initialize ROS 2 and start spinning in the background.
if not rclpy.ok():
    rclpy.init(args=None)
node = RoverNode()
ros_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True, name="ros-spin")
ros_thread.start()

app = FastAPI(title="Prana Rover Dashboard Backend", version="1.1.0")


def mjpeg_response(frame_generator):
    return StreamingResponse(
        frame_generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache"},
    )


def zed_generator():
    """Read the ZED feed through OpenCV. Change ZED_INDEX if needed."""
    cap = None
    try:
        while True:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(ZED_INDEX)
                if not cap.isOpened():
                    log.warning("Could not open ZED source index %s", ZED_INDEX)
                    cap.release()
                    cap = None
                    time.sleep(1.0)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue
            encoded_ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if encoded_ok:
                jpeg = encoded.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(1.0 / CAMERA_FPS)
    finally:
        if cap is not None:
            cap.release()


def realsense_generator():
    """Stream the latest image received from the RealSense ROS 2 topic."""
    while True:
        with image_lock:
            frame = None if latest_realsense_frame is None else latest_realsense_frame.copy()
        if frame is not None:
            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )
            if ok:
                jpeg = encoded.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1.0 / CAMERA_FPS)


@app.get("/")
def root():
    return {
        "name": "Prana Rover Dashboard Backend",
        "status": "ok",
        "endpoints": ["/health", "/ws", "/camera/zed", "/camera/realsense", "/kill", "/relocalize", "/clear_costmap"],
    }


@app.get("/health")
def health():
    with image_lock:
        has_realsense_frame = latest_realsense_frame is not None
        realsense_error = last_realsense_error or None
    return {
        "ok": True,
        "ros2_ok": rclpy.ok(),
        "realsense_topic": REALSENSE_TOPIC,
        "realsense_has_frame": has_realsense_frame,
        "realsense_error": realsense_error,
        "zed_index": ZED_INDEX,
    }


@app.post("/kill")
def kill():
    if not rclpy.ok():
        raise HTTPException(status_code=503, detail="ROS 2 is not available")
    node.kill_pub.publish(Bool(data=True))
    return {"status": "KILL SENT"}


@app.post("/relocalize")
def relocalize():
    if not rclpy.ok():
        raise HTTPException(status_code=503, detail="ROS 2 is not available")
    node.reloc_pub.publish(Bool(data=True))
    return {"status": "RELOCALIZE SENT"}


@app.post("/clear_costmap")
def clear_costmap():
    if not rclpy.ok():
        raise HTTPException(status_code=503, detail="ROS 2 is not available")
    node.clear_pub.publish(Bool(data=True))
    return {"status": "COSTMAP CLEARED"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with state_lock:
                packet = json.loads(json.dumps(node.state))
            await ws.send_text(json.dumps(packet))
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.debug("WebSocket connection ended: %s", exc)


@app.get("/camera/zed")
def zed_cam():
    return mjpeg_response(zed_generator())


@app.get("/camera/realsense")
def realsense_cam():
    return mjpeg_response(realsense_generator())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", "8000")))
