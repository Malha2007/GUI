"""
Prana Rover backend for the dashboard (index.html).

Built from backend2.py. Same ROS topics and the same endpoints, extended so
every field and button in the GUI has something behind it:

  WebSocket  /ws                  telemetry JSON, 10 Hz
  GET        /camera/<id>         MJPEG: zed, realsense, ipcam1..ipcam4
  GET        /object/image        latest detected-object picture
  GET        /panorama/image      latest panorama picture
  POST       /kill  /relocalize  /clear_costmap
  GET        /                    serves index.html (+ logo / placeholder)

Everything marked "ASSUMED" below is a topic or message layout that
backend2.py did not have. Rename it in the CONFIG block to match your rover.
"""

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import rclpy
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, CompressedImage, NavSatFix
from std_msgs.msg import Bool, Float32MultiArray, String

# ---------------------------------------------------------------- CONFIG ----

HERE = Path(__file__).resolve().parent
GUI_FILE = HERE / "index.html"
GUI_ASSETS = {"logo.jpeg", "panorama_placeholder.jpg"}   # files the page links to

# Camera configuration.
# ZED and RealSense use their vendor SDKs (not generic webcam indices).
# IP camera URLs can be configured with IPCAM1_URL ... IPCAM4_URL.
def configured_camera_source(env_name, default=None):
    value = os.environ.get(env_name, default)
    if value is None:
        return None
    value = value.strip() if isinstance(value, str) else value
    if value == "":
        return None
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


CAMERA_SOURCES = {
    "zed": "zed",
    "realsense": "realsense",
    "ipcam1": configured_camera_source("IPCAM1_URL"),
    "ipcam2": configured_camera_source("IPCAM2_URL"),
    "ipcam3": configured_camera_source("IPCAM3_URL"),
    "ipcam4": configured_camera_source("IPCAM4_URL"),
}
JPEG_QUALITY = 80

# Topics already used in backend2.py
TOPIC_KILL = "/rover/kill_switch"
TOPIC_RELOC = "/autonomy/relocalize"
TOPIC_CLEAR = "/autonomy/clear_costmap"
TOPIC_BATTERY = "/battery/state"
TOPIC_GPS = "/gps/fix"
TOPIC_ODOM = "/odom"
TOPIC_WHEELS = "/wheel_states"
TOPIC_ABEX = "/abex/data"

# ASSUMED topics (not in backend2.py)
TOPIC_AUTONOMY = "/autonomy/status"            # std_msgs/String: "ON" | "OFF" | "PAUSED"
TOPIC_OBJECT_NAME = "/object/name"             # std_msgs/String
TOPIC_OBJECT_IMAGE = "/object/image/compressed"       # sensor_msgs/CompressedImage
TOPIC_PANO_IMAGE = "/panorama/image/compressed"       # sensor_msgs/CompressedImage
TOPIC_PANO_INFO = "/panorama/info"             # std_msgs/String, JSON:
#   {"direction": "North-East", "lat": 0.0, "lon": 0.0, "alt": 0.0, "accuracy": "± 1.5 m"}

# /abex/data layout. The first three match backend2.py; the rest are ASSUMED
# and are only filled when the array is long enough.
ABEX_ORDER = ["soil_temp", "atm_temp", "soil_ph",
              "soil_hum", "atm_hum", "atm_ph", "soil_pres", "atm_pres"]

# ------------------------------------------------------------- ROS NODE -----


class RoverNode(Node):
    def __init__(self):
        super().__init__("rover_backend")


