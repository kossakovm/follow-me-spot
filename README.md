# follow-me-spot
# Autonomous Human-Following System Using Camera-Based Detection Methods on Boston Dynamics Spot Robot

This repository contains the implementation of an autonomous human-following system for the Boston Dynamics Spot robot using camera-based detection. The system integrates YOLO for human detection, DeepSORT for tracking, and two movement strategies (linear and exponential velocity control) for smooth autonomous following.

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `__init__.py` | Initialization file for the Python package. |
| `depth_estimation.py` | Estimates distance to the detected human using bounding box logic. |
| `human_detection.py` | Detects humans using YOLO (version should be specified in code). |
| `human_tracking.py` | Tracks the detected human using DeepSORT. Ensure the DeepSORT model is downloaded before connecting to the robot. |
| `linear.py` | Implements velocity-based following logic. You must configure Spot’s IP address, username, and password. |
| `exponential.py` | Enhanced version of `linear.py` using exponential (sqrt-shaped) speed control with a GAMMA parameter. Provides smoother acceleration in mid-range distances. |

> ⚠️ `__pycache__` directories will be created automatically after running scripts.

---

## Features

- **Boston Dynamics Spot SDK** https://github.com/boston-dynamics/spot-sdk
- **YOLO-based Human Detection** (YOLOv5/YOLOv8 nano and small versions are suitable) here is the link to official versions https://github.com/ultralytics/ultralytics  
- **DeepSORT-based Human Tracking**  here is the link to download https://github.com/ModelBunker/Deep-SORT-PyTorch.git
- **Depth Estimation** from bounding box  
- **Autonomous Movement** using linear and exponential velocity control  
- Supports real-time Spot robot control over Wi-Fi connection  

---

## 🔧 Requirements

Before connecting to Spot, ensure **all dependencies are installed** locally since the robot will lose internet access once connected.

### Python Version

- Python **3.10+**

### Install Dependencies

```bash
pip install numpy opencv-python spot-sdk

## 📩 Contact

For questions, feedback, or collaboration opportunities, feel free to:

- Open an [Issue](https://github.com/kossakovm/follow-me-spot/issues)
- Or email me directly at [marlen.kossakov@gmail.com](mailto:marlen.kossakov@gmail.com)

