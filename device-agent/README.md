# Device Agent (Capstone)

This local agent exposes your existing traffic pipeline to the Nuxt dashboard.

API contract (matches dashboard):
- GET /status
- POST /start
- POST /stop
- GET /video_feed

## 1) Install dependencies

From the Capstone root:

pip install -r requirements.txt

## 2) Configure runtime (optional)

You can copy `.env.example` from the project root to `.env` and edit values there. The agent now loads the root `.env` automatically.

Default source is `traffic_bi.mp4` if it exists, otherwise `traffic.mp4`.

PowerShell example:

$env:DEVICE_NAME = "Home PC"
$env:PORT = "8001"
$env:VIDEO_SOURCE = "traffic_bi.mp4"
$env:USE_QUANTUM = "true"
$env:YOLO_DEVICE = "cuda"

To switch to the second demo clip:

$env:VIDEO_SOURCE = "traffic.mp4"
$env:LOOP_VIDEO = "true"

## 3) Run the agent

python device-agent/agent.py

Then set your Nuxt dashboard env:

PC_AGENT_URL=http://<your-pc-ip>:8001

If presenting from another machine, ensure both machines are on the same network and allow inbound firewall access for port 8001.
