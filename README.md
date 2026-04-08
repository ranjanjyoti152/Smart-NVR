# SmartNVR

SmartNVR is a modern AI-powered Network Video Recorder (NVR) built with Flask, MongoDB, OpenCV, and Ultralytics models.

It combines live camera monitoring, AI detection, timeline playback, face intelligence, ROI-based alerting, and system monitoring in a single web application.

## What's New (April 2026)

- 2026 UI refresh across core pages with responsive layout improvements.
- Dashboard, Monitor, and Playback visual upgrades with cleaner controls and chart styling.
- New **Use Cases** page (`/usecase`) with charts and workflow guidance for end users.
- Settings page responsiveness improvements for normal browser zoom and laptop resolutions.
- Face gallery performance upgrades:
  - paginated face listing API
  - lighter list queries with projection
  - lazy image loading and faster image route behavior
- Notification emails redesigned to match the modern UI language.
- Login and Register pages tuned for better scaling across desktop zoom levels.

## Core Capabilities

### Realtime Operations
- Live multi-camera dashboard with detection overlays.
- Quick stream quality selection and fullscreen controls.
- Camera and model status visibility.

### AI Detection and ROI
- Object detection with configurable confidence thresholds.
- ROI creation and class filtering.
- Time-based ROI schedules (days and active windows).
- Optional Gemini-generated human-friendly notification summaries.

### Playback and Investigation
- Timeline-based playback with event context.
- Event filtering and camera/date scoping.
- Snapshot and clip review workflows.

### Face Intelligence
- UniFace-based face detection pipeline.
- Face profile gallery for naming and management.
- Face recognition/grouping with configurable thresholds.
- Optional auto-assimilation behavior for known identities.

### Monitoring and Administration
- CPU, memory, disk, and GPU visibility in System Monitor.
- User management and model management for admins.
- Profile and application-level settings panels.

## Product Navigation

After login, key sections are available in the sidebar:

- `Dashboard`
- `Playback`
- `Cameras`
- `Faces`
- `Settings`
- `System Monitor`
- `Use Cases` (new)

Admin users also see:

- `User Management`
- `AI Models`

## Tech Stack

- **Backend**: Flask, Flask-Login
- **Database**: MongoDB (`flask-pymongo`, `pymongo`)
- **Video and CV**: OpenCV, NumPy
- **Detection**: Ultralytics (`ultralytics` package)
- **Face Detection**: UniFace (ONNX Runtime providers)
- **Face Recognition**: facenet-pytorch
- **Frontend**: Jinja templates, Bootstrap 5, Chart.js, custom CSS/JS

## Requirements

- Python 3.8+ (3.10+ recommended)
- MongoDB 4.4+
- RTSP/IP camera sources
- 8 GB RAM minimum (16 GB+ recommended for multi-camera workloads)
- NVIDIA GPU recommended for realtime inference

## Quick Start

### 1) Clone and enter project

```bash
git clone https://github.com/ranjanjyoti152/Smart-NVR.git
cd Smart-NVR
```

### 2) Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Start MongoDB

Example (Linux systemd):

```bash
sudo systemctl start mongod
sudo systemctl status mongod
```

### 5) Initialize database

```bash
python initialize_db.py
```

Default admin account created by initializer:

- Username: `admin`
- Password: `admin`

### 6) Run application

```bash
python run.py
```

Open:

- `http://localhost:8000`

## Configuration

Primary runtime configuration is in:

- `config/settings.json`

General app config class:

- `config.py`

Useful environment variables:

- `SECRET_KEY`
- `MONGO_URI`
- `SMARTNVR_GEMINI_API_KEY`
- `SMARTNVR_GEMINI_MODEL`
- `API_KEY` (internal API communication)
- `PUBLIC_BASE_URL` (optional, recommended for email playback links)

## Recommended First-Time Workflow

1. Login as admin and open `Cameras`.
2. Add at least one RTSP camera and enable detection.
3. Configure ROIs (and schedule if needed).
4. Open `Dashboard` to confirm live inference.
5. Open `Playback` and validate timeline/event review.
6. Enable email and optional Gemini enhancement in `Settings`.
7. Visit `Use Cases` page for team onboarding and deployment patterns.

## Notification System

SmartNVR sends ROI-triggered email notifications with:

- detection metadata
- optional attached snapshot
- optional playback deep link
- optional Gemini summary

The email templates were modernized in the latest update to align with the UI style system.

## Face Gallery Performance Notes

Recent optimization updates include:

- paginated `/api/faces` responses
- lighter face list projection (reduced payload)
- faster face image fetch path
- lazy-loaded gallery thumbnails

These changes improve large-library responsiveness significantly.

## API Notes

The app exposes API routes under `/api` for camera, detection, playback, ROI, face, and configuration flows.

Examples:

- `/api/cameras`
- `/api/cameras/<camera_id>/frame`
- `/api/faces`
- `/api/test_email`
- `/api/test_gemini`

## Troubleshooting

- Check logs in `logs/` when diagnosing startup or runtime issues.
- Verify MongoDB connectivity before launching the app.
- For camera issues, validate RTSP reachability from host machine.
- For performance issues:
  - reduce stream resolution/fps
  - tune model size and confidence
  - use focused ROIs instead of full-frame detection
- For face or alert delays, verify settings toggles and camera processor status.

## Contributing

Contributions are welcome.

Basic flow:

1. Fork repository
2. Create feature branch
3. Commit focused changes
4. Push branch
5. Open pull request

## Acknowledgments

- Flask
- MongoDB
- OpenCV
- Ultralytics
- UniFace
- facenet-pytorch
- Chart.js
