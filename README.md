# Smart-NVR-GPU

A powerful Network Video Recorder (NVR) application that leverages GPU acceleration for real-time AI object detection, smart recording, and efficient video management. Built with Python, Flask, and YOLOv5, this application provides enterprise-grade surveillance capabilities with a user-friendly interface.

![Smart-NVR-GPU Dashboard](static/img/dashboard-preview.png)

## Features

- **GPU-Accelerated AI Detection**: Real-time object detection using YOLOv5 ,v8,v9,v10 models with CUDA acceleration
- **Smart Recording Management**: Automatic recording based on motion or specific AI detection events
- **Live Camera Dashboard**: Monitor multiple RTSP/IP cameras simultaneously with object detection overlays
- **Regions of Interest (ROI)**: Define specific areas for detection to reduce false positives
- **Advanced Playback**: Timeline-based video playback with object detection markers and filtering
- **System Resource Monitoring**: Track CPU, RAM, GPU, and disk usage in real-time
- **Modern UI**: Clean, responsive interface designed for ease of use
- **Multi-User Support**: Role-based access with administrative and standard user accounts
- **Notifications**: Configurable alerts for specific object detections via email
- **API Access**: RESTful API for integration with other systems

## Requirements

- Python 3.8+ (3.10+ recommended)
- CUDA-compatible GPU (strongly recommended for real-time processing)
- NVIDIA drivers and CUDA toolkit (for GPU acceleration)
- RTSP/IP compatible cameras
- 8GB+ RAM (16GB+ recommended for multiple camera streams)
- Linux, Windows, or macOS (tested primarily on Linux)

## Installation

### Standard Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Smart-NVR-GPU.git
cd Smart-NVR-GPU
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
python initialize_db.py
```

5. Start the application:
```bash
python run.py
```

6. Access the web interface at http://localhost:8000

### Docker Installation

```bash
# Build the Docker image
docker build -t smart-nvr-gpu .

# Run the container with GPU support
docker run --gpus all -p 8000:8000 -v /path/to/storage:/app/storage smart-nvr-gpu
```

## Quick Start Guide

1. Login with the default credentials:
   - Username: `admin`
   - Password: `admin`

2. Navigate to Camera Management and add your first camera:
   - Provide an RTSP URL, camera name, and credentials
   - Select desired AI model and confidence threshold
   - Enable recording and detection as needed

3. Return to Dashboard to view your camera feeds with AI detection

4. Configure Regions of Interest (ROI) to focus detection on specific areas

5. Use the Recordings section to review detection events and continuous recordings

## Advanced Configuration

The application can be configured through the web interface or by editing these files:

- `config/settings.json`: Main application settings
- `app/utils/camera_processor.py`: Camera processing and AI detection parameters
- Environment variables:
  - `SMARTNVR_SECRET_KEY`: Flask secret key
  - `SMARTNVR_PORT`: Web server port (default: 8000)
  - `SMARTNVR_GPU_ENABLED`: Enable/disable GPU acceleration (default: true)

## Models

Smart-NVR-GPU comes with the following YOLOv5 models:

- **YOLOv5n**: Nano model (fastest, lowest accuracy)
- **YOLOv5s**: Small model (good balance for most use cases)
- **YOLOv5m**: Medium model (higher accuracy, moderate resource usage)
- **YOLOv5l**: Large model (high accuracy, higher resource usage)
- **YOLOv5x**: Extra large model (highest accuracy, highest resource usage)

Custom models can be added through the Admin > AI Models section.

## Performance Optimization

- Use the smallest YOLOv5 model that meets your detection needs
- Lower the resolution or frame rate of camera feeds for better performance
- Create focused Regions of Interest rather than analyzing the entire frame
- Configure detection thresholds to balance accuracy and false positives
- Ensure your GPU has adequate VRAM for the number of camera streams

## Troubleshooting

- Check logs in the `/logs` directory for detailed error information
- Verify camera RTSP URLs are accessible from the host machine
- Ensure proper GPU drivers are installed for CUDA acceleration
- For memory issues, reduce the number of cameras or lower resolution
- Database errors can usually be resolved by running `initialize_db.py`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) for the object detection models
- [Flask](https://flask.palletsprojects.com/) web framework
- [OpenCV](https://opencv.org/) for video processing
- [PyTorch](https://pytorch.org/) for deep learning functionality
- [SQLAlchemy](https://www.sqlalchemy.org/) for database operations