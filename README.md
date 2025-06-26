# Smart-NVR-GPU

A powerful Network Video Recorder (NVR) application that leverages GPU acceleration for real-time AI object detection, smart recording, and efficient video management. Built with Python, Flask, and YOLOv5, this application provides enterprise-grade surveillance capabilities with a user-friendly interface.

![Smart-NVR-GPU Dashboard](static/img/dashboard-preview.png)

## Features

- **GPU-Accelerated AI Detection**: Real-time object detection using YOLOv5, v8, v9, v10 models with CUDA acceleration
- **Smart Recording Management**: Automatic recording based on motion or specific AI detection events
- **Live Camera Dashboard**: Monitor multiple RTSP/IP cameras simultaneously with object detection overlays
- **Regions of Interest (ROI)**: Define specific areas for detection with time-based scheduling support
- **Advanced Playback**: Timeline-based video playback with object detection markers and filtering
- **Gemini AI Integration**: Human-friendly, context-aware descriptions for detection events using Google's Gemini AI
- **System Resource Monitoring**: Track CPU, RAM, GPU, and disk usage in real-time
- **Modern UI**: Clean, responsive interface designed for ease of use
- **Multi-User Support**: Role-based access with administrative and standard user accounts
- **Notifications**: Configurable alerts for specific object detections via email
- **API Access**: RESTful API for integration with other systems
- **MongoDB Support**: Scalable NoSQL database for improved performance and flexibility

## Requirements

- Python 3.8+ (3.10+ recommended)
- CUDA-compatible GPU (strongly recommended for real-time processing)
- NVIDIA drivers and CUDA toolkit (for GPU acceleration)
- RTSP/IP compatible cameras
- 8GB+ RAM (16GB+ recommended for multiple camera streams)
- MongoDB 4.4+ (for database storage)
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

4. Install MongoDB:
```bash
# For Ubuntu/Debian
sudo apt update
sudo apt install -y mongodb-org

# For CentOS/RHEL
sudo yum install -y mongodb-org

# For macOS with Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
sudo systemctl start mongod    # Linux
brew services start mongodb-community  # macOS
```

5. Initialize the database:
```bash
python initialize_db.py
```

6. Start the application:
```bash
python run.py
```

7. Access the web interface at http://localhost:8000

### Docker Installation

```bash
# Build the Docker image
docker build -t smart-nvr-gpu .

# Run the container with GPU support and MongoDB
docker run --gpus all -p 8000:8000 -v /path/to/storage:/app/storage \
  --name smart-nvr -d smart-nvr-gpu
```

### Docker Compose Setup (Recommended)

1. Create a `docker-compose.yml` file:
```yaml
version: '3'
services:
  smart-nvr:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./storage:/app/storage
      - ./config:/app/config
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - mongodb
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/smartnvr
  
  mongodb:
    image: mongo:4.4
    volumes:
      - mongodb_data:/data/db
    ports:
      - "27017:27017"

volumes:
  mongodb_data:
```

2. Start the application using Docker Compose:
```bash
docker-compose up -d
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

4. Configure Regions of Interest (ROI) to focus detection on specific areas:
   - Click "Manage ROI" on any camera card
   - Draw regions using the interface
   - Choose between "Always Active" or "Time-Based Schedule" ROI types
   - Configure time schedules and active days for time-based ROIs
   - Select specific object classes for detection in each region
   - Enable email notifications if desired

5. Use the Recordings section to review detection events and continuous recordings

## MongoDB Transition

As of April 2025, Smart-NVR has transitioned from SQLite to MongoDB as its primary database engine. This transition provides several benefits:

### Benefits of MongoDB
- **Improved Performance**: Better handling of concurrent read/write operations
- **Enhanced Scalability**: Easily scale to thousands of cameras and detection events
- **Flexible Schema**: Adapt to changing data requirements without migration hassles
- **Native JSON Support**: Simplified API interactions and data processing
- **Robust Querying**: Advanced query capabilities for complex filtering

### Migration Instructions
If you're upgrading from a previous SQLite-based version:

1. Back up your existing data:
```bash
python backup_db.py
```

2. Install MongoDB following the installation instructions above

3. Run the migration script:
```bash
python migrate_to_mongodb.py
```

4. Start Smart-NVR with MongoDB:
```bash
python run.py --db-type=mongodb
```

### MongoDB Configuration
The MongoDB connection can be configured in `config/settings.json`:

```json
{
  "database": {
    "type": "mongodb",
    "uri": "mongodb://localhost:27017/",
    "name": "smartnvr"
  }
}
```

Or using environment variables:
```bash
export SMARTNVR_DB_TYPE=mongodb
export SMARTNVR_DB_URI=mongodb://localhost:27017/smartnvr
```

## Gemini AI Integration

As of May 2025, Smart-NVR-GPU includes integration with Google's Gemini AI for enhanced detection notifications:

### Features
- **Smart Detection Descriptions**: Convert technical detection metadata into human-friendly descriptions
- **Context-Aware Notifications**: Email alerts that describe what's happening in natural language
- **Configurable per ROI**: Enable/disable Gemini AI on a per-region basis
- **Fallback Mechanism**: Automatically reverts to standard notifications if Gemini AI is unavailable

### Setup Instructions
1. **Obtain a Gemini API Key**:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to get your API key
   - Free tier is available with generous limits for typical home usage

2. **Configure in Settings**:
   - Navigate to Settings page in Smart-NVR
   - Under "AI Detection Settings", find "Gemini AI Enhancement"  
   - Enable Gemini AI, enter your API key, and select Gemini model
   - Click "Test Gemini Connection" to verify functionality

3. **Enable for ROI Notifications**:
   - In the Camera Management page, click "Manage ROI" for any camera
   - Create or edit a region, and ensure email notifications are enabled
   - In the "Smart Email Notifications" panel, enable "Use Gemini AI Smart Descriptions"

### Gemini Models
- **Gemini 1.5 Flash**: Fastest response, good for typical descriptions (recommended)
- **Gemini 1.5 Pro**: More detailed analysis, but slightly slower
- **Gemini Pro**: Original model, maintained for backward compatibility

### Example Usage
When Gemini AI is enabled for an ROI, instead of a standard notification like:
```
Object detected: person (0.87 confidence) in ROI "Backyard"
```

You'll receive a more descriptive notification such as:
```
A person was detected walking through your Backyard at 3:42 PM. They appear to be carrying a package and approaching your back door.
```

### Environment Variables
- `SMARTNVR_GEMINI_API_KEY`: API key for Gemini AI (alternative to UI configuration)
- `SMARTNVR_GEMINI_MODEL`: Model to use (gemini-1.5-flash, gemini-1.5-pro, gemini-pro)

## Advanced Configuration

The application can be configured through the web interface or by editing these files:

- `config/settings.json`: Main application settings
- `app/utils/camera_processor.py`: Camera processing and AI detection parameters
- Environment variables:
  - `SMARTNVR_SECRET_KEY`: Flask secret key
  - `SMARTNVR_PORT`: Web server port (default: 8000)
  - `SMARTNVR_GPU_ENABLED`: Enable/disable GPU acceleration (default: true)
  - `SMARTNVR_DB_TYPE`: Database type (mongodb or sqlite)
  - `SMARTNVR_DB_URI`: MongoDB connection URI

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
- With MongoDB, consider indexing frequently queried fields for faster retrieval

## Troubleshooting

- Check logs in the `/logs` directory for detailed error information
- Verify camera RTSP URLs are accessible from the host machine
- Ensure proper GPU drivers are installed for CUDA acceleration
- For memory issues, reduce the number of cameras or lower resolution
- MongoDB connection errors: Check if MongoDB is running with `sudo systemctl status mongod`
- For MongoDB authentication issues, verify credentials in `config/settings.json`

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
- [MongoDB](https://www.mongodb.com/) for database operations
- [PyMongo](https://pymongo.readthedocs.io/) for MongoDB connectivity in Python