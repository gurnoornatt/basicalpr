# 🚗 License Plate Recognition (LPR) System

A real-time license plate recognition system using YOLOv8 for vehicle detection and EasyOCR for text recognition, with a web dashboard for monitoring detections.

## ✨ Features

- **Real-time Detection**: Live webcam-based license plate recognition
- **Web Dashboard**: Modern web interface at http://localhost:8000
- **Image Snapshots**: Automatic saving of detected plate images
- **Database Logging**: SQLite database for detection history
- **Performance Monitoring**: Real-time FPS and processing time stats
- **Responsive Design**: Dashboard works on desktop and mobile

## 🎯 Perfect For

- Testing with 3-4 cars (lightweight, laptop-friendly)
- Security monitoring
- Parking lot management
- Educational/research projects

## 📋 Requirements

- **Python**: 3.10 or higher
- **Operating System**: macOS, Linux, or Windows
- **Webcam**: Any USB or built-in camera
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd testyolo
```

### 2. Run the System
```bash
./run.sh
```

That's it! The script will:
- Create a virtual environment
- Install all dependencies
- Start the dashboard server
- Launch the detection system

### 3. Access Dashboard
Open your browser to: **http://localhost:8000**

### 4. Stop the System
Press `Ctrl+C` in the terminal

## 📁 Project Structure

```
testyolo/
├── alpr.py              # Main detection system
├── dashboard.py         # Web dashboard backend
├── database.py          # Database management
├── webcam_capture.py    # Camera interface
├── plate_detector.py    # YOLOv8 plate detection
├── plate_ocr.py         # EasyOCR text recognition
├── run.sh              # Startup script
├── requirements.txt     # Python dependencies
├── static/             # Dashboard frontend
│   └── index.html      # Web interface
├── plates/             # Saved plate images
├── hits.db             # Detection database
└── yolov8n.pt          # AI model weights
```

## ⚙️ Configuration

### Command Line Options

```bash
# Custom database location
python alpr.py --db /path/to/custom.db

# Custom plates directory
python alpr.py --plates-dir /path/to/plates

# Custom hotlist file
python alpr.py --hotlist /path/to/hotlist.csv
```

### Dashboard Configuration

The dashboard runs on port 8000 by default. To change:

```bash
uvicorn dashboard:app --host 0.0.0.0 --port 8080
```

### Performance Tuning

Edit these values in the respective files:

**Webcam Settings** (`webcam_capture.py`):
- Resolution: 1280x720 (default)
- FPS: 30 (default)

**Detection Settings** (`plate_detector.py`):
- Confidence threshold: 0.5
- Model: YOLOv8n (fastest)

**OCR Settings** (`plate_ocr.py`):
- Confidence threshold: 0.5
- Language: English only

## 📊 Dashboard Features

### Statistics Cards
- **Total Detections**: All-time detection count
- **Recent (24h)**: Detections in last 24 hours  
- **Alerts**: Hot-list matches (if enabled)

### Detection Table
- **Time**: When the plate was detected
- **Plate**: License plate text
- **Image**: Clickable thumbnail (opens full size)
- **Alert**: Hot-list match status

### Auto-Refresh
- Updates every 5 seconds automatically
- Manual refresh button available
- Pauses when browser tab is hidden (saves resources)

## 🔧 Troubleshooting

### Common Issues

**"Address already in use" Error**
```bash
# Kill processes using port 8000
lsof -ti:8000 | xargs kill -9
```

**Webcam Not Found**
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).read()[0]])"
```

**Low Detection Accuracy**
- Ensure good lighting conditions
- Position camera 5-10 feet from plates
- Clean camera lens
- Adjust webcam angle for clear plate view

**High CPU Usage**
- Reduce webcam resolution in `webcam_capture.py`
- Lower FPS setting
- Use YOLOv8n model (fastest)
- Close other applications

**Database Errors**
```bash
# Reset database
rm hits.db
python -c "from alpr import LPRSystem; LPRSystem()"
```

### Performance Optimization

**For Older Laptops:**
- Set webcam to 640x480 resolution
- Reduce FPS to 15
- Use CPU-only mode for OCR

**For Better Accuracy:**
- Use 1920x1080 resolution
- Ensure stable lighting
- Position camera perpendicular to plates

### Log Files

Check these for debugging:
- Terminal output (real-time logs)
- Database: `hits.db` (SQLite browser)
- Images: `plates/` directory

## 🛠️ Development

### Adding New Features

1. **Custom Detection Logic**: Modify `plate_detector.py`
2. **OCR Improvements**: Edit `plate_ocr.py`  
3. **Dashboard Changes**: Update `static/index.html`
4. **Database Schema**: Modify `database.py`

### Testing

```bash
# Test individual components
python -c "from alpr import LPRSystem; lpr = LPRSystem(); print('✅ System OK')"

# Test webcam
python -c "from webcam_capture import WebcamCapture; w = WebcamCapture(); print('✅ Camera OK')"

# Test dashboard
curl http://localhost:8000/hits
```

## 📝 Dependencies

Core libraries installed automatically:
- `opencv-python-headless`: Computer vision
- `ultralytics`: YOLOv8 object detection
- `easyocr`: Optical character recognition
- `fastapi`: Web dashboard backend
- `uvicorn`: ASGI server

## 🔒 Privacy & Security

- All data stored locally (no cloud uploads)
- Images saved in `plates/` directory
- Database contains only plate text and timestamps
- No personal information collected
- Dashboard accessible only on local network

## 📄 License

This project is for educational and research purposes. Ensure compliance with local privacy laws when using for surveillance.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review terminal output for error messages
3. Ensure all requirements are met
4. Test with good lighting and camera positioning

---

**Happy License Plate Recognition! 🚗📸** 