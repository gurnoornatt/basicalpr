#!/bin/bash

# LPR System Startup Script
# Boots detector + dashboard as specified in PRD

echo "🚗 Starting LPR System..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create plates directory if it doesn't exist
mkdir -p plates

# Initialize sample hotlist if it doesn't exist
if [ ! -f "hotlist.csv" ]; then
    echo "Creating sample hotlist..."
    python3 update_hotlist.py init
fi

# Initialize database if it doesn't exist
if [ ! -f "hits.db" ]; then
    echo "Initializing database..."
    python3 -c "from alpr import LPRSystem; lpr = LPRSystem(); print('Database initialized')"
fi

echo "✅ Setup complete!"
echo ""

# Function to kill background processes on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down LPR system..."
    kill $DASHBOARD_PID $DETECTOR_PID 2>/dev/null
    wait $DASHBOARD_PID $DETECTOR_PID 2>/dev/null
    echo "✅ Shutdown complete"
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start dashboard in background
echo "🌐 Starting dashboard server at http://localhost:8000..."
python3 dashboard.py --host localhost --port 8000 &
DASHBOARD_PID=$!

# Wait a moment for dashboard to start
sleep 2

# Check if dashboard started successfully
if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "✅ Dashboard server started (PID: $DASHBOARD_PID)"
    echo "📊 Open http://localhost:8000 in your browser to view detections"
else
    echo "❌ Failed to start dashboard server"
    exit 1
fi

echo ""
echo "🎥 Starting LPR detection system..."
echo "👀 Point your webcam at license plates"
echo "⚠️  Press Ctrl+C to stop the system"
echo ""

# Start detector in foreground
python3 alpr.py &
DETECTOR_PID=$!

# Wait for processes
wait $DETECTOR_PID 