#!/usr/bin/env python3
"""
Webcam Capture Module
Handles video frame capture from webcam with frame rate control and error handling
"""

import cv2
import time
import threading
import logging
from typing import Optional, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebcamCapture:
    """
    A robust webcam capture class with frame rate control and error handling.
    Designed for MacBook built-in webcam but works with any compatible camera.
    """
    
    def __init__(self, camera_id: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Camera device ID (default: 0 for built-in webcam)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        
        # Camera object
        self.cap = None
        self.is_opened = False
        
        # Frame rate control
        self.last_frame_time = 0
        self.actual_fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Threading for continuous capture
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Initialize camera
        self._initialize_camera()
    
    def _initialize_camera(self) -> None:
        """Initialize the camera with error handling."""
        try:
            logger.info(f"Initializing camera {self.camera_id}...")
            
            # Create VideoCapture object
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera {self.camera_id}")
            
            # Configure camera properties
            self._configure_camera()
            
            # Test frame capture
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Failed to capture test frame")
            
            self.is_opened = True
            logger.info(f"Camera {self.camera_id} initialized successfully")
            logger.info(f"Actual resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            raise
    
    def _configure_camera(self) -> None:
        """Configure camera properties."""
        try:
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # macOS specific optimizations
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize latency
            
            # Optional: Set additional properties for better quality
            # Uncomment these if your camera supports them
            # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
            # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
            # self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)
            # self.cap.set(cv2.CAP_PROP_SATURATION, 0.5)
            
            logger.info("Camera properties configured")
            
        except Exception as e:
            logger.warning(f"Some camera properties could not be set: {e}")
    
    def get_frame(self) -> Optional[any]:
        """
        Capture a single frame with frame rate control.
        
        Returns:
            Frame as numpy array or None if capture failed
        """
        if not self.is_opened or not self.cap:
            logger.error("Camera not initialized")
            return None
        
        try:
            # Frame rate control
            current_time = time.time()
            time_since_last = current_time - self.last_frame_time
            
            if time_since_last < self.frame_interval:
                sleep_time = self.frame_interval - time_since_last
                time.sleep(sleep_time)
            
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                logger.warning("Failed to capture frame")
                return None
            
            # Update timing
            self.last_frame_time = time.time()
            self._update_fps_counter()
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
    
    def start_continuous_capture(self) -> None:
        """Start continuous frame capture in a separate thread."""
        if self.capture_thread and self.capture_thread.is_alive():
            logger.warning("Continuous capture already running")
            return
        
        self.stop_event.clear()
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Started continuous capture")
    
    def stop_continuous_capture(self) -> None:
        """Stop continuous frame capture."""
        if self.capture_thread and self.capture_thread.is_alive():
            self.stop_event.set()
            self.capture_thread.join(timeout=2.0)
            logger.info("Stopped continuous capture")
    
    def get_latest_frame(self) -> Optional[any]:
        """
        Get the latest frame from continuous capture.
        
        Returns:
            Latest frame or None if not available
        """
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def _capture_loop(self) -> None:
        """Main loop for continuous capture."""
        while not self.stop_event.is_set():
            frame = self.get_frame()
            if frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
    
    def _update_fps_counter(self) -> None:
        """Update actual FPS counter."""
        self.frame_count += 1
        
        # Calculate actual FPS every second
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.actual_fps = self.frame_count / (current_time - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and current settings.
        
        Returns:
            Dictionary with camera information
        """
        if not self.is_opened or not self.cap:
            return {"error": "Camera not initialized"}
        
        info = {
            "camera_id": self.camera_id,
            "is_opened": self.is_opened,
            "target_fps": self.target_fps,
            "actual_fps": round(self.actual_fps, 2),
            "resolution": {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            },
            "properties": {
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "buffer_size": self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
            }
        }
        
        return info
    
    def set_camera_property(self, property_id: int, value: float) -> bool:
        """
        Set a camera property.
        
        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS)
            value: Property value
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_opened or not self.cap:
            logger.error("Camera not initialized")
            return False
        
        try:
            success = self.cap.set(property_id, value)
            if success:
                logger.info(f"Set camera property {property_id} to {value}")
            else:
                logger.warning(f"Failed to set camera property {property_id} to {value}")
            return success
        except Exception as e:
            logger.error(f"Error setting camera property: {e}")
            return False
    
    def release(self) -> None:
        """Release camera resources and clean up."""
        logger.info("Releasing camera resources...")
        
        # Stop continuous capture
        self.stop_continuous_capture()
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.is_opened = False
        logger.info("Camera resources released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.release()


def test_webcam_capture():
    """Test function to verify webcam capture functionality."""
    print("Testing WebcamCapture...")
    
    try:
        # Test basic capture
        with WebcamCapture(camera_id=0, width=1280, height=720, fps=30) as webcam:
            print("Camera initialized successfully")
            
            # Print camera info
            info = webcam.get_camera_info()
            print(f"Camera info: {info}")
            
            # Test single frame capture
            print("Testing single frame capture...")
            for i in range(5):
                frame = webcam.get_frame()
                if frame is not None:
                    print(f"Frame {i+1}: {frame.shape}")
                else:
                    print(f"Frame {i+1}: Failed to capture")
                time.sleep(0.1)
            
            # Test continuous capture
            print("Testing continuous capture for 3 seconds...")
            webcam.start_continuous_capture()
            
            start_time = time.time()
            while time.time() - start_time < 3.0:
                frame = webcam.get_latest_frame()
                if frame is not None:
                    print(f"Continuous frame: {frame.shape}, FPS: {webcam.actual_fps:.1f}")
                time.sleep(0.5)
            
            webcam.stop_continuous_capture()
            print("Test completed successfully!")
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_webcam_capture() 