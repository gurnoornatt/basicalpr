#!/usr/bin/env python3
"""
Main FastALPR Production System
Continuous webcam license plate detection using FastALPR
Automatically saves detections to dashboard database
"""

import cv2
import time
import logging
import signal
import sys
from datetime import datetime
from fast_alpr_system import FastALPRSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductionFastALPR:
    def __init__(self):
        """Initialize the production FastALPR system"""
        self.running = False
        self.cap = None
        self.alpr_system = None
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        
        # Detection settings
        self.detection_interval = 10  # Scan every 10 frames (3 times per second at 30 FPS)
        
    def initialize(self):
        """Initialize camera and ALPR system"""
        try:
            # Initialize FastALPR system
            logger.info("🚀 Initializing FastALPR production system...")
            self.alpr_system = FastALPRSystem()
            logger.info("✅ FastALPR system initialized")
            
            # Initialize camera
            logger.info("📹 Initializing camera...")
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
                
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("✅ Camera initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("🧹 Cleaning up resources...")
        self.running = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        runtime = time.time() - self.start_time
        avg_fps = self.frame_count / runtime if runtime > 0 else 0
        
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   Runtime: {runtime:.1f} seconds")
        logger.info(f"   Frames processed: {self.frame_count}")
        logger.info(f"   Plates detected: {self.detection_count}")
        logger.info(f"   Average FPS: {avg_fps:.1f}")
        logger.info("✅ Cleanup complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("🛑 Shutdown signal received")
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """Main detection loop"""
        if not self.initialize():
            return False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🎥 Starting continuous license plate detection...")
        logger.info("📊 Dashboard available at: http://localhost:8000")
        logger.info("⚠️  Press Ctrl+C to stop")
        logger.info("")
        
        self.running = True
        last_fps_time = time.time()
        fps_counter = 0
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame from camera")
                    continue
                
                self.frame_count += 1
                fps_counter += 1
                
                # Create display frame (copy for annotation)
                display_frame = frame.copy()
                
                # Add status overlay to frame
                status_text = f"FastALPR | Frame: {self.frame_count} | Plates Found: {self.detection_count}"
                cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Perform detection at intervals
                if self.frame_count % self.detection_interval == 0:
                    logger.info(f"🔍 Scanning frame {self.frame_count} for license plates...")
                    detections, annotated_frame = self.alpr_system.process_frame(frame)
                    
                    if annotated_frame is not None:
                        display_frame = annotated_frame  # Use annotated frame with boxes
                        # Re-add status overlay on top of annotations
                        cv2.putText(display_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if detections:
                        self.detection_count += len(detections)
                        logger.info(f"✅ Found {len(detections)} plate(s)")
                
                # Show live video feed with detection boxes and text
                cv2.imshow('FastALPR - License Plate Detection', display_frame)
                
                # Check for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested quit")
                    break
                
                # Update FPS display every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    logger.info(f"🚀 Live FPS: {fps:.1f} | Frame: {self.frame_count}")
                    
                    fps_counter = 0
                    last_fps_time = current_time
                    
        except KeyboardInterrupt:
            logger.info("🛑 Detection interrupted by user")
        except Exception as e:
            logger.error(f"❌ Detection error: {e}")
        finally:
            self.cleanup()
        
        return True

def main():
    """Main entry point"""
    print("🚗 FastALPR Production System")
    print("=============================")
    
    system = ProductionFastALPR()
    success = system.run()
    
    if success:
        print("✅ System stopped successfully")
        return 0
    else:
        print("❌ System failed to start")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 