#!/usr/bin/env python3
"""
License Plate Recognition System - Main Detection Script
Real-time webcam-based LPR using YOLOv8 and EasyOCR
"""

import cv2
import sqlite3
import easyocr
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO
import argparse
from database import DatabaseManager
from webcam_capture import WebcamCapture
from plate_detector import PlateDetector
from plate_ocr import PlateOCR

class LPRSystem:
    def __init__(self, db_path="hits.db", plates_dir="plates", hotlist_path="hotlist.csv"):
        self.db_path = db_path
        self.plates_dir = plates_dir
        self.hotlist_path = hotlist_path
        
        # Initialize database using the new DatabaseManager
        self.db_manager = DatabaseManager(db_path)
        
        # Create plates directory
        os.makedirs(self.plates_dir, exist_ok=True)
        
        # Initialize advanced plate detector
        print("Initializing PlateDetector...")
        self.plate_detector = PlateDetector()
        
        # Initialize enhanced OCR module
        print("Initializing PlateOCR...")
        self.plate_ocr = PlateOCR(confidence_threshold=0.5)
        
        # Skip hotlist loading (Task 6 skipped)
        self.hotlist = set()  # Empty set since we're not using hotlist
        
        print("LPR System initialized successfully!")
    
    # Database initialization is now handled by DatabaseManager in __init__
    
    def load_hotlist(self):
        """Load hotlist from CSV file"""
        hotlist = set()
        if os.path.exists(self.hotlist_path):
            with open(self.hotlist_path, 'r') as f:
                for line in f:
                    plate = line.strip().upper()
                    if plate:
                        hotlist.add(plate)
        print(f"Loaded {len(hotlist)} plates from hotlist")
        return hotlist
    
    def detect_plates(self, frame):
        """Detect license plates in frame using advanced PlateDetector"""
        try:
            # Use the new PlateDetector for more accurate detection
            plate_detections = self.plate_detector.detect_plates(frame)
            
            # Extract just the plate crops for backward compatibility
            plate_crops = []
            for plate_crop, detection_info in plate_detections:
                plate_crops.append(plate_crop)
            
            return plate_crops
            
        except Exception as e:
            print(f"Plate detection error: {e}")
            return []
    
    def ocr_plate_text(self, plate_crop):
        """Extract text from plate crop using enhanced PlateOCR"""
        try:
            # Use the enhanced OCR module
            result = self.plate_ocr.read_plate(plate_crop)
            
            if result and result.get('text'):
                # Return the text along with additional info for debugging
                plate_text = result['text']
                confidence = result.get('confidence', 0)
                processing_time = result.get('processing_time_ms', 0)
                
                print(f"OCR: '{plate_text}' (conf: {confidence:.3f}, time: {processing_time:.1f}ms)")
                return plate_text
            
        except Exception as e:
            print(f"OCR error: {e}")
        
        return None
    
    def save_detection(self, plate_text, image_path, is_alert):
        """Save detection to database using DatabaseManager"""
        timestamp = datetime.now().isoformat()
        
        success = self.db_manager.insert_hit(
            timestamp, 
            plate_text, 
            image_path, 
            int(is_alert)
        )
        
        if success:
            print(f"💾 Saved to database: {plate_text}")
        else:
            print(f"❌ Failed to save detection: {plate_text}")
    
    def save_plate_image(self, plate_crop, plate_text):
        """Save plate crop image to disk with error handling and console feedback"""
        try:
            # Ensure plates directory exists
            os.makedirs(self.plates_dir, exist_ok=True)
            
            # Create timestamp and clean filename
            timestamp = int(time.time())
            # Clean plate text for filename (remove special chars)
            clean_plate = "".join(c for c in plate_text if c.isalnum())
            filename = f"{timestamp}_{clean_plate}.jpg"
            image_path = os.path.join(self.plates_dir, filename)
            
            # Save image
            cv2.imwrite(image_path, plate_crop)
            print(f"📸 Saved plate image: {filename}")
            return image_path
            
        except Exception as e:
            print(f"❌ Error saving plate image: {e}")
            return None
    
    def run_detection(self):
        """Main detection loop using WebcamCapture module"""
        print("Starting webcam detection... Press 'q' to quit")
        
        try:
            # Initialize webcam using the new WebcamCapture module
            with WebcamCapture(camera_id=0, width=1280, height=720, fps=30) as webcam:
                print("Webcam initialized successfully")
                
                # Print camera info
                camera_info = webcam.get_camera_info()
                print(f"Camera FPS: {camera_info.get('target_fps', 'unknown')}")
                print(f"Resolution: {camera_info.get('resolution', {}).get('width', 'unknown')}x{camera_info.get('resolution', {}).get('height', 'unknown')}")
                
                # Performance monitoring
                frame_count = 0
                detection_start_time = time.time()
                
                while True:
                    # Get frame using the webcam capture module
                    frame = webcam.get_frame()
                    if frame is None:
                        print("Error reading frame")
                        continue
                    
                    # Detect potential plate regions
                    plate_crops = self.detect_plates(frame)
                    
                    # Process each detected region
                    for crop in plate_crops:
                        plate_text = self.ocr_plate_text(crop)
                        
                        if plate_text:
                            # Save plate image (Task 7 implementation)
                            image_path = self.save_plate_image(crop, plate_text)
                            
                            # Skip hotlist checking (Task 6 skipped)
                            is_alert = False  # No hotlist checking
                            
                            # Save to database
                            self.save_detection(plate_text, image_path, is_alert)
                            
                            # Visual confirmation of detection
                            print(f"🚗 Detected plate: {plate_text}")
                    
                    # Display frame (optional)
                    cv2.imshow('LPR System', frame)
                    
                    # Show actual FPS and performance stats in window title
                    actual_fps = webcam.actual_fps
                    frame_count += 1
                    
                    if actual_fps > 0:
                        # Get detection and OCR performance stats every 30 frames
                        if frame_count % 30 == 0:
                            perf_stats = self.plate_detector.get_performance_stats()
                            ocr_stats = self.plate_ocr.get_performance_stats()
                            
                            detection_time = perf_stats.get('recent_detection_time_ms', 0)
                            ocr_time = ocr_stats.get('recent_processing_time_ms', 0)
                            ocr_success_rate = ocr_stats.get('success_rate', 0)
                            
                            cv2.setWindowTitle('LPR System', 
                                             f'LPR System - FPS: {actual_fps:.1f} | Det: {detection_time:.1f}ms | OCR: {ocr_time:.1f}ms')
                            
                            # Print performance summary
                            elapsed = time.time() - detection_start_time
                            avg_fps = frame_count / elapsed if elapsed > 0 else 0
                            print(f"Frame {frame_count}: Webcam FPS: {actual_fps:.1f}, "
                                  f"Avg FPS: {avg_fps:.1f}, Detection: {detection_time:.1f}ms, "
                                  f"OCR: {ocr_time:.1f}ms, OCR Success: {ocr_success_rate:.1%}")
                        else:
                            cv2.setWindowTitle('LPR System', f'LPR System - FPS: {actual_fps:.1f}')
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        except KeyboardInterrupt:
            print("\nStopping detection...")
        except Exception as e:
            print(f"Detection error: {e}")
        finally:
            cv2.destroyAllWindows()
            print("Detection stopped")

def main():
    parser = argparse.ArgumentParser(description='License Plate Recognition System')
    parser.add_argument('--db', default='hits.db', help='Database file path')
    parser.add_argument('--plates-dir', default='plates', help='Directory to save plate images')
    parser.add_argument('--hotlist', default='hotlist.csv', help='Hotlist CSV file path')
    
    args = parser.parse_args()
    
    # Initialize and run LPR system
    lpr = LPRSystem(args.db, args.plates_dir, args.hotlist)
    lpr.run_detection()

if __name__ == "__main__":
    main() 