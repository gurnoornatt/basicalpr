
#!/usr/bin/env python3
"""
FastALPR-based License Plate Recognition System
A simplified, high-performance ALPR system using the FastALPR library
"""

import cv2
import os
import time
from datetime import datetime
import logging
import re
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from fast_alpr import ALPR
from database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastALPRSystem:
    """
    High-performance License Plate Recognition System using FastALPR
    """
    
    def __init__(self, db_path="hits.db", plates_dir="plates", hotlist_path="hotlist.csv"):
        """
        Initialize the FastALPR System
        
        Args:
            db_path: Path to SQLite database for storing detections
            plates_dir: Directory to save detected plate images
            hotlist_path: Path to CSV file with hotlisted plates
        """
        # Initialize FastALPR with high-performance models
        logger.info("Initializing FastALPR system...")
        try:
            self.alpr = ALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model",
                detector_providers=['CPUExecutionProvider'],  # Force CPU for M3 compatibility
                ocr_providers=['CPUExecutionProvider'],       # Force CPU for M3 compatibility
            )
            logger.info("✅ FastALPR initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize FastALPR: {e}")
            raise
        
        # Initialize database
        self.db_manager = DatabaseManager(db_path)
        logger.info(f"✅ Database initialized: {db_path}")
        
        # Set up directories
        self.plates_dir = plates_dir
        os.makedirs(plates_dir, exist_ok=True)
        
        # Load hotlist
        self.hotlist_path = hotlist_path
        self.hotlist = self.load_hotlist()
        
        # Performance tracking
        self.frame_count = 0
        self.detection_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # Detection settings
        self.min_confidence = 0.5
        self.duplicate_time_threshold = 30  # seconds to avoid duplicate detections
        self.recent_detections = {}  # plate_text -> timestamp
        
        logger.info("🚀 FastALPR System initialized successfully!")
    
    def load_hotlist(self) -> set:
        """Load hotlist from CSV file"""
        hotlist = set()
        if os.path.exists(self.hotlist_path):
            try:
                with open(self.hotlist_path, 'r') as f:
                    for line in f:
                        plate = line.strip().upper()
                        if plate:
                            hotlist.add(plate)
                logger.info(f"✅ Loaded {len(hotlist)} plates from hotlist")
            except Exception as e:
                logger.error(f"❌ Error loading hotlist: {e}")
        else:
            logger.warning(f"⚠️ Hotlist file not found: {self.hotlist_path}")
        return hotlist
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in frame using FastALPR
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries with plate info
        """
        try:
            start_time = time.time()
            
            # Run FastALPR prediction
            alpr_results = self.alpr.predict(frame)
            
            detection_time = time.time() - start_time
            logger.debug(f"⚡ FastALPR detection took {detection_time:.3f}s")
            
            detections = []
            
            if alpr_results:
                # FastALPR returns list[ALPRResult] where each has .detection and .ocr
                for alpr_result in alpr_results:
                    # Extract plate information from ALPRResult structure
                    plate_text = ""
                    confidence = 0.0
                    bbox = None
                    
                    # Get OCR results
                    if hasattr(alpr_result, 'ocr') and alpr_result.ocr:
                        plate_text = alpr_result.ocr.text if hasattr(alpr_result.ocr, 'text') else ""
                        confidence = alpr_result.ocr.confidence if hasattr(alpr_result.ocr, 'confidence') else 0.0
                    
                    # Get bounding box from detection
                    if hasattr(alpr_result, 'detection') and hasattr(alpr_result.detection, 'bounding_box'):
                        bbox = alpr_result.detection.bounding_box
                    
                    # Clean and validate plate text
                    cleaned_text = self.clean_plate_text(plate_text)
                    
                    if cleaned_text and confidence >= self.min_confidence:
                        # bbox already extracted above
                        
                        detection = {
                            'plate_text': cleaned_text,
                            'confidence': confidence,
                            'bbox': bbox,
                            'raw_text': plate_text,
                            'detection_time': detection_time
                        }
                        
                        detections.append(detection)
                        logger.info(f"🚗 DETECTED PLATE: {cleaned_text} (confidence: {confidence:.2f})")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error in plate detection: {e}")
            return []
    
    def clean_plate_text(self, text: str) -> str:
        """
        Clean and standardize plate text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned plate text
        """
        if not text:
            return ""
        
        # Remove non-alphanumeric characters and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Common OCR corrections
        corrections = {
            'O': '0',  # O -> 0
            'I': '1',  # I -> 1
            'S': '5',  # S -> 5
            'Z': '2',  # Z -> 2
            'G': '6',  # G -> 6
            'B': '8',  # B -> 8
        }
        
        for old, new in corrections.items():
            cleaned = cleaned.replace(old, new)
        
        return cleaned
    
    def is_valid_plate_text(self, text: str) -> bool:
        """
        Check if text looks like a valid license plate
        
        Args:
            text: Plate text to validate
            
        Returns:
            True if text appears to be a valid plate
        """
        if not text or len(text) < 3:
            return False
        
        # Length check (most plates are 3-8 characters)
        if not (3 <= len(text) <= 8):
            return False
        
        # Should have at least one number or letter
        has_letter = any(c.isalpha() for c in text)
        has_number = any(c.isdigit() for c in text)
        
        return has_letter or has_number
    
    def is_duplicate_detection(self, plate_text: str) -> bool:
        """
        Check if this plate was recently detected to avoid duplicates
        
        Args:
            plate_text: The detected plate text
            
        Returns:
            True if this is a duplicate detection
        """
        current_time = time.time()
        
        if plate_text in self.recent_detections:
            time_diff = current_time - self.recent_detections[plate_text]
            if time_diff < self.duplicate_time_threshold:
                return True
        
        # Update detection time
        self.recent_detections[plate_text] = current_time
        
        # Clean old detections
        cutoff_time = current_time - self.duplicate_time_threshold * 2
        self.recent_detections = {
            plate: timestamp for plate, timestamp in self.recent_detections.items()
            if timestamp > cutoff_time
        }
        
        return False
    
    def save_plate_image(self, frame: np.ndarray, detection: Dict[str, Any]) -> str:
        """
        Save detected plate image to disk
        
        Args:
            frame: Full frame image
            detection: Detection dictionary with bbox info
            
        Returns:
            Path to saved image file
        """
        try:
            plate_text = detection['plate_text']
            timestamp = int(time.time())
            
            # Create filename
            filename = f"auto_scan_{plate_text}_{timestamp}.jpg"
            filepath = os.path.join(self.plates_dir, filename)
            
            # If we have bbox info, crop the plate region
            if detection.get('bbox'):
                bbox = detection['bbox']
                # Handle different bbox formats
                if hasattr(bbox, 'x1'):
                    # BoundingBox object with x1, y1, x2, y2 attributes
                    x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    # List/tuple format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox[:4]
                else:
                    # Fallback: save full frame
                    cv2.imwrite(filepath, frame)
                    logger.info(f"💾 Saved full frame: {filename}")
                    return filename
                
                # Crop plate region from frame
                plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                if plate_crop.size > 0:
                    cv2.imwrite(filepath, plate_crop)
                else:
                    # Fallback to full frame
                    cv2.imwrite(filepath, frame)
            else:
                # Save full frame if no bbox available
                cv2.imwrite(filepath, frame)
            
            logger.info(f"💾 Saved: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error saving plate image: {e}")
            return ""
    
    def save_detection(self, detection: Dict[str, Any], image_path: str, is_hotlist: bool = False):
        """
        Save detection to database
        
        Args:
            detection: Detection dictionary
            image_path: Path to saved image
            is_hotlist: Whether this plate is on the hotlist
        """
        try:
            plate_text = detection['plate_text']
            confidence = detection['confidence']
            
            # Create timestamp
            timestamp = datetime.now().isoformat()
            
            # Insert into database
            success = self.db_manager.insert_hit(
                timestamp=timestamp,
                plate_text=plate_text,
                image_path=image_path,
                alert_status=1 if is_hotlist else 0
            )
            
            if success:
                status = "🚨 HOTLIST HIT" if is_hotlist else "✅ Detection saved"
                logger.info(f"{status}: {plate_text} (confidence: {confidence:.2f})")
            else:
                logger.error(f"❌ Failed to save detection to database")
            
        except Exception as e:
            logger.error(f"❌ Error saving detection: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Process a single frame for license plate detection
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (detections, annotated_frame)
        """
        # Update frame count and FPS
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        # Detect plates
        detections = self.detect_plates(frame)
        
        # Process each detection
        valid_detections = []
        for detection in detections:
            plate_text = detection['plate_text']
            
            # Skip if duplicate
            if self.is_duplicate_detection(plate_text):
                logger.debug(f"⏭️ Skipping duplicate detection: {plate_text}")
                continue
            
            # Check if it's a hotlist hit
            is_hotlist = plate_text in self.hotlist
            
            # Save plate image
            image_path = self.save_plate_image(frame, detection)
            
            # Save to database
            self.save_detection(detection, image_path, is_hotlist)
            
            valid_detections.append(detection)
            self.detection_count += 1
        
        # Draw annotations on frame
        annotated_frame = self.draw_detections(frame, valid_detections)
        
        return valid_detections, annotated_frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection annotations on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in detections:
            plate_text = detection['plate_text']
            confidence = detection['confidence']
            bbox = detection.get('bbox')
            
            # Draw bounding box if available
            if bbox:
                # Handle different bbox formats
                if hasattr(bbox, 'x1'):
                    # BoundingBox object with x1, y1, x2, y2 attributes
                    x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    # List/tuple format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, bbox[:4])
                else:
                    # Skip drawing if unknown format
                    continue
                
                # Choose color based on hotlist status
                color = (0, 0, 255) if plate_text in self.hotlist else (0, 255, 0)  # Red for hotlist, green for normal
                
                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw text
                text = f"{plate_text} ({confidence:.2f})"
                cv2.putText(annotated, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Draw FPS and stats
        stats_text = f"FPS: {self.fps:.1f} | Detections: {self.detection_count}"
        cv2.putText(annotated, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated
    
    def run_webcam_detection(self, camera_index: int = 0):
        """
        Run real-time license plate detection from webcam
        
        Args:
            camera_index: Camera index (usually 0 for default camera)
        """
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("❌ Failed to open webcam")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info("🚀 Starting webcam detection... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("❌ Failed to read frame from webcam")
                    break
                
                # Process frame
                detections, annotated_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow('FastALPR - License Plate Detection', annotated_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("🛑 Detection stopped by user")
                    break
                    
        except KeyboardInterrupt:
            logger.info("🛑 Detection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("🧹 Resources cleaned up")
    
    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process a single image file for license plate detection
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detections
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                logger.error(f"❌ Could not load image: {image_path}")
                return []
            
            detections, annotated_frame = self.process_frame(frame)
            
            # Save annotated result
            output_path = f"annotated_{os.path.basename(image_path)}"
            cv2.imwrite(output_path, annotated_frame)
            logger.info(f"💾 Saved annotated result: {output_path}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error processing image: {e}")
            return []

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='FastALPR License Plate Recognition System')
    parser.add_argument('--image', type=str, help='Process single image file')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for webcam detection')
    parser.add_argument('--db', type=str, default='hits.db', help='Database path')
    parser.add_argument('--plates-dir', type=str, default='plates', help='Directory to save plate images')
    parser.add_argument('--hotlist', type=str, default='hotlist.csv', help='Path to hotlist CSV file')
    
    args = parser.parse_args()
    
    # Initialize system
    alpr_system = FastALPRSystem(
        db_path=args.db,
        plates_dir=args.plates_dir,
        hotlist_path=args.hotlist
    )
    
    if args.image:
        # Process single image
        logger.info(f"Processing image: {args.image}")
        detections = alpr_system.process_image(args.image)
        logger.info(f"Found {len(detections)} license plates")
    else:
        # Run webcam detection
        alpr_system.run_webcam_detection(args.camera)

if __name__ == "__main__":
    main() 