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
from datetime import datetime
from ultralytics import YOLO
import argparse

class LPRSystem:
    def __init__(self, db_path="hits.db", plates_dir="plates", hotlist_path="hotlist.csv"):
        self.db_path = db_path
        self.plates_dir = plates_dir
        self.hotlist_path = hotlist_path
        
        # Initialize database
        self.init_database()
        
        # Create plates directory
        os.makedirs(self.plates_dir, exist_ok=True)
        
        # Initialize YOLOv8 model (lazy loading to avoid initialization issues)
        print("Loading YOLOv8 model...")
        self.yolo_model = None  # Will be loaded on first use
        
        # Initialize EasyOCR (lazy loading)
        print("Initializing EasyOCR...")
        self.ocr_reader = None  # Will be loaded on first use
        
        # Load hotlist
        self.hotlist = self.load_hotlist()
        
        print("LPR System initialized successfully!")
    
    def init_database(self):
        """Initialize SQLite database with required schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create hits table as per PRD schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hits(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                plate TEXT,
                image TEXT,
                alert INTEGER
            )
        ''')
        
        # Create index for faster plate lookups
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_plate ON hits(plate)')
        
        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_path}")
    
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
        """Detect license plates in frame using YOLOv8"""
        # Lazy load YOLO model on first use
        if self.yolo_model is None:
            import torch
            # Set weights_only=False to handle PyTorch 2.6+ compatibility
            torch.serialization._WEIGHTS_ONLY_DEFAULT = False
            self.yolo_model = YOLO('yolov8n.pt')
        
        # For now, we'll use a simple vehicle detection approach
        # In a full implementation, you'd use a specialized plate detection model
        results = self.yolo_model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
        
        plate_crops = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Crop the detected vehicle region
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size > 0:
                        plate_crops.append(crop)
        
        return plate_crops
    
    def ocr_plate_text(self, plate_crop):
        """Extract text from plate crop using EasyOCR"""
        try:
            # Lazy load OCR reader on first use
            if self.ocr_reader is None:
                self.ocr_reader = easyocr.Reader(['en'])
            
            results = self.ocr_reader.readtext(plate_crop)
            
            # Filter and clean results
            plate_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5 and len(text) >= 3:  # Minimum confidence and length
                    # Clean the text (remove spaces, special chars)
                    clean_text = ''.join(c.upper() for c in text if c.isalnum())
                    if clean_text:
                        plate_texts.append((clean_text, confidence))
            
            # Return the most confident result
            if plate_texts:
                return max(plate_texts, key=lambda x: x[1])[0]
            
        except Exception as e:
            print(f"OCR error: {e}")
        
        return None
    
    def save_detection(self, plate_text, image_path, is_alert):
        """Save detection to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO hits (ts, plate, image, alert)
            VALUES (?, ?, ?, ?)
        ''', (timestamp, plate_text, image_path, int(is_alert)))
        
        conn.commit()
        conn.close()
        
        print(f"Saved detection: {plate_text} {'[ALERT]' if is_alert else ''}")
    
    def save_plate_image(self, plate_crop, plate_text):
        """Save plate crop image to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{plate_text}.jpg"
        image_path = os.path.join(self.plates_dir, filename)
        
        cv2.imwrite(image_path, plate_crop)
        return image_path
    
    def run_detection(self):
        """Main detection loop"""
        print("Starting webcam detection... Press 'q' to quit")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Detect potential plate regions
                plate_crops = self.detect_plates(frame)
                
                # Process each detected region
                for crop in plate_crops:
                    plate_text = self.ocr_plate_text(crop)
                    
                    if plate_text:
                        # Save plate image
                        image_path = self.save_plate_image(crop, plate_text)
                        
                        # Check if plate is in hotlist
                        is_alert = plate_text in self.hotlist
                        
                        # Save to database
                        self.save_detection(plate_text, image_path, is_alert)
                        
                        if is_alert:
                            print(f"🚨 HOTLIST ALERT: {plate_text}")
                
                # Display frame (optional)
                cv2.imshow('LPR System', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping detection...")
        finally:
            cap.release()
            cv2.destroyAllWindows()

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