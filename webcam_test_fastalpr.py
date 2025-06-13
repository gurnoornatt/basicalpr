#!/usr/bin/env python3
"""
Simple Webcam Test for FastALPR on M3
Real-time license plate detection using the MacBook's built-in camera
"""

import cv2
import time
import logging
from fast_alpr import ALPR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_webcam_fastalpr():
    """Test FastALPR with webcam on M3 MacBook"""
    
    print("🚀 Starting FastALPR Webcam Test for M3...")
    
    # Initialize FastALPR
    try:
        alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="global-plates-mobile-vit-v2-model",
        )
        print("✅ FastALPR initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize FastALPR: {e}")
        return
    
    # Initialize webcam (try index 0 first, which is usually the built-in camera on M3)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set camera properties for better performance on M3
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📹 Webcam opened successfully")
    print("🎯 Looking for license plates... Press 'q' to quit, 'c' to capture frame")
    
    frame_count = 0
    detection_count = 0
    last_detection_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            # Process every 10th frame to avoid overwhelming the system
            if frame_count % 10 == 0:
                try:
                    start_time = time.time()
                    
                    # Run FastALPR detection
                    alpr_results = alpr.predict(frame)
                    
                    detection_time = time.time() - start_time
                    
                    # Process results
                    if alpr_results:
                        # FastALPR returns list[ALPRResult] where each has .detection and .ocr
                        for i, alpr_result in enumerate(alpr_results):
                            detection_count += 1
                            
                            # Extract plate information from ALPRResult structure
                            plate_text = ""
                            confidence = 0.0
                            
                            # Get OCR results
                            if hasattr(alpr_result, 'ocr') and alpr_result.ocr:
                                plate_text = alpr_result.ocr.text if hasattr(alpr_result.ocr, 'text') else ""
                                confidence = alpr_result.ocr.confidence if hasattr(alpr_result.ocr, 'confidence') else 0.0
                            
                            if plate_text and confidence > 0.3:  # Lower threshold for testing
                                print(f"🚗 DETECTED: {plate_text} (confidence: {confidence:.2f}, time: {detection_time:.3f}s)")
                                last_detection_time = time.time()
                    
                    # Draw results on frame using FastALPR's built-in method
                    try:
                        annotated_frame = alpr.draw_predictions(frame)
                    except:
                        annotated_frame = frame
                    
                except Exception as e:
                    logger.error(f"Detection error: {e}")
                    annotated_frame = frame
            else:
                annotated_frame = frame
            
            # Add status text
            status_text = f"Frame: {frame_count} | Detections: {detection_count}"
            cv2.putText(annotated_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('FastALPR Webcam Test', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('c'):
                # Capture current frame for testing
                timestamp = int(time.time())
                filename = f"webcam_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Captured frame saved as: {filename}")
                
                # Test detection on captured frame
                try:
                    results = alpr.predict(frame)
                    print(f"🔍 Detection test on captured frame: {len(results) if results else 0} plates found")
                    if results:
                        for alpr_result in results:
                            if hasattr(alpr_result, 'ocr') and alpr_result.ocr:
                                text = alpr_result.ocr.text
                                conf = alpr_result.ocr.confidence
                                print(f"  - {text} (confidence: {conf:.2f})")
                            else:
                                print(f"  - {alpr_result}")
                except Exception as e:
                    print(f"❌ Detection test failed: {e}")
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"📊 Final stats: {frame_count} frames processed, {detection_count} detections")

if __name__ == "__main__":
    test_webcam_fastalpr() 