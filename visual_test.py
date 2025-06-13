#!/usr/bin/env python3

from fast_alpr_system import FastALPRSystem
import cv2
import time

def test_visual_display():
    """Test that visual annotations are working properly"""
    
    print('🎥 Testing FastALPR visual display... Press Q to quit')
    
    # Initialize FastALPR system
    alpr = FastALPRSystem()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    
    while frame_count < 100:  # Test for 100 frames
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame and get annotations
        detections, annotated_frame = alpr.process_frame(frame)
        
        # Add test overlay to verify display is working
        test_text = f'FastALPR Visual Test - Frame {frame_count}'
        cv2.putText(annotated_frame, test_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Add detection status
        status_text = f'Detections: {len(detections)} plates found'
        cv2.putText(annotated_frame, status_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if detections:
            print(f'📸 Frame {frame_count}: {len(detections)} plate(s) detected with visual overlay!')
            for det in detections:
                print(f'   🚗 {det["plate_text"]} (confidence: {det["confidence"]:.2f})')
        
        # Show the frame with annotations
        cv2.imshow('FastALPR Visual Test', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print('✅ Visual test completed!')

if __name__ == "__main__":
    test_visual_display() 