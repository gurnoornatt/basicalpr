#!/usr/bin/env python3
"""
Simple test script for FastALPR functionality
"""

import cv2
from fast_alpr import ALPR

def test_fastalpr():
    print("🚀 Testing FastALPR...")
    
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
    
    # Test with existing image
    image_path = "test_synthetic_plate.jpg"
    print(f"📸 Testing with image: {image_path}")
    
    try:
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        print(f"📏 Image shape: {frame.shape}")
        
        # Run prediction
        print("🔍 Running FastALPR prediction...")
        alpr_results = alpr.predict(frame)
        
        print(f"📊 Raw ALPR results type: {type(alpr_results)}")
        print(f"📊 Raw ALPR results: {alpr_results}")
        
        # Check if we have results
        if alpr_results:
            if hasattr(alpr_results, 'plates'):
                print(f"🎯 Found {len(alpr_results.plates)} plates")
                for i, plate in enumerate(alpr_results.plates):
                    print(f"  Plate {i+1}:")
                    print(f"    Raw object: {plate}")
                    if hasattr(plate, 'ocr'):
                        print(f"    OCR object: {plate.ocr}")
                        if hasattr(plate.ocr, 'text'):
                            print(f"    Text: {plate.ocr.text}")
                        if hasattr(plate.ocr, 'confidence'):
                            print(f"    Confidence: {plate.ocr.confidence}")
                    if hasattr(plate, 'detection'):
                        print(f"    Detection: {plate.detection}")
            else:
                print(f"📊 Results don't have 'plates' attribute")
                print(f"📊 Available attributes: {dir(alpr_results)}")
        else:
            print("❌ No results returned")
        
        # Try to draw predictions
        try:
            annotated_frame = alpr.draw_predictions(frame)
            cv2.imwrite("test_annotated.jpg", annotated_frame)
            print("💾 Saved annotated image as test_annotated.jpg")
        except Exception as e:
            print(f"⚠️ Could not draw predictions: {e}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fastalpr() 