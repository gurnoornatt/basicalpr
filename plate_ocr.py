#!/usr/bin/env python3
"""
Enhanced License Plate OCR Module
Implements advanced EasyOCR-based text extraction with preprocessing and validation
"""

import cv2
import numpy as np
import easyocr
import re
import logging
from typing import List, Tuple, Optional, Dict
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateOCR:
    """
    Advanced license plate OCR using EasyOCR with comprehensive preprocessing
    and validation for improved accuracy.
    """
    
    def __init__(self, languages=['en'], gpu=False, confidence_threshold=0.5):
        """
        Initialize the PlateOCR system.
        
        Args:
            languages: List of languages for OCR (default: ['en'])
            gpu: Whether to use GPU acceleration (default: False for MacBook compatibility)
            confidence_threshold: Minimum confidence for text detection (default: 0.5)
        """
        self.languages = languages
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.reader = None
        
        # Performance tracking
        self.ocr_times = []
        self.total_processed = 0
        self.successful_reads = 0
        
        # Common license plate patterns (US formats)
        self.plate_patterns = [
            r'^[A-Z0-9]{2,3}[A-Z0-9]{3,4}$',  # Standard format: AB1234 or ABC123
            r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,2}$',  # Mixed format: A123B
            r'^[0-9]{1,3}[A-Z]{2,4}[0-9]{0,2}$',  # Number first: 1ABC23
            r'^[A-Z0-9]{5,8}$',  # General alphanumeric
        ]
        
        logger.info("PlateOCR initialized")
    
    def load_reader(self):
        """Lazy load the EasyOCR reader."""
        if self.reader is None:
            try:
                logger.info(f"Loading EasyOCR reader with languages: {self.languages}")
                start_time = time.time()
                
                self.reader = easyocr.Reader(
                    self.languages, 
                    gpu=self.gpu,
                    verbose=False
                )
                
                load_time = (time.time() - start_time) * 1000
                logger.info(f"EasyOCR reader loaded successfully in {load_time:.1f}ms")
                
            except Exception as e:
                logger.error(f"Failed to load EasyOCR reader: {e}")
                raise
        
        return self.reader
    
    def preprocess_plate_image(self, plate_image: np.ndarray) -> List[np.ndarray]:
        """
        Apply multiple preprocessing techniques to improve OCR accuracy.
        
        Args:
            plate_image: Input plate crop image
            
        Returns:
            List of preprocessed images to try OCR on
        """
        if plate_image is None or plate_image.size == 0:
            return []
        
        processed_images = []
        
        try:
            # Convert to grayscale if needed
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image.copy()
            
            # Original grayscale
            processed_images.append(gray)
            
            # Method 1: OTSU thresholding
            try:
                _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(thresh_otsu)
            except:
                pass
            
            # Method 2: Adaptive thresholding
            try:
                thresh_adaptive = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                processed_images.append(thresh_adaptive)
            except:
                pass
            
            # Method 3: Contrast enhancement + OTSU
            try:
                enhanced = cv2.equalizeHist(gray)
                _, thresh_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(thresh_enhanced)
            except:
                pass
            
            # Method 4: Morphological operations
            try:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
                _, thresh_morph = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(thresh_morph)
            except:
                pass
            
            # Method 5: Gaussian blur + threshold
            try:
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                _, thresh_blur = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                processed_images.append(thresh_blur)
            except:
                pass
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            # Fallback to original image
            if len(plate_image.shape) == 3:
                processed_images.append(cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY))
            else:
                processed_images.append(plate_image)
        
        return processed_images
    
    def validate_plate_text(self, text: str) -> bool:
        """
        Validate if extracted text matches common license plate patterns.
        
        Args:
            text: Extracted text string
            
        Returns:
            True if text matches a valid plate pattern
        """
        if not text or len(text) < 4 or len(text) > 8:
            return False
        
        # Clean text - remove spaces and convert to uppercase
        clean_text = ''.join(c.upper() for c in text if c.isalnum())
        
        # Check against known patterns
        for pattern in self.plate_patterns:
            if re.match(pattern, clean_text):
                return True
        
        return False
    
    def clean_plate_text(self, text: str) -> str:
        """
        Clean and format extracted plate text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and formatted plate text
        """
        if not text:
            return ""
        
        # Remove spaces, special characters, keep only alphanumeric
        clean_text = ''.join(c.upper() for c in text if c.isalnum())
        
        # Remove common OCR errors
        # O -> 0, I -> 1, etc. (context-dependent)
        replacements = {
            'O': '0',  # O often misread as 0 in plates
            'I': '1',  # I often misread as 1 in plates
            'S': '5',  # S sometimes misread as 5
            'G': '6',  # G sometimes misread as 6
        }
        
        # Apply replacements carefully (only for obvious cases)
        # This is a basic implementation - more sophisticated logic could be added
        
        return clean_text
    
    def read_plate(self, plate_image: np.ndarray) -> Optional[Dict]:
        """
        Extract text from license plate image using OCR.
        
        Args:
            plate_image: License plate crop image
            
        Returns:
            Dictionary with plate text, confidence, and processing info, or None
        """
        if plate_image is None or plate_image.size == 0:
            return None
        
        start_time = time.time()
        self.total_processed += 1
        
        try:
            reader = self.load_reader()
            
            # Preprocess image with multiple methods
            processed_images = self.preprocess_plate_image(plate_image)
            
            best_result = None
            best_confidence = 0
            
            # Try OCR on each preprocessed image
            for i, processed_img in enumerate(processed_images):
                try:
                    # Resize image if too small (EasyOCR works better with larger images)
                    height, width = processed_img.shape
                    if height < 32 or width < 64:
                        scale_factor = max(64 / width, 32 / height)
                        new_width = int(width * scale_factor)
                        new_height = int(height * scale_factor)
                        processed_img = cv2.resize(processed_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # Run OCR
                    results = reader.readtext(processed_img, detail=1, paragraph=False)
                    
                    # Process results
                    for bbox, text, confidence in results:
                        if confidence > self.confidence_threshold and len(text.strip()) >= 3:
                            cleaned_text = self.clean_plate_text(text)
                            
                            if cleaned_text and self.validate_plate_text(cleaned_text):
                                if confidence > best_confidence:
                                    best_result = {
                                        'text': cleaned_text,
                                        'confidence': confidence,
                                        'preprocessing_method': i,
                                        'raw_text': text,
                                        'bbox': bbox
                                    }
                                    best_confidence = confidence
                
                except Exception as e:
                    logger.debug(f"OCR failed on preprocessing method {i}: {e}")
                    continue
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            self.ocr_times.append(processing_time)
            
            if best_result:
                self.successful_reads += 1
                best_result['processing_time_ms'] = processing_time
                logger.debug(f"Successfully read plate: {best_result['text']} "
                            f"(confidence: {best_result['confidence']:.3f}, "
                            f"time: {processing_time:.1f}ms)")
                return best_result
            else:
                logger.debug(f"No valid plate text found (time: {processing_time:.1f}ms)")
                return None
                
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.ocr_times.append(processing_time)
            logger.error(f"OCR processing failed: {e}")
            return None
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get OCR performance statistics."""
        if not self.ocr_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.ocr_times),
            'max_processing_time_ms': max(self.ocr_times),
            'min_processing_time_ms': min(self.ocr_times),
            'recent_processing_time_ms': self.ocr_times[-1] if self.ocr_times else 0,
            'total_processed': self.total_processed,
            'successful_reads': self.successful_reads,
            'success_rate': (self.successful_reads / self.total_processed) if self.total_processed > 0 else 0
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.ocr_times = []
        self.total_processed = 0
        self.successful_reads = 0

def test_plate_ocr():
    """Test the PlateOCR module."""
    print("Testing PlateOCR...")
    
    ocr = PlateOCR(confidence_threshold=0.3)  # Lower threshold for testing
    
    # Create a test image with text (simulated license plate)
    test_img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Add some text using OpenCV (simple simulation)
    cv2.putText(test_img, 'ABC123', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Testing OCR on simulated plate...")
    result = ocr.read_plate(test_img)
    
    if result:
        print(f"✅ OCR Result: {result['text']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Processing time: {result['processing_time_ms']:.1f}ms")
    else:
        print("❌ No text detected")
    
    # Print performance stats
    stats = ocr.get_performance_stats()
    print(f"📊 Performance: {stats}")

if __name__ == "__main__":
    test_plate_ocr() 