#!/usr/bin/env python3
"""
Advanced License Plate Detection Module
Implements YOLOv8-based vehicle detection with specialized plate region extraction
"""

import cv2
import numpy as np
import time
import logging
from typing import List, Tuple, Optional, Dict, Any
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlateDetector:
    """
    Advanced license plate detector using YOLOv8 for vehicle detection
    and computer vision techniques for plate region extraction.
    """
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize the plate detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for vehicle detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.detection_times = []
        
        # Vehicle classes in YOLO (COCO dataset)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # Plate detection parameters
        self.plate_aspect_ratio_min = 2.0  # Width/height ratio
        self.plate_aspect_ratio_max = 6.0
        self.plate_area_min = 500  # Minimum pixel area
        self.plate_area_max = 15000  # Maximum pixel area
        
        logger.info("PlateDetector initialized")
    
    def load_model(self) -> YOLO:
        """Load YOLOv8 model with proper error handling."""
        if self.model is None:
            try:
                logger.info(f"Loading YOLOv8 model from {self.model_path}...")
                
                # Handle PyTorch 2.6+ compatibility issues
                import warnings
                warnings.filterwarnings('ignore', category=UserWarning)
                
                # Temporarily monkey-patch torch.load to use weights_only=False
                original_torch_load = torch.load
                torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, 'weights_only': False})
                
                try:
                    self.model = YOLO(self.model_path)
                    
                    # Warm up the model with a dummy inference to avoid first-frame latency
                    logger.info("Warming up model...")
                    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    _ = self.model(dummy_frame, verbose=False)
                    logger.info("Model warm-up completed")
                    
                finally:
                    # Restore original torch.load
                    torch.load = original_torch_load
                logger.info("YOLOv8 model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {e}")
                raise
        
        return self.model
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the frame using YOLOv8.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of vehicle detection dictionaries with bbox, confidence, class
        """
        model = self.load_model()
        
        # Resize frame for faster inference while maintaining aspect ratio
        original_height, original_width = frame.shape[:2]
        max_size = 640  # YOLOv8 optimal input size
        
        if max(original_height, original_width) > max_size:
            scale = max_size / max(original_height, original_width)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            scale_factor = 1.0 / scale
        else:
            resized_frame = frame
            scale_factor = 1.0
        
        # Run inference
        results = model(resized_frame, classes=self.vehicle_classes, conf=self.confidence_threshold, verbose=False)
        
        vehicles = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract detection data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Scale coordinates back to original frame size
                    x1, y1, x2, y2 = x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor
                    
                    vehicles.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': self._get_class_name(class_id)
                    })
        
        return vehicles
    
    def _get_class_name(self, class_id: int) -> str:
        """Convert YOLO class ID to readable name."""
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'vehicle')
    
    def extract_plate_regions(self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract potential license plate regions from a vehicle bounding box.
        
        Args:
            frame: Full image frame
            vehicle_bbox: Vehicle bounding box (x1, y1, x2, y2)
            
        Returns:
            List of tuples: (plate_crop, absolute_bbox)
        """
        x1, y1, x2, y2 = vehicle_bbox
        
        # Extract vehicle region with some padding
        padding = 10
        vehicle_crop = frame[max(0, y1-padding):min(frame.shape[0], y2+padding), 
                           max(0, x1-padding):min(frame.shape[1], x2+padding)]
        
        if vehicle_crop.size == 0:
            return []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        
        # Find potential plate regions using multiple methods
        plate_regions = []
        
        # Method 1: Edge detection + contour analysis
        plate_regions.extend(self._find_plates_by_edges(gray, vehicle_crop, x1-padding, y1-padding))
        
        # Method 2: Morphological operations
        plate_regions.extend(self._find_plates_by_morphology(gray, vehicle_crop, x1-padding, y1-padding))
        
        # Method 3: Template matching (if vehicle is front-facing)
        if self._is_front_facing_vehicle(vehicle_crop):
            plate_regions.extend(self._find_plates_by_position(gray, vehicle_crop, x1-padding, y1-padding))
        
        # Remove duplicates and filter by quality
        plate_regions = self._filter_and_deduplicate_regions(plate_regions)
        
        return plate_regions
    
    def _find_plates_by_edges(self, gray: np.ndarray, vehicle_crop: np.ndarray, offset_x: int, offset_y: int) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Find plate regions using edge detection and contour analysis."""
        plates = []
        
        try:
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio and size
                if self._is_valid_plate_region(w, h):
                    # Extract plate crop
                    plate_crop = vehicle_crop[y:y+h, x:x+w]
                    
                    if plate_crop.size > 0:
                        # Convert to absolute coordinates
                        abs_bbox = (offset_x + x, offset_y + y, offset_x + x + w, offset_y + y + h)
                        plates.append((plate_crop, abs_bbox))
                        
        except Exception as e:
            logger.warning(f"Edge detection failed: {e}")
        
        return plates
    
    def _find_plates_by_morphology(self, gray: np.ndarray, vehicle_crop: np.ndarray, offset_x: int, offset_y: int) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Find plate regions using morphological operations."""
        plates = []
        
        try:
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to highlight rectangular regions
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, rect_kernel)
            
            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                if self._is_valid_plate_region(w, h):
                    plate_crop = vehicle_crop[y:y+h, x:x+w]
                    
                    if plate_crop.size > 0:
                        abs_bbox = (offset_x + x, offset_y + y, offset_x + x + w, offset_y + y + h)
                        plates.append((plate_crop, abs_bbox))
                        
        except Exception as e:
            logger.warning(f"Morphological detection failed: {e}")
        
        return plates
    
    def _find_plates_by_position(self, gray: np.ndarray, vehicle_crop: np.ndarray, offset_x: int, offset_y: int) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Find plate regions based on typical position in front-facing vehicles."""
        plates = []
        
        try:
            h, w = gray.shape
            
            # Focus on lower portion of vehicle (typical plate location)
            roi_y_start = int(h * 0.6)  # Bottom 40% of vehicle
            roi_y_end = h
            roi_x_start = int(w * 0.2)  # Middle 60% horizontally
            roi_x_end = int(w * 0.8)
            
            roi = gray[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            
            if roi.size > 0:
                # Apply edge detection in ROI
                edges = cv2.Canny(roi, 30, 100)
                
                # Find horizontal line segments (plate characteristics)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=5)
                
                if lines is not None:
                    # Group nearby horizontal lines
                    horizontal_lines = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle = abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                        if angle < 15 or angle > 165:  # Nearly horizontal
                            horizontal_lines.append((x1, y1, x2, y2))
                    
                    if len(horizontal_lines) >= 2:  # At least 2 horizontal lines (top and bottom of plate)
                        # Create bounding box around lines
                        all_x = [x for line in horizontal_lines for x in [line[0], line[2]]]
                        all_y = [y for line in horizontal_lines for y in [line[1], line[3]]]
                        
                        if all_x and all_y:
                            min_x, max_x = min(all_x), max(all_x)
                            min_y, max_y = min(all_y), max(all_y)
                            
                            # Add padding
                            padding = 5
                            min_x = max(0, min_x - padding)
                            min_y = max(0, min_y - padding)
                            max_x = min(roi.shape[1], max_x + padding)
                            max_y = min(roi.shape[0], max_y + padding)
                            
                            w_plate = max_x - min_x
                            h_plate = max_y - min_y
                            
                            if self._is_valid_plate_region(w_plate, h_plate):
                                # Convert ROI coordinates to vehicle coordinates
                                x_vehicle = roi_x_start + min_x
                                y_vehicle = roi_y_start + min_y
                                
                                plate_crop = vehicle_crop[y_vehicle:y_vehicle+h_plate, x_vehicle:x_vehicle+w_plate]
                                
                                if plate_crop.size > 0:
                                    abs_bbox = (offset_x + x_vehicle, offset_y + y_vehicle, 
                                              offset_x + x_vehicle + w_plate, offset_y + y_vehicle + h_plate)
                                    plates.append((plate_crop, abs_bbox))
                        
        except Exception as e:
            logger.warning(f"Position-based detection failed: {e}")
        
        return plates
    
    def _is_front_facing_vehicle(self, vehicle_crop: np.ndarray) -> bool:
        """Determine if vehicle is likely front-facing based on aspect ratio."""
        h, w = vehicle_crop.shape[:2]
        aspect_ratio = w / h
        # Front-facing vehicles tend to be wider than tall
        return 1.2 <= aspect_ratio <= 3.0
    
    def _is_valid_plate_region(self, width: int, height: int) -> bool:
        """Check if dimensions are valid for a license plate."""
        if width <= 0 or height <= 0:
            return False
        
        aspect_ratio = width / height
        area = width * height
        
        return (self.plate_aspect_ratio_min <= aspect_ratio <= self.plate_aspect_ratio_max and
                self.plate_area_min <= area <= self.plate_area_max)
    
    def _filter_and_deduplicate_regions(self, regions: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Remove duplicate and poor quality regions."""
        if not regions:
            return []
        
        # Remove duplicates by checking overlap
        filtered = []
        for i, (crop1, bbox1) in enumerate(regions):
            is_duplicate = False
            
            for j, (crop2, bbox2) in enumerate(filtered):
                if self._calculate_overlap(bbox1, bbox2) > 0.5:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append((crop1, bbox1))
        
        # Sort by area (larger regions first) and limit to top candidates
        filtered.sort(key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]), reverse=True)
        
        return filtered[:3]  # Keep top 3 candidates per vehicle
    
    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return 0.0
        
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def detect_plates(self, frame: np.ndarray) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Main method to detect license plates in a frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of tuples: (plate_crop, detection_info)
        """
        start_time = time.time()
        
        # Detect vehicles first
        vehicles = self.detect_vehicles(frame)
        
        all_plates = []
        
        for vehicle in vehicles:
            try:
                # Extract plate regions from each vehicle
                plate_regions = self.extract_plate_regions(frame, vehicle['bbox'])
                
                for plate_crop, plate_bbox in plate_regions:
                    detection_info = {
                        'vehicle_bbox': vehicle['bbox'],
                        'vehicle_confidence': vehicle['confidence'],
                        'vehicle_class': vehicle['class_name'],
                        'plate_bbox': plate_bbox,
                        'plate_area': (plate_bbox[2] - plate_bbox[0]) * (plate_bbox[3] - plate_bbox[1])
                    }
                    
                    all_plates.append((plate_crop, detection_info))
                    
            except Exception as e:
                logger.warning(f"Failed to process vehicle {vehicle}: {e}")
        
        # Record detection time
        detection_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        self.detection_times.append(detection_time)
        
        # Keep only recent timing data
        if len(self.detection_times) > 100:
            self.detection_times = self.detection_times[-100:]
        
        return all_plates
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.detection_times:
            return {"avg_detection_time_ms": 0, "max_detection_time_ms": 0}
        
        return {
            "avg_detection_time_ms": np.mean(self.detection_times),
            "max_detection_time_ms": max(self.detection_times),
            "min_detection_time_ms": min(self.detection_times),
            "recent_detection_time_ms": self.detection_times[-1] if self.detection_times else 0
        }
    
    def draw_detections(self, frame: np.ndarray, vehicles: List[Dict[str, Any]], plates: List[Tuple[np.ndarray, Dict[str, Any]]]) -> np.ndarray:
        """
        Draw detection results on frame for visualization.
        
        Args:
            frame: Original frame
            vehicles: Vehicle detections
            plates: Plate detections
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        # Draw vehicle bounding boxes
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{vehicle['class_name']}: {vehicle['confidence']:.2f}"
            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw plate bounding boxes
        for _, detection_info in plates:
            x1, y1, x2, y2 = detection_info['plate_bbox']
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            label = "Plate"
            cv2.putText(result_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return result_frame


def test_plate_detector():
    """Test function for the plate detector."""
    import time
    from webcam_capture import WebcamCapture
    
    print("Testing PlateDetector...")
    
    try:
        detector = PlateDetector()
        
        # Test with webcam
        with WebcamCapture(fps=10) as webcam:  # Lower FPS for testing
            print("Starting 10-second test...")
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 10.0:
                frame = webcam.get_frame()
                if frame is not None:
                    # Detect plates
                    plates = detector.detect_plates(frame)
                    
                    frame_count += 1
                    if frame_count % 30 == 0:
                        stats = detector.get_performance_stats()
                        print(f"Frames processed: {frame_count}")
                        print(f"Plates detected: {len(plates)}")
                        print(f"Avg detection time: {stats['avg_detection_time_ms']:.1f}ms")
                        
                        if plates:
                            print("Sample detection info:")
                            for i, (crop, info) in enumerate(plates[:2]):
                                print(f"  Plate {i+1}: {crop.shape}, Vehicle: {info['vehicle_class']}")
            
            print("Test completed successfully!")
            final_stats = detector.get_performance_stats()
            print(f"Final performance: {final_stats}")
            
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_plate_detector() 