import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional, Tuple

class ObjectLearner:
    def __init__(self, data_file="custom_objects.json"):
        self.data_file = data_file
        self.known_objects = {} # name -> {histogram: list, features: ...}
        self.load_data()
        
    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    # Convert lists back to numpy arrays where needed
                    self.known_objects = data
                    logging.info(f"📚 Loaded {len(self.known_objects)} custom objects.")
            except Exception as e:
                logging.error(f"❌ Failed to load custom objects: {e}")
                self.known_objects = {}
        else:
            self.known_objects = {}

    def save_data(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.known_objects, f)
            logging.info("💾 Custom objects saved.")
        except Exception as e:
            logging.error(f"❌ Failed to save custom objects: {e}")

    def _get_center_crop(self, frame):
        """Gets the center square of the frame for learning."""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Crop size: 1/3 of the smallest dimension
        crop_size = min(h, w) // 3
        half_size = crop_size // 2
        
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(w, center_x + half_size)
        y2 = min(h, center_y + half_size)
        
        return frame[y1:y2, x1:x2]

    def _calculate_histogram(self, image):
        """Calculates color histogram as a simple fingerprint."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Calculate hist for H, S, V
        # 8 bins for Hue, 8 for Saturation, 8 for Value is efficient
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().tolist()

    def learn_object(self, frame, name: str) -> bool:
        """Learns the object in the center of the frame."""
        print(f"DEBUG: Learning object '{name}'...")
        crop = self._get_center_crop(frame)
        if crop.size == 0:
            logging.error("❌ Empty crop, cannot learn.")
            return False
            
        hist = self._calculate_histogram(crop)
        
        # Save to memory and disk
        self.known_objects[name] = {
            "histogram": hist,
            "learned_at": time.time()
        }
        self.save_data()
        logging.info(f"🎓 Learned new object: {name}")
        return True

    def identify_object(self, frame, threshold=0.65) -> Optional[str]:
        """
        Identifies if the center of the frame matches a known object.
        Returns: Name of object or None.
        """
        if not self.known_objects:
            return None
            
        crop = self._get_center_crop(frame)
        current_hist = self._calculate_histogram(crop)
        current_hist_np = np.array(current_hist, dtype=np.float32)
        
        best_match = None
        best_score = 0.0
        
        for name, data in self.known_objects.items():
            saved_hist = np.array(data["histogram"], dtype=np.float32)
            
            # Compare histograms using correlation
            score = cv2.compareHist(current_hist_np, saved_hist, cv2.HISTCMP_CORREL)
            
            if score > best_score:
                best_score = score
                best_match = name
                
        if best_score > threshold:
            logging.info(f"🔍 Custom object match: {best_match} ({best_score:.2f})")
            return best_match
            
        return None
import time
